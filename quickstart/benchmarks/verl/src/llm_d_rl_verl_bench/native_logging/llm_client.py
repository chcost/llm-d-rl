"""LLMServerClient that keeps verl's native routing but logs each request.

This is the native-mode twin of llmd_epp.EPPLLMClient: it does NOT change
routing at all - it uses verl's stock GlobalRequestLoadBalancer (least in-flight)
via ``_acquire_server``/``_release_server`` exactly as the base LLMServerClient
does. The only addition is a per-request JSONL record (endpoint, timings, token
counts) written to VERL_REQLOG_DIR, byte-compatible with the EPP reqlog so the
same analysis tooling works for both modes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
from uuid import uuid4

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

from llm_d_rl_common.reqlog import log_request, open_reqlog, phash

logger = logging.getLogger(__name__)


class LoggingLLMClient(LLMServerClient):
    """Native verl routing + per-request reqlog.

    Routing is unchanged from the base class: each request is dispatched to the
    server chosen by the GlobalRequestLoadBalancer, and released in a finally
    block. We only measure and record; behaviour matches stock native rollout.
    """

    def __setstate__(self, state):
        # File handles do not pickle; the reqlog is (re)opened on the worker
        # process after unpickling, same as EPPLLMClient.
        self.__dict__.update(state)
        self._reqlog_f = open_reqlog()
        # Per-trajectory turn counter, keyed by the (stable) incoming request_id.
        # ToolAgentLoop reuses one request_id across all turns of a trajectory, so
        # this yields a 0-based turn index for multi-turn agentic rollouts (always 0
        # for single-turn tasks). Not GC'd: bounded by trajectory count per run, and
        # the client is recreated each run.
        self._turn_counts: dict[str, int] = {}

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        # Same acquire/generate/release flow as the base LLMServerClient; we only
        # wrap it with timing and a reqlog write. server_id is the endpoint addr.
        t0 = time.monotonic()
        server_id, server = await self._acquire_server(request_id)
        t_pick = time.monotonic()
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            # priority is only supported by vLLM rollout server.
            priority = kwargs.pop("priority", 0)
            priority_kwargs = (
                {"priority": priority}
                if priority != 0 and self.config.actor_rollout_ref.rollout.name == "vllm"
                else {}
            )
            output: TokenOutput = await server.generate.remote(
                request_id=uuid4().hex,  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **priority_kwargs,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            t_end = time.monotonic()

            try:
                ntok = len(output.token_ids) if getattr(output, "token_ids", None) is not None else None
            except Exception:
                ntok = None
            rid = str(request_id)
            turn = self._turn_counts.get(rid, 0)
            self._turn_counts[rid] = turn + 1
            log_request(self._reqlog_f, {
                "ts": time.time(),
                "request_id": rid,
                "turn": turn,
                "endpoint": server_id,
                "prompt_hash": phash(prompt_ids),
                "prompt_tokens": len(prompt_ids),
                "output_tokens": ntok,
                "pick_s": round(t_pick - t0, 5),
                "gen_s": round(t_end - t_pick, 5),
            })
            return output
        finally:
            self._release_server(server_id)
