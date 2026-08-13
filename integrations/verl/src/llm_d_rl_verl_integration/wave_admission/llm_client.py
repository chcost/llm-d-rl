"""LLMServerClient that routes each turn via the AdmissionLedger actor."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
from uuid import uuid4

import ray

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

from llm_d_rl_common.reqlog import log_request, open_reqlog, phash, tag_global_steps

logger = logging.getLogger(__name__)


def _forced_output_len(sampling_params: dict[str, Any]) -> int:
    # trace_player forces max_tokens == min_tokens == out_len, ignore_eos=True,
    # so the output length is known deterministically before generation.
    for key in ("max_tokens", "min_tokens"):
        val = sampling_params.get(key)
        if val:
            return int(val)
    return 0


class WaveAdmissionLLMClient(LLMServerClient):
    """Asks the ledger for a replica, then dispatches to that server actor."""

    # Tells trace_player it may pass session_final/session_work. Other clients
    # leave this false, so the hints never reach verl's generate().
    wants_session_hint = True

    def __init__(
        self,
        config,
        load_balancer_handle=None,
        *,
        address_to_handle: dict[str, ray.actor.ActorHandle],
        admission_ledger: ray.actor.ActorHandle,
        p2p_nosidecar: bool = False,
        **kwargs,
    ):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._address_to_handle = address_to_handle
        self._admission_ledger = admission_ledger
        # When set, build kv_transfer_params directly instead of a sidecar header.
        self._p2p_nosidecar = p2p_nosidecar

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reqlog_f = open_reqlog()
        # Per-trajectory turn counter, keyed by incoming request_id.
        self._turn_counts: dict[str, int] = {}

    async def on_trajectory_done(self, request_id) -> None:
        """Optional hook called by TracePlayerAgentLoop after a trajectory's
        last turn. Releases this conversation's ledger entry and feeds the
        causal growth estimator. Native/EPP clients don't define this method,
        so the loop's ``getattr(..., None)`` check is a no-op for them.
        """
        await self._admission_ledger.on_trajectory_done.remote(str(request_id))

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
        t0 = time.monotonic()
        rid = str(request_id)
        turn = self._turn_counts.get(rid, 0)
        context_size = float(len(prompt_ids) + _forced_output_len(sampling_params))

        placement = await self._admission_ledger.acquire.remote(
            rid, turn_index=turn, context_size=context_size,
            session_final=kwargs.pop("session_final", None),
            session_work=kwargs.pop("session_work", None),
        )
        replica = placement["replica"]
        kv_source = placement.get("kv_source")
        t_pick = time.monotonic()
        actor = self._address_to_handle[replica]

        multimodal_kwargs = {}
        if audio_data is not None:
            multimodal_kwargs["audio_data"] = audio_data
        if mm_processor_kwargs:
            multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        # Set only on a migration with p2p_kv_available: names the replica the
        # KV was resident on so the destination pulls it. Passed only when set -
        # the non-P2P server's generate() has no such parameter.
        if kv_source:
            if self._p2p_nosidecar:
                # Shape expected by vLLM's P2P manager (_parse_source).
                host, _, port = kv_source.rpartition(":")
                kwargs["kv_transfer_params"] = {
                    "remote_kv_source": {
                        "remote_host": host,
                        "remote_port": int(port),
                        "kv_request_id": uuid4().hex,
                    }
                }
            else:
                kwargs["sidecar_headers"] = {"x-kv-cache-source-host-port": kv_source}

        out = None
        try:
            out = await actor.generate.remote(
                request_id=uuid4().hex,  # use a new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **kwargs,
            )
            # verl's _compute_metrics int()s min/max_global_steps on every tag, so a
            # missing key becomes None and kills the run after the first rollout.
            tag_global_steps(out)
            return out
        finally:
            t_end = time.monotonic()
            try:
                ntok = len(out.token_ids) if out is not None and getattr(out, "token_ids", None) is not None else 0
            except Exception:  # noqa: BLE001
                ntok = 0
            actual_context_size = float(len(prompt_ids) + ntok)
            await self._admission_ledger.record_turn.remote(
                rid, turn_index=turn, context_size=actual_context_size,
            )
            self._turn_counts[rid] = turn + 1
            log_request(self._reqlog_f, {
                "ts": time.time(),
                "request_id": rid,
                "turn": turn,
                "endpoint": replica,
                "prompt_hash": phash(prompt_ids),
                "prompt_tokens": len(prompt_ids),
                "output_tokens": ntok,
                "pick_s": round(t_pick - t0, 5),
                "gen_s": round(t_end - t_pick, 5),
                # Non-null iff the ledger migrated this turn.
                "kv_source": kv_source,
                # vLLM's echo of kv_transfer_params (p2p_nosidecar mode only).
                # verl's generate() names this key "kv_transfer_params"; the
                # reqlog field keeps its old name so analyze_arm.py still reads it.
                "kv_transfer_params_response": (
                    getattr(out, "extra_fields", None) or {}
                ).get("kv_transfer_params") if out is not None else None,
                # Connector-agnostic prefix-hit ground truth. Sidecar mode only -
                # verl's generate() does not report it, so this is null on the
                # nosidecar path and pull rate must be read from a sidecar arm.
                "cached_tokens": (
                    getattr(out, "extra_fields", None) or {}
                ).get("cached_tokens") if out is not None else None,
            })
