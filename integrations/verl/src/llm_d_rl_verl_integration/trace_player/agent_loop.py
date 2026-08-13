"""Router-agnostic trace player agent loop.

Registered as ``agent_name="trace_player"``. It replays a normalized
conversation trace (per-turn input/output token counts + preceding off-GPU gaps)
through ``self.server_manager.generate(...)`` and nothing else, so it runs
UNCHANGED under any routing mode: native (verl's GlobalRequestLoadBalancer) or
EPP, with or without completion reporting (``custom.epp_report_completion``).
The router is selected purely by which AgentLoopManager
(``agent_loop_manager_class``) installs ``self.server_manager``; this loop
never references it.

Per turn it synthesizes deterministic prefix-nesting prompt_ids of length
``input_tokens`` and forces exactly ``output_tokens`` of decode
(``max_tokens = min_tokens = out, ignore_eos = True``) so per-turn GPU load
matches the trace without needing the original prompt text. Inter-turn gaps are
injected as non-blocking ``asyncio.sleep`` so other concurrent conversations keep
the GPU busy during a gap (the simulator's gap-overlap). Reward is dummy - this
workload measures scheduling/rollout wall-clock, not answer quality.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from llm_d_rl_verl_integration.trace_player.trace import nested_prompt_ids, parse_trace_turns

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Hard cap on injected inter-turn gap (seconds). The trace's raw gaps span
# tool-exec / human-think idle that can be minutes; cap so a single trajectory's
# client-side sleeps stay bounded. Overridable via env for GAP A/B (0 vs 10).
_GAP_CAP_S = float(os.getenv("VERL_TRACE_GAP_CAP_S", "10"))


@register("trace_player")
class TracePlayerAgentLoop(AgentLoopBase):
    """Replays a conversation trace as a sequence of forced-length generations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    def _get_trace(self, kwargs: dict[str, Any]):
        extra = kwargs.get("extra_info")
        raw = kwargs.get("trace_turns")
        if raw is None and extra is not None:
            # extra_info arrives as a per-row dict from non_tensor_batch.
            raw = extra["trace_turns"] if "trace_turns" in extra else extra.get("trace_turns")
        trace = parse_trace_turns(raw)
        # If the payload was a bare turns list (no embedded conv_id), take conv_id
        # from extra_info so each conversation gets distinct synthetic token ids.
        if trace.conv_id == "conv" and extra is not None:
            cid = extra["conv_id"] if "conv_id" in extra else extra.get("conv_id")
            if cid:
                trace.conv_id = str(cid)
        return trace

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        trace = self._get_trace(kwargs)

        # ONE stable request_id for the whole trajectory (all turns), unique per
        # rollout. Native LB uses it for sticky-session affinity (a conversation
        # pins to one replica across turns); EPP-with-completion uses it as the
        # x-request-id / PluginState key. The uuid suffix keeps the G GRPO
        # rollouts of one sample distinct so they pin independently.
        request_id = f"trace-{trace.conv_id}-{uuid4().hex[:8]}"

        # Trace-derived size hints for admission. Capped exactly as the turn loop
        # caps below, so the hint matches what will really be sent. Passed only
        # when the manager asks for them: every other client forwards **kwargs to
        # verl's generate(), which declares no **kwargs and would raise.
        hint: dict[str, Any] = {}
        if getattr(self.server_manager, "wants_session_hint", False) and trace.turns:
            _last = trace.turns[-1]
            hint = {
                "session_final": float(
                    min(_last.input_tokens, self.prompt_length)
                    + min(_last.output_tokens, self.response_length)
                ),
                "session_work": float(sum(
                    min(t.input_tokens, self.prompt_length)
                    + min(t.output_tokens, self.response_length)
                    for t in trace.turns
                )),
            }

        metrics: dict[str, Any] = {}
        last_prompt_ids: list[int] = [0]
        last_output: TokenOutput | None = None
        num_turns = 0

        with simple_timer("generate_sequences", metrics):
            for turn in trace.turns:
                if turn.pre_gap_s > 0:
                    # Non-blocking: other conversations keep the GPU busy during
                    # this conversation's off-GPU gap.
                    await asyncio.sleep(min(turn.pre_gap_s, _GAP_CAP_S))

                prompt_ids = nested_prompt_ids(trace.conv_id, turn.input_tokens)
                if len(prompt_ids) > self.prompt_length:
                    prompt_ids = prompt_ids[: self.prompt_length]

                out_len = max(1, min(turn.output_tokens, self.response_length))
                sp = dict(sampling_params)
                sp.update({"max_tokens": out_len, "min_tokens": out_len, "ignore_eos": True})

                last_output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sp,
                    **hint,
                )
                last_prompt_ids = prompt_ids
                num_turns += 1

        # Optional hook: server_managers that gate/track admission per
        # trajectory (e.g. WaveAdmissionLLMClient) release their ledger entry
        # here. Native/EPP clients don't define this, so this is a no-op for
        # every other mode.
        done_hook = getattr(self.server_manager, "on_trajectory_done", None)
        if done_hook is not None:
            await done_hook(request_id)

        # Return the last turn's (bounded) prompt+response as the trajectory
        # tensors. Reward is dummy, so a single well-formed turn keeps the
        # training-step tensors valid and small; the point of the run is the
        # rollout wall-clock and the gen/train ratio, which come from executing
        # every turn's generate() above.
        response_ids = list(last_output.token_ids) if last_output is not None else [0]
        response_ids = response_ids[: self.response_length]
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=last_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields={"turn_scores": [], "tool_rewards": [], "conv_id": trace.conv_id},
        )
        return output
