"""Dummy reward for the trace-player benchmark.

The weka replay measures rollout/scheduling wall-clock and the gen/train ratio,
not answer quality (there is no ground truth - prompts are synthetic token ids).
verl has no built-in reward for data_source='weka_cc_traces', so we point
custom_reward_function.path at this file (name=compute_score) to return a
constant. A constant reward gives a degenerate GRPO advantage, which is fine:
the training step still executes end to end for the timing measurement.
"""

from __future__ import annotations


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs) -> float:
    return 0.0
