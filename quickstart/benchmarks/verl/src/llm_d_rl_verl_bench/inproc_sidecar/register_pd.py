"""Rollout backend registration for llm-d PD disaggregated vLLM.

Import this module via verl's model.external_lib hook so both of verl's rollout
registries have the "vllm-llmd-pd" entry before anything looks it up:

  - _ROLLOUT_REGISTRY (verl.workers.rollout.base): consulted early, during FSDP
    worker model-config instantiation, before get_rollout_class().
  - RolloutReplicaRegistry (verl.workers.rollout.replica): consulted later, when
    TaskRunnerV1 launches the replica servers.

The loader imports pd_replica lazily (as verl's own built-in loaders do), so
importing an EPP mode never drags vLLM in.

In the run script:
    actor_rollout_ref.model.external_lib=llm_d_rl_verl_bench.inproc_sidecar.register_pd
"""

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import RolloutReplicaRegistry

_ROLLOUT_REGISTRY[("vllm-llmd-pd", "async")] = (
    "llm_d_rl_verl_bench.inproc_sidecar.pd_replica.PDServerAdapter"
)


def _load_llmd_pd():
    from llm_d_rl_verl_bench.inproc_sidecar.pd_replica import PDEngineReplicaFactory
    return PDEngineReplicaFactory


RolloutReplicaRegistry.register("vllm-llmd-pd", _load_llmd_pd)
