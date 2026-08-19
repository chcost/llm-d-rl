"""AgentLoopManager that starts EPP + Envoy as a Ray actor and routes via Envoy.

LlmdActor is pinned to the head node (where GCS runs).

YAML config (no verl code changes needed):
    actor_rollout_ref:
      rollout:
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.llmd_serving.agent_loop_manager.LlmdAgentLoopManager
        custom:
          epp_config_file: /path/to/config.yaml
          epp_endpoints_file: /tmp/epp-endpoints.yaml
          envoy_config: /path/to/envoy.yaml    # required
          # envoy_port: 8081                   # optional
"""

from __future__ import annotations

import logging

import ray
from omegaconf import OmegaConf

from llm_d_rl_verl_integration.base_agent_loop_manager import LlmdBaseAgentLoopManager
from llm_d_rl_verl_integration.llmd_actor import LlmdActor, start_kwargs
from llm_d_rl_verl_integration.llmd_serving.llm_client import EnvoyLLMClient
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)


class LlmdAgentLoopManager(LlmdBaseAgentLoopManager):
    """Starts EPP + Envoy via a Ray actor pinned to the head node, then routes via Envoy."""

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        rollout_cfg = self.rollout_config

        self._stack_actor = LlmdActor.options(
            scheduling_strategy=self.head_node_strategy()
        ).remote()

        pd_mode = getattr(rollout_cfg, "name", None) == "vllm-llmd-pd"
        server_roles = self.infer_roles(server_addresses, rollout_cfg) if pd_mode else None

        self._envoy_address = ray.get(
            self._stack_actor.start.remote(**start_kwargs(
                OmegaConf.to_container(rollout_cfg, resolve=True),
                server_addresses=server_addresses,
                model_config=OmegaConf.to_container(self.model_config, resolve=True),
                server_roles=server_roles,
                with_envoy=True,
            ))
        )
        logger.info("[LlmdAgentLoopManager] Envoy ready at %s", self._envoy_address)

    def _create_llm_client(self) -> LLMServerClient:
        return EnvoyLLMClient(
            config=self.config,
            load_balancer_handle=self.llm_client._load_balancer,
            envoy_address=self._envoy_address,
            model_name=self.model_config.path,
        )
