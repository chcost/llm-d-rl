"""AgentLoopManager for native (baseline) verl rollout with per-request logging.

This keeps verl's default rollout behaviour intact - default GlobalRequestLoadBalancer
routing, no EPP, no external gateway - and only swaps the stock LLMServerClient for
LoggingLLMClient so that a native run produces the same per-request reqlog as an EPP
run. It also writes the standard endpoints YAML so the vLLM /metrics scraper can run
against native replicas, giving a fully symmetric native-vs-EPP data set.

To use, set in the training YAML config:

    actor_rollout_ref:
      rollout:
        name: vllm
        agent:
          agent_loop_manager_class: llm_d_rl_verl_bench.native_logging.agent_loop_manager.NativeLoggingAgentLoopManager
        custom:
          epp_endpoints_file: /tmp/epp-endpoints.yaml   # optional, for the scraper
"""

from __future__ import annotations

import logging

from omegaconf import OmegaConf

from llm_d_rl_verl_integration.base_agent_loop_manager import LlmdBaseAgentLoopManager
from llm_d_rl_common.endpoints import write_rollout_endpoints
from llm_d_rl_verl_bench.native_logging.llm_client import LoggingLLMClient
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)

# Default matches what vllm_scrape.py reads and what EPP mode writes.
_DEFAULT_ENDPOINTS_FILE = "/tmp/epp-endpoints.yaml"


class NativeLoggingAgentLoopManager(LlmdBaseAgentLoopManager):
    """Native routing + reqlog. Writes endpoints YAML for the /metrics scraper."""

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        custom = OmegaConf.to_container(self.rollout_config.get("custom") or {}, resolve=True)
        endpoints_file = custom.get("epp_endpoints_file") or _DEFAULT_ENDPOINTS_FILE
        write_rollout_endpoints(endpoints_file, server_addresses, self.model_config)
        logger.info(
            "[NativeLoggingAgentLoopManager] native rollout, %d servers, endpoints -> %s",
            len(server_addresses),
            endpoints_file,
        )

    def _create_llm_client(self) -> LLMServerClient:
        # Reuse the original load balancer handle: routing stays identical to
        # stock native rollout; we only add logging.
        return LoggingLLMClient(
            config=self.config,
            load_balancer_handle=self.llm_client._load_balancer,
        )
