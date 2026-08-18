"""AgentLoopManager that launches EPP and routes via gRPC ext-proc.

To use, set in the training YAML config:

  Non-PD (standard vllm):
    actor_rollout_ref:
      rollout:
        name: vllm
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager
        custom:
          epp_config_file: /path/to/epp-config.yaml
          epp_endpoints_file: /tmp/epp-endpoints.yaml
          epp_grpc_port: 9002      # optional, default 9002
          epp_report_completion: true  # optional, default false - keep the ext_proc
                                        # stream open through generation and report
                                        # completion, so EPP's in-flight counter is
                                        # honest (needed for active-request-scorer /
                                        # a concurrency cap; adds one held-open stream
                                        # + 2 extra ext_proc messages per request)

  PD disaggregated (llm-d vllm):
    actor_rollout_ref:
      rollout:
        name: vllm-llmd-pd          # needs model.external_lib=...register_pd
        disaggregation:
          prefill_replicas: 2       # do NOT set enabled=True (avoids NotImplementedError from verl)
          decode_replicas: 2
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager
        custom:
          epp_config_file: /path/to/epp-config.yaml
          epp_endpoints_file: /tmp/epp-endpoints.yaml
          sidecar_connector: nixlv2

  P2P KV-cache sharing (llm-d vllm, aggregated - every replica both pulls and serves):
    actor_rollout_ref:
      rollout:
        name: vllm-llmd-p2p         # needs model.external_lib=...register_p2p
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager
        custom:
          epp_config_file: /path/to/epp-config-p2p.yaml
          epp_endpoints_file: /tmp/epp-endpoints.yaml
"""

from __future__ import annotations

import logging

import ray
from omegaconf import OmegaConf

from llm_d_rl_verl_integration.base_agent_loop_manager import LlmdBaseAgentLoopManager
from llm_d_rl_verl_integration.llmd_actor import LlmdActor, start_kwargs
from llm_d_rl_verl_integration.llmd_epp.llm_client import EPPLLMClient
from verl.workers.rollout.llm_server import LLMServerClient

# No pd_replica / p2p_replica import here: both import vLLM at module scope, which
# would make this module (and any subclass of it, e.g. the SGLang variant) need
# vLLM installed. register_pd.py / register_p2p.py own those registrations and
# import them lazily; both are loaded via model.external_lib for PD/P2P runs.

logger = logging.getLogger(__name__)


class LlmdRouterAgentLoopManager(LlmdBaseAgentLoopManager):
    """Launches EPP subprocess (via a Ray actor) and swaps in EPPLLMClient.

    Server actor handles are looked up by Ray actor name using the convention
    established by vLLMReplica.launch_servers(): ``"vllm_server_{rank}_0"``.
    server_addresses[i] from GlobalRequestLoadBalancer corresponds to
    replica_rank i (insertion order is preserved).

    A different rollout engine behind the same EPP routing overrides the four class
    attributes below and nothing else - see llmd_epp_sglang.
    """

    #: Ray actor-name prefix its replica class registers ("<prefix>_{rank}_0").
    server_actor_prefix = "vllm_server"
    #: Written to each endpoints-YAML entry; picks EPP's Prometheus metric mapping.
    engine_type = "vllm"
    #: Extra LlmdActor.options() for the EPP actor (SGLang needs num_cpus=0).
    epp_actor_options: dict = {}
    #: Client class swapped in for the workers.
    client_cls = EPPLLMClient

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        rollout_cfg = self.rollout_config

        # Detect PD mode by backend name (not disaggregation.enabled, which we
        # intentionally leave False to avoid verl's sglang-only guard). P2P mode
        # is detected the same way; unlike PD it has no role split, so it never
        # calls infer_roles() and server_roles stays None (LlmdActor.start() then
        # takes the plain write_rollout_endpoints() path, same as non-sidecar EPP).
        self._pd_mode = rollout_cfg.name == "vllm-llmd-pd"
        self._p2p_mode = rollout_cfg.name == "vllm-llmd-p2p"

        server_roles = None
        if self._pd_mode:
            server_roles = self.infer_roles(server_addresses, rollout_cfg)

        # Model name for EPP / generate body.
        self._model_name = self.model_config.path

        # Build address -> actor handle map. server_addresses[i] is the address for
        # replica_rank i, and the replica class names its node-0 server actor
        # "<server_actor_prefix>_{i}_0".
        self._address_to_handle = {}
        for i, addr in enumerate(server_addresses):
            actor_name = f"{self.server_actor_prefix}_{i}_0"
            try:
                self._address_to_handle[addr] = ray.get_actor(actor_name)
            except ValueError:
                raise RuntimeError(
                    f"Could not find Ray actor {actor_name!r} for server {addr}. "
                    f"Make sure the rollout backend matches {self.engine_type!r} and servers are started."
                )
        logger.info("[%s] address -> handle map: %s",
                    type(self).__name__, list(self._address_to_handle.keys()))

        # Launch EPP via a Ray actor pinned to the head node.
        epp_actor = LlmdActor.options(
            scheduling_strategy=self.head_node_strategy(), **self.epp_actor_options
        ).remote()

        self._grpc_addr = ray.get(
            epp_actor.start.remote(**start_kwargs(
                OmegaConf.to_container(rollout_cfg, resolve=True),
                server_addresses=server_addresses,
                model_config=OmegaConf.to_container(self.model_config, resolve=True),
                server_roles=server_roles,
                engine_type=self.engine_type,
            ))
        )
        self._epp_actor = epp_actor
        logger.info("[%s] EPP ready at %s", type(self).__name__, self._grpc_addr)

    def _create_llm_client(self) -> LLMServerClient:
        custom = OmegaConf.to_container(self.rollout_config.get("custom") or {}, resolve=True)
        return self.client_cls(
            config=self.config,
            load_balancer_handle=self.llm_client._load_balancer,
            grpc_addr=self._grpc_addr,
            address_to_handle=self._address_to_handle,
            model_name=self._model_name,
            # Both PD and P2P dispatch to a Ray actor whose generate() HTTP-calls a
            # local sidecar and needs EPP's response headers forwarded to build that
            # request; plain (non-sidecar) EPP mode calls the vLLM actor directly and
            # ignores sidecar_headers regardless of this flag.
            use_sidecar=self._pd_mode or self._p2p_mode,
            report_completion=bool(custom.get("epp_report_completion", False)),
        )


