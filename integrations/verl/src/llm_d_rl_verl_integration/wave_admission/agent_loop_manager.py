"""AgentLoopManager for wave-based admission control.

Set actor_rollout_ref.rollout.agent.agent_loop_manager_class to this class.
Tunables go under actor_rollout_ref.rollout.custom as wave_admission_*
(wave1_size, gpu_capacity_gb, gpu_util, weights_gb, kv_bytes_per_token,
initial_growth_guess, prior_weight, max_wait_s, poll_interval_s,
allow_migration, reserve_mode, reserve_z, migration_cost_ratio,
p2p_kv_available, p2p_port, migration_cost_ratio_p2p, p2p_nosidecar) plus
epp_endpoints_file.
"""

from __future__ import annotations

import logging
from typing import Any

import ray
from omegaconf import OmegaConf

from llm_d_rl_verl_integration.base_agent_loop_manager import LlmdBaseAgentLoopManager
from llm_d_rl_verl_integration.p2p_addressing import DEFAULT_P2P_CONNECTOR_PORT
from llm_d_rl_verl_integration.wave_admission.admission import (
    AdmissionLedger,
    compute_budget_tokens_per_replica,
)
from llm_d_rl_verl_integration.wave_admission.llm_client import WaveAdmissionLLMClient
from llm_d_rl_common.endpoints import write_rollout_endpoints
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)

# Default matches what vllm_scrape.py reads and what EPP/native modes write.
_DEFAULT_ENDPOINTS_FILE = "/tmp/epp-endpoints.yaml"


def _custom_get(custom: dict[str, Any], key: str, default: Any) -> Any:
    val = custom.get(key, default)
    return default if val is None else val


class WaveAdmissionAgentLoopManager(LlmdBaseAgentLoopManager):
    """Wave-based admission control, sticky-after-admit, no migration (Phase 1)."""

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        custom = OmegaConf.to_container(self.rollout_config.get("custom") or {}, resolve=True)
        endpoints_file = _custom_get(custom, "epp_endpoints_file", _DEFAULT_ENDPOINTS_FILE)
        write_rollout_endpoints(endpoints_file, server_addresses, self.model_config)

        # vLLMReplica names replica i's server actor "vllm_server_{i}_0".
        self._address_to_handle = {}
        for i, addr in enumerate(server_addresses):
            actor_name = f"vllm_server_{i}_0"
            try:
                self._address_to_handle[addr] = ray.get_actor(actor_name)
            except ValueError:
                raise RuntimeError(
                    f"Could not find Ray actor {actor_name!r} for server {addr}. "
                    "Make sure the rollout backend is vllm and servers are started."
                )

        budget = compute_budget_tokens_per_replica(
            gpu_capacity_gb=_custom_get(custom, "wave_admission_gpu_capacity_gb", None),
            gpu_memory_utilization=_custom_get(custom, "wave_admission_gpu_util", None),
            weights_gb=_custom_get(custom, "wave_admission_weights_gb", None),
            bytes_per_token=_custom_get(custom, "wave_admission_kv_bytes_per_token", None),
        )
        wave1_size = int(_custom_get(custom, "wave_admission_wave1_size", 128))
        initial_growth_guess = float(_custom_get(custom, "wave_admission_initial_growth_guess", 100_000.0))
        prior_weight = float(_custom_get(custom, "wave_admission_prior_weight", 15.0))
        max_wait_s = float(_custom_get(custom, "wave_admission_max_wait_s", 60.0))
        poll_interval_s = float(_custom_get(custom, "wave_admission_poll_interval_s", 0.5))
        allow_reactive_migration = bool(_custom_get(custom, "wave_admission_allow_migration", True))
        reserve_mode = str(_custom_get(custom, "wave_admission_reserve_mode", "size"))
        reserve_z = float(_custom_get(custom, "wave_admission_reserve_z", 1.5))
        migration_cost_ratio = float(_custom_get(custom, "wave_admission_migration_cost_ratio", 1.0))
        p2p_kv_available = bool(_custom_get(custom, "wave_admission_p2p_kv_available", False))
        p2p_port = int(_custom_get(custom, "wave_admission_p2p_port", DEFAULT_P2P_CONNECTOR_PORT))
        migration_cost_ratio_p2p = float(_custom_get(custom, "wave_admission_migration_cost_ratio_p2p", 0.0))
        p2p_nosidecar = bool(_custom_get(custom, "wave_admission_p2p_nosidecar", False))
        oracle_reserve = bool(_custom_get(custom, "wave_admission_oracle_reserve", False))
        lpt_window_s = float(_custom_get(custom, "wave_admission_lpt_window_s", 0.0))
        rebalance_slack = float(_custom_get(custom, "wave_admission_rebalance_slack", -1.0))
        # Read by _create_llm_client() below, which has no access to `custom`.
        self._p2p_nosidecar = p2p_nosidecar

        logger.info(
            "[WaveAdmissionAgentLoopManager] %d replicas, budget=%.0f tok/replica, "
            "wave1_size=%d, initial_growth_guess=%.0f, prior_weight=%.1f, max_wait_s=%.0f, "
            "allow_reactive_migration=%s, reserve_mode=%s, reserve_z=%.1f, migration_cost_ratio=%.1f, "
            "p2p_kv_available=%s, migration_cost_ratio_p2p=%.2f, p2p_nosidecar=%s, p2p_port=%d",
            len(server_addresses), budget, wave1_size, initial_growth_guess, prior_weight, max_wait_s,
            allow_reactive_migration, reserve_mode, reserve_z, migration_cost_ratio,
            p2p_kv_available, migration_cost_ratio_p2p, p2p_nosidecar, p2p_port,
        )

        # One long-lived ledger actor, pinned to the head, shared fleet-wide.
        self._admission_ledger = AdmissionLedger.options(
            scheduling_strategy=self.head_node_strategy()
        ).remote(
            replicas=server_addresses,
            budget_tokens_per_replica=budget,
            wave1_size=wave1_size,
            initial_growth_guess=initial_growth_guess,
            prior_weight=prior_weight,
            max_wait_s=max_wait_s,
            poll_interval_s=poll_interval_s,
            allow_reactive_migration=allow_reactive_migration,
            reserve_mode=reserve_mode,
            reserve_z=reserve_z,
            migration_cost_ratio=migration_cost_ratio,
            p2p_kv_available=p2p_kv_available,
            p2p_port=p2p_port,
            migration_cost_ratio_p2p=migration_cost_ratio_p2p,
            p2p_nosidecar=p2p_nosidecar,
            oracle_reserve=oracle_reserve,
            lpt_window_s=lpt_window_s,
            rebalance_slack=rebalance_slack,
        )

    def _create_llm_client(self) -> LLMServerClient:
        return WaveAdmissionLLMClient(
            config=self.config,
            load_balancer_handle=self.llm_client._load_balancer,
            address_to_handle=self._address_to_handle,
            admission_ledger=self._admission_ledger,
            p2p_nosidecar=self._p2p_nosidecar,
        )
