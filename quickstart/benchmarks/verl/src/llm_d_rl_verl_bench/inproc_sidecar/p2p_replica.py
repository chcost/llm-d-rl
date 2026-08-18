# P2P KV-cache-sharing replica: symmetric (no prefill/decode split, unlike PD).
# Subclasses PDDecodeVLLMHttpServer and overrides only the sidecar launch
# (--kv-connector=offloading) and peer addressing.
#
# Each replica binds its P2P control socket on its own loopback IP
# (p2p_addressing.p2p_listener_host) with a flat port, so both dispatch paths
# address peers by host.
#
# VERL_P2P_NOSIDECAR skips the sidecar and calls verl's own generate(), which
# takes kv_transfer_params directly. NOTE: that path does not report
# cached_tokens, so per-request pull evidence is available only in sidecar mode.
#
# Requires kv_connector_extra_config spec_name=TieringOffloadingSpec and
# secondary_tiers=[{type:p2p}] (set by run_test.sh); without both there is no
# P2P tier and remote_kv_source is silently ignored.
#
# Design rationale: HANDOVER.md "CODE RATIONALE INDEX".
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Optional

import ray

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from llm_d_rl_verl_bench.inproc_sidecar.p2p_addressing import (
    DEFAULT_P2P_CONNECTOR_PORT,
    p2p_listener_host,
)
from llm_d_rl_verl_bench.inproc_sidecar.pd_replica import (
    _SIDECAR_BINARY,
    _find_free_port,
    PDDecodeVLLMHttpServer,
    PDServerAdapter as P2PServerAdapter,
)

logger = logging.getLogger(__name__)

class P2PVLLMHttpServer(PDDecodeVLLMHttpServer):
    async def launch_server(self, master_address=None, master_port=None, dp_rpc_port=None):
        # Bind vLLM's P2P control socket on this replica's own loopback IP.
        # admission.py derives the same address when naming a migration source.
        os.environ["VLLM_P2P_SIDE_CHANNEL_HOST"] = p2p_listener_host(self.replica_rank)
        os.environ["VLLM_P2P_SIDE_CHANNEL_PORT"] = str(self._p2p_listener_port())
        await super().launch_server(
            master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port,
        )

    def _p2p_listener_port(self) -> int:
        """Port vLLM's P2P tier binds on this replica. Flat across replicas."""
        return int(os.environ.get("VERL_P2P_CONNECTOR_PORT", DEFAULT_P2P_CONNECTOR_PORT))

    @staticmethod
    def _nosidecar() -> bool:
        # "enabled" is what run_test.sh sets; Ray's runtime_env.env_vars rejects
        # the bool Hydra would infer from a bare "true".
        return os.environ.get("VERL_P2P_NOSIDECAR", "false").strip().lower() in (
            "1", "true", "yes", "enabled",
        )

    def _launch_sidecar(self) -> None:
        if self._nosidecar():
            logger.info(
                "VERL_P2P_NOSIDECAR set: skipping sidecar launch for replica %s, "
                "dispatching directly to vLLM's native endpoint instead.",
                self.replica_rank,
            )
            self._sidecar_port = None
            return
        sidecar_log_level = os.environ.get("VERL_SIDECAR_LOG_LEVEL", "1")
        vllm_port = self._server_port
        self._sidecar_port = _find_free_port()
        # The sidecar names peers by host only: it keeps extractHost(source) from
        # our header and overwrites the port with this flat value.
        p2p_port = self._p2p_listener_port()
        cmd = [
            _SIDECAR_BINARY,
            f"--port={self._sidecar_port}",
            f"--vllm-port={vllm_port}",
            "--kv-connector=offloading",
            f"--p2p-connector-port={p2p_port}",
            "--secure-proxy=false",
            f"--zap-log-level={sidecar_log_level}",
        ]
        log_path = f"/tmp/sidecar-p2p-{self.replica_rank}.log"
        logger.info("Launching llm-d routing sidecar (P2P): %s (log: %s)", " ".join(cmd), log_path)
        self._sidecar_log = open(log_path, "w")
        # POD_IP lets the sidecar recognise itself as the source and skip the pull.
        sidecar_env = dict(os.environ)
        sidecar_env["POD_IP"] = self._server_address
        self._sidecar_process = subprocess.Popen(
            cmd, stdout=self._sidecar_log, stderr=subprocess.STDOUT, env=sidecar_env
        )

    def get_server_address(self):
        assert self._server_port is not None, "server not launched"
        if self._nosidecar():
            # No sidecar exists in this mode - register vLLM's own port as the
            # dispatch/discovery address instead (impacts anything that reads
            # this: write_rollout_endpoints()/EPP file-discovery, vllm_scrape.py).
            return self._server_address, self._server_port
        return self._server_address, self._sidecar_port

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        sidecar_headers: Optional[dict] = None,
        **kwargs,
    ) -> TokenOutput:
        if self._nosidecar():
            # verl's generate() takes kv_transfer_params, folds it into
            # sampling_params.extra_args and echoes vLLM's reply back in
            # extra_fields["kv_transfer_params"]. Call it directly, skipping
            # PDDecodeVLLMHttpServer.generate() which posts to the sidecar.
            # It reports no cached_tokens - see the module header.
            out = await vLLMHttpServer.generate(
                self, prompt_ids, sampling_params, request_id, **kwargs
            )
            self._completed_requests += 1
            return out
        return await super().generate(
            prompt_ids, sampling_params, request_id, sidecar_headers=sidecar_headers, **kwargs
        )


class P2PEngineReplica(vLLMReplica):
    def __init__(self, replica_rank, config, model_config, gpus_per_node=8, **kwargs):
        super().__init__(replica_rank, config, model_config, gpus_per_node, **kwargs)
        self.server_class = ray.remote(P2PVLLMHttpServer)
        self._engine_role = "p2p"

    async def launch_servers(self):
        await super().launch_servers()
        logger.info("P2P engine %s ready at %s (sidecar)", self.replica_rank, self._server_address)


def P2PEngineReplicaFactory(replica_rank, config, model_config, gpus_per_node=8, **kwargs):
    return P2PEngineReplica(replica_rank, config, model_config, gpus_per_node, **kwargs)
