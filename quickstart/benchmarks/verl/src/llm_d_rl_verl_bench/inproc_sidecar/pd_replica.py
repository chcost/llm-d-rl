# Ported from verl/workers/rollout/vllm_rollout/llmd_pd_vllm_server.py
# (router-plugin-abstraction branch).
# Only change: model_label_for_epp import points to our local epp.endpoints.
from __future__ import annotations

import logging
import os
import socket
import subprocess
from typing import Any, Optional

import aiohttp
import ray

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

from llm_d_rl_common.endpoints import model_label as model_label_for_epp

logger = logging.getLogger(__name__)

# The sidecar is injected at runtime (fetch-sidecar initContainer on the worker)
# into /opt/llm-d-bins, the same way the EPP/Envoy binaries are on the head, so
# swapping it never rebuilds the verl image. The manifest sets VERL_SIDECAR_BINARY;
# the default mirrors the init container's extract path.
_SIDECAR_BINARY = os.environ.get(
    "VERL_SIDECAR_BINARY", "/opt/llm-d-bins/pd-sidecar"
)
_DEFAULT_NIXL_BASE_PORT = 5600
_VLLM_LOCAL_BIND_HOST = "127.0.0.1"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class PDPoolCoordinator:
    def __init__(self, config):
        disagg = config.disaggregation
        self.n_prefill: int = disagg.prefill_replicas
        self.n_decode: int = disagg.decode_replicas


class PDPrefillVLLMHttpServer(vLLMHttpServer):
    async def launch_server(self, master_address=None, master_port=None, dp_rpc_port=None):
        nixl_port = _DEFAULT_NIXL_BASE_PORT + self.replica_rank
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = self._server_address
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(nixl_port)
        os.environ.setdefault("UCX_TLS", "cuda_ipc,cuda_copy,tcp")
        vllm_log_level = os.environ.get("VERL_VLLM_LOG_LEVEL")
        if vllm_log_level:
            os.environ["VLLM_LOGGING_LEVEL"] = vllm_log_level
        await super().launch_server(
            master_address=master_address,
            master_port=master_port,
            dp_rpc_port=dp_rpc_port,
        )

    async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
        raise RuntimeError(
            "PDPrefillVLLMHttpServer.generate() must never be called directly. "
            "The llm-d sidecar on the decode node calls the prefill pod over HTTP."
        )


class PDDecodeVLLMHttpServer(vLLMHttpServer):
    _sidecar_session: aiohttp.ClientSession | None = None
    _completed_requests: int = 0

    async def get_completed_requests(self) -> int:
        count = self._completed_requests
        self._completed_requests = 0
        return count

    async def _get_sidecar_session(self) -> aiohttp.ClientSession:
        if self._sidecar_session is None or self._sidecar_session.closed:
            self._sidecar_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=36000),
            )
        return self._sidecar_session

    async def launch_server(self, master_address=None, master_port=None, dp_rpc_port=None):
        nixl_port = _DEFAULT_NIXL_BASE_PORT + self.replica_rank
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = self._server_address
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(nixl_port)
        os.environ.setdefault("UCX_TLS", "cuda_ipc,cuda_copy,tcp")
        vllm_log_level = os.environ.get("VERL_VLLM_LOG_LEVEL")
        if vllm_log_level:
            os.environ["VLLM_LOGGING_LEVEL"] = vllm_log_level
        node_ip = self._server_address
        self._server_address = _VLLM_LOCAL_BIND_HOST
        try:
            await super().launch_server(
                master_address=master_address,
                master_port=master_port,
                dp_rpc_port=dp_rpc_port,
            )
        finally:
            self._server_address = node_ip
        self._launch_sidecar()

    def _launch_sidecar(self) -> None:
        custom = self.config.get("custom") or {}
        connector = custom.get("sidecar_connector", "nixlv2")
        sidecar_log_level = os.environ.get("VERL_SIDECAR_LOG_LEVEL", "1")
        vllm_port = self._server_port
        self._sidecar_port = _find_free_port()
        cmd = [
            _SIDECAR_BINARY,
            f"--port={self._sidecar_port}",
            f"--vllm-port={vllm_port}",
            f"--kv-connector={connector}",
            "--secure-proxy=false",
            f"--zap-log-level={sidecar_log_level}",
        ]
        log_path = f"/tmp/sidecar-decode-{self.replica_rank}.log"
        logger.info("Launching llm-d routing sidecar: %s (log: %s)", " ".join(cmd), log_path)
        self._sidecar_log = open(log_path, "w")
        self._sidecar_process = subprocess.Popen(
            cmd, stdout=self._sidecar_log, stderr=subprocess.STDOUT
        )

    def get_server_address(self):
        assert self._server_port is not None, "server not launched"
        return self._server_address, self._sidecar_port

    async def get_global_steps(self) -> int | None:
        return self.global_steps

    def _prepare_sampling_params(self, sampling_params: dict, prompt_ids: list[int]) -> dict:
        params = {k: v for k, v in sampling_params.items() if v is not None}

        max_model_len = getattr(self.config, "max_model_len", None)
        if max_model_len is None:
            max_model_len = self.config.prompt_length + self.config.response_length

        max_possible_tokens = max_model_len - len(prompt_ids)
        if max_possible_tokens < 1:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) leaves no room to generate within "
                f"max_model_len={max_model_len}; need at least 1 token of headroom."
            )

        if "max_tokens" in params:
            max_tokens = params.pop("max_tokens")
        elif "max_new_tokens" in params:
            max_tokens = params.pop("max_new_tokens")
        else:
            max_tokens = min(
                self.config.response_length,
                self.config.prompt_length + self.config.response_length - len(prompt_ids),
            )
        params["max_tokens"] = max(1, min(max_tokens, max_possible_tokens))

        if params.pop("logprobs", False):
            params["logprobs"] = 0

        repetition_penalty = getattr(self.config, "repetition_penalty", None)
        params.setdefault("repetition_penalty", repetition_penalty if repetition_penalty is not None else 1.0)
        return params

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        sidecar_headers: Optional[dict] = None,
        **kwargs,
    ) -> TokenOutput:
        epp = sidecar_headers or {}
        path = epp.get(":path", "/inference/v1/generate")
        method = epp.get(":method", "POST").upper()
        url = f"http://localhost:{self._sidecar_port}{path}"

        body = {
            "model": model_label_for_epp(self.model_config),
            "token_ids": prompt_ids,
            "sampling_params": self._prepare_sampling_params(sampling_params, prompt_ids),
        }
        headers = {k: v for k, v in epp.items() if not k.startswith(":") and k.lower() != "content-length"}

        import asyncio as _asyncio

        session = await self._get_sidecar_session()
        try:
            async with session.request(method, url, json=body, headers=headers) as resp:
                if not resp.ok:
                    error_body = await resp.text()
                    raise RuntimeError(
                        f"Sidecar returned {resp.status}: {error_body} | "
                        f"sidecar_headers={list(headers.keys())} | "
                        f"sampling_params={body.get('sampling_params')}"
                    )
                data = await resp.json()
        except _asyncio.CancelledError:
            logger.error(
                "generate() task was CANCELLED mid-sidecar-request - "
                "this orphans NIXL blocks on the prefill. request_id=%s url=%s",
                request_id,
                url,
            )
            raise
        except Exception as e:
            logger.error(
                "generate() raised %s: %s - request_id=%s url=%s",
                type(e).__name__,
                e,
                request_id,
                url,
            )
            raise

        choices = data.get("choices") or []
        if choices:
            choice = choices[0]
            token_ids = [int(t) for t in (choice.get("token_ids") or [])]
            finish_reason = choice.get("finish_reason")
            logprobs_content = (choice.get("logprobs") or {}).get("content") or []
            log_probs = [e["logprob"] for e in logprobs_content] if logprobs_content else None
        else:
            token_ids, finish_reason, log_probs = [], None, None

        self._completed_requests += 1
        return TokenOutput(
            token_ids=token_ids,
            stop_reason=finish_reason,
            log_probs=log_probs,
            extra_fields={
                "global_steps": self.global_steps,
                # Connector-agnostic prefix-hit ground truth. Requires
                # engine_kwargs.vllm.enable_prompt_tokens_details=true.
                "cached_tokens": (
                    (data.get("usage") or {}).get("prompt_tokens_details") or {}
                ).get("cached_tokens"),
            },
        )


class PDPrefillEngineReplica(vLLMReplica):
    def __init__(self, replica_rank, config, model_config, gpus_per_node=8, **kwargs):
        super().__init__(replica_rank, config, model_config, gpus_per_node, **kwargs)
        self.server_class = ray.remote(PDPrefillVLLMHttpServer)
        self._engine_role = "prefill"

    async def launch_servers(self):
        await super().launch_servers()
        logger.info("Prefill engine %s ready at %s", self.replica_rank, self._server_address)


class PDDecodeEngineReplica(vLLMReplica):
    def __init__(self, replica_rank, config, model_config, gpus_per_node=8, **kwargs):
        super().__init__(replica_rank, config, model_config, gpus_per_node, **kwargs)
        self.server_class = ray.remote(PDDecodeVLLMHttpServer)
        self._engine_role = "decode"

    async def launch_servers(self):
        await super().launch_servers()
        logger.info("Decode engine %s ready at %s (sidecar)", self.replica_rank, self._server_address)


class PDServerAdapter(ServerAdapter):
    def _get_server_name_prefix(self) -> str:
        return "vllm_"


def PDEngineReplicaFactory(replica_rank, config, model_config, gpus_per_node=8, **kwargs):
    coordinator = PDPoolCoordinator(config)
    n_total = coordinator.n_prefill + coordinator.n_decode
    if replica_rank >= n_total:
        raise ValueError(
            f"replica_rank {replica_rank} is out of range: "
            f"prefill_replicas={coordinator.n_prefill} + decode_replicas={coordinator.n_decode} = {n_total}."
        )
    if replica_rank < coordinator.n_prefill:
        return PDPrefillEngineReplica(replica_rank, config, model_config, gpus_per_node, **kwargs)
    return PDDecodeEngineReplica(replica_rank, config, model_config, gpus_per_node, **kwargs)
