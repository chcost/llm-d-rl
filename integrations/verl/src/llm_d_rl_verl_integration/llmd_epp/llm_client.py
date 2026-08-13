"""LLMServerClient that routes via EPP gRPC, then delegates inference to the
chosen vLLM actor handle exactly as original verl does.

Plain EPP:  EPP picks endpoint → call actor.generate.remote() → vLLM handles it.
PD / P2P:   EPP picks endpoint + sidecar headers → call actor.generate.remote(sidecar_headers=...)
            → PDDecodeVLLMHttpServer / P2PVLLMHttpServer.generate() → HTTP to local sidecar.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import ray

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

from llm_d_rl_common.reqlog import log_request, open_reqlog, phash, tag_global_steps

logger = logging.getLogger(__name__)


class EPPLLMClient(LLMServerClient):
    """Routes each request through EPP gRPC to pick a server, then calls
    that server's Ray actor directly — same as original verl flow.

    Args:
        config: verl DictConfig.
        load_balancer_handle: original GlobalRequestLoadBalancer (kept for
            compatibility but not used for routing decisions).
        grpc_addr: EPP gRPC address (``host:port``).
        address_to_handle: ``{server_address: ray_actor_handle}`` map built
            at startup. server_address must match what EPP returns as the
            ``x-gateway-destination-endpoint`` header.
        model_name: model identifier sent in the EPP request body.
        use_sidecar: if True, forward sidecar_headers returned by EPP to
            actor.generate.remote() so the actor's local sidecar (PD's
            PDDecodeVLLMHttpServer or P2P's P2PVLLMHttpServer) can reach it.
        report_completion: passed to ``EPPGrpcClient.route()`` as
            ``track_completion``. If True, EPP's in-flight counter reflects the
            real generation window instead of the pick-time snapshot. Needed for
            an active-request-scorer or a per-endpoint
            concurrency cap (flow control) to bind on anything meaningful;
            adds one held-open stream per in-flight request plus two extra
            ext_proc messages per request, so leave it off unless the EPP config
            actually consumes the in-flight signal.
    """

    def __init__(
        self,
        config,
        load_balancer_handle=None,
        *,
        grpc_addr: str,
        address_to_handle: dict[str, ray.actor.ActorHandle],
        model_name: str,
        use_sidecar: bool = False,
        report_completion: bool = False,
        **kwargs,
    ):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._grpc_addr = grpc_addr
        self._address_to_handle = address_to_handle
        self._model_name = model_name
        self._use_sidecar = use_sidecar
        self._report_completion = report_completion
        self._epp_client = None  # created on workers after unpickling via __setstate__

    def __setstate__(self, state):
        self.__dict__.update(state)
        from llm_d_rl_common.epp_grpc_client import EPPGrpcClient
        self._epp_client = EPPGrpcClient(self._grpc_addr)
        self._reqlog_f = open_reqlog()
        # Per-trajectory turn counter keyed by the (stable) incoming request_id;
        # see NativeLogging twin. 0-based turn index per trajectory (0 for single-turn).
        self._turn_counts: dict[str, int] = {}

    def _actor_kwargs(self, sidecar_headers, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extra kwargs for actor.generate.remote() - the one backend-specific seam.

        vLLM: forward EPP's sidecar headers when the actor has a local sidecar (PD /
        P2P). Nothing else is forwarded: vLLMHttpServer.generate() declares an
        explicit parameter list with no **kwargs, so an unknown key raises.
        """
        if self._use_sidecar and sidecar_headers:
            return {"sidecar_headers": sidecar_headers}
        return {}

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        **kwargs,
    ) -> TokenOutput:
        t0 = time.monotonic()

        result = await self._epp_client.route(
            self._model_name, prompt_ids, str(request_id),
            track_completion=self._report_completion,
        )
        endpoint, sidecar_headers = result.endpoint, result.sidecar_headers
        t_pick = time.monotonic()

        if endpoint is None:
            await result.complete(0)
            raise RuntimeError(f"EPP returned no endpoint for request {request_id}")

        actor = self._address_to_handle.get(endpoint)
        if actor is None:
            await result.complete(0)
            raise RuntimeError(
                f"EPP returned endpoint {endpoint!r} which is not in the known server map. "
                f"Known: {list(self._address_to_handle.keys())}"
            )

        extra_kwargs = self._actor_kwargs(sidecar_headers, kwargs)

        out = None
        try:
            out = await actor.generate.remote(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                **extra_kwargs,
            )
            # verl's _compute_metrics int()s min/max_global_steps on every tag, so a
            # missing key becomes None and kills the run. The server sets only
            # global_steps; a single-turn generate spans one weight version.
            tag_global_steps(out)
            return out
        finally:
            t_end = time.monotonic()
            try:
                ntok = len(out.token_ids) if out is not None and getattr(out, "token_ids", None) is not None else 0
            except Exception:  # noqa: BLE001
                ntok = 0
            await result.complete(ntok)
            rid = str(request_id)
            turn = self._turn_counts.get(rid, 0)
            self._turn_counts[rid] = turn + 1
            log_request(self._reqlog_f, {
                "ts": time.time(),
                "request_id": rid,
                "turn": turn,
                "endpoint": endpoint,
                "prompt_hash": phash(prompt_ids),
                "prompt_tokens": len(prompt_ids),
                "output_tokens": ntok,
                "pick_s": round(t_pick - t0, 5),
                "gen_s": round(t_end - t_pick, 5),
            })
