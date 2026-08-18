"""The router stack as a Ray actor, pinned wherever the caller schedules it.

Ray is platform, not framework: nothing here knows about verl, vime or slime.
The actor takes explicit arguments rather than a framework's config object, so
mapping a framework's own config onto it stays in that framework's package.

Two ways to start the router stack ship in this package and this is one of them:

  * in-process, from inside the training job - this module (verl does that,
    pinning the actor to the head node where Ray's GCS runs).
  * as a foreground process from a pod lifecycle hook - the ``llm-d-rl-router``
    console script in cli.py (vime and slime do that).

Both drive the same RouterStack, so the EPP and Envoy argv cannot drift between
them. ``ray`` is an optional dependency of this package (the ``[ray]`` extra) and
is imported only here, so installing llm-d-rl-common does not require it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import ray

from llm_d_rl_common.endpoints import write_pd_endpoints, write_rollout_endpoints
from llm_d_rl_common.router_stack import (
    DEFAULT_ENVOY_PORT,
    DEFAULT_EPP_GRPC_PORT,
    DEFAULT_EPP_HEALTH_PORT,
    RouterStack,
)

logger = logging.getLogger(__name__)


@ray.remote
class RayRouter:
    """Writes the endpoints file, starts EPP, and optionally Envoy.

    Returns the address workers should talk to: EPP's gRPC address normally, or
    Envoy's when ``envoy_config`` is given.
    """

    def __init__(self) -> None:
        self._stack = RouterStack()

    async def start(
        self,
        *,
        server_addresses: list[str],
        epp_config: str,
        endpoints_file: Optional[str] = None,
        model_config: Optional[dict] = None,
        engine_type: str = "vllm",
        server_roles: Optional[list[Optional[str]]] = None,
        epp_grpc_port: int = DEFAULT_EPP_GRPC_PORT,
        epp_health_port: int = DEFAULT_EPP_HEALTH_PORT,
        epp_pool_name: str = "file-discovery",
        epp_pool_namespace: str = "default",
        envoy_config: Optional[str] = None,
        envoy_port: int = DEFAULT_ENVOY_PORT,
    ) -> str:
        # Endpoints are written on this node, co-located with EPP that reads them.
        if endpoints_file:
            if server_roles and any(r is not None for r in server_roles):
                write_pd_endpoints(endpoints_file, server_addresses, server_roles, model_config)
            else:
                write_rollout_endpoints(
                    endpoints_file, server_addresses, model_config, engine_type=engine_type
                )
            logger.info("[ray_router] wrote endpoints to %s", endpoints_file)

        grpc_port, health_port = await self._stack.start_epp(
            epp_config,
            grpc_port=int(epp_grpc_port),
            health_port=int(epp_health_port),
            pool_name=epp_pool_name,
            pool_namespace=epp_pool_namespace,
        )
        logger.info("[ray_router] EPP ready on grpc=%d health=%d", grpc_port, health_port)

        if envoy_config:
            port = await self._stack.start_envoy(envoy_config, port=int(envoy_port))
            logger.info("[ray_router] Envoy ready on :%d", port)
            return f"{ray.util.get_node_ip_address()}:{port}"

        return f"{ray.util.get_node_ip_address()}:{grpc_port}"

    async def stop(self) -> None:
        self._stack.stop()


def start_kwargs_from_custom(
    custom: dict[str, Any],
    *,
    server_addresses: list[str],
    model_config: Optional[dict] = None,
    engine_type: str = "vllm",
    server_roles: Optional[list[Optional[str]]] = None,
    with_envoy: bool = False,
) -> dict[str, Any]:
    """Map a ``rollout.custom.*``-style dict onto RayRouter.start()'s arguments.

    Kept here rather than in a framework package because the key names
    (epp_config_file, epp_endpoints_file, epp_grpc_port, ...) are this
    integration's own contract, not any one framework's.
    """
    if not custom.get("epp_config_file"):
        raise RuntimeError("rollout.custom.epp_config_file is required")
    kw: dict[str, Any] = {
        "server_addresses": server_addresses,
        "epp_config": custom["epp_config_file"],
        "endpoints_file": custom.get("epp_endpoints_file"),
        "model_config": model_config,
        "engine_type": engine_type,
        "server_roles": server_roles,
        "epp_grpc_port": int(custom.get("epp_grpc_port", DEFAULT_EPP_GRPC_PORT)),
        "epp_health_port": int(custom.get("epp_grpc_health_port", DEFAULT_EPP_HEALTH_PORT)),
        "epp_pool_name": custom.get("epp_pool_name", "file-discovery"),
        "epp_pool_namespace": custom.get("epp_pool_namespace", "default"),
    }
    if with_envoy:
        if not custom.get("envoy_config"):
            raise RuntimeError("rollout.custom.envoy_config is required")
        kw["envoy_config"] = custom["envoy_config"]
        kw["envoy_port"] = int(custom.get("envoy_port", DEFAULT_ENVOY_PORT))
    return kw
