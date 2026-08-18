"""verl's binding to the shared Ray router actor.

The actor and the ``rollout.custom.*`` mapping both live in
``llm_d_rl_common.ray_router`` - Ray is platform, not framework, and the custom.*
key names are this integration's contract rather than verl's. What is left here
is one verl-specific fact: PD mode is spelled ``rollout.name == "vllm-llmd-pd"``.

``LlmdActor`` stays the name the managers import, so nothing downstream changed.
"""

from __future__ import annotations

from typing import Any, Optional

from llm_d_rl_common.ray_router import RayRouter as LlmdActor
from llm_d_rl_common.ray_router import start_kwargs_from_custom

PD_ROLLOUT_NAME = "vllm-llmd-pd"

__all__ = ["LlmdActor", "PD_ROLLOUT_NAME", "start_kwargs"]


def start_kwargs(
    rollout_config: dict,
    *,
    server_addresses: list[str],
    model_config: Optional[dict] = None,
    engine_type: str = "vllm",
    server_roles: Optional[list[Optional[str]]] = None,
    with_envoy: bool = False,
) -> dict[str, Any]:
    """Arguments for ``LlmdActor.start.remote()`` from a verl rollout config."""
    roles = server_roles
    if rollout_config.get("name") != PD_ROLLOUT_NAME:
        # Only PD writes role-tagged endpoints; anything else writes a flat list
        # even if the caller inferred roles.
        roles = None
    return start_kwargs_from_custom(
        rollout_config.get("custom") or {},
        server_addresses=server_addresses,
        model_config=model_config,
        engine_type=engine_type,
        server_roles=roles,
        with_envoy=with_envoy,
    )
