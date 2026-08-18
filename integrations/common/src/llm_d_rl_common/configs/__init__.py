"""Shipped llm-d routing configs (EPP scorer profiles, Envoy listeners).

These are data, not code: they are mounted into a pod or passed to EPP/Envoy by
path. They live inside the package so ``pip install llm-d-rl-common`` is enough
to get them - an adopter with their own Ray and training framework needs no
checkout of this repository.

    from llm_d_rl_common import configs
    configs.path("envoy.yaml")            # -> PosixPath(.../configs/envoy.yaml)
    configs.path("epp/base.yaml")         # subdirectories work the same way
    configs.names()                       # what is available
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def path(name: str) -> Path:
    """Absolute path to a shipped config file. Raises if it is not there."""
    p = _ROOT / name
    if not p.is_file():
        raise FileNotFoundError(f"no shipped config {name!r} (looked in {_ROOT})")
    return p


def root() -> Path:
    """Directory holding the shipped configs."""
    return _ROOT


def names() -> list[str]:
    """Every shipped config, as paths relative to root()."""
    return sorted(
        str(p.relative_to(_ROOT))
        for p in _ROOT.rglob("*")
        if p.is_file() and p.suffix in (".yaml", ".yml", ".env")
    )
