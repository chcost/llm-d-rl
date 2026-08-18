"""Resolve a mode name into the verl overrides that turn the integration on.

The matrix lives in modes.yaml beside this file, so it ships with the package and
an adopter running their own launcher gets the same contract the benchmark driver
uses. See that file for what each mode means.

    llm-d-rl-verl-overrides --list
    llm-d-rl-verl-overrides epp
    llm-d-rl-verl-overrides epp-p2p --epp-config-dir /etc/llmd-configs
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

MODES_FILE = Path(__file__).resolve().parent / "modes.yaml"
DEFAULT_CONFIG_DIR = "/etc/llmd-configs"

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: str, env: dict[str, str]) -> str:
    """Expand ${VAR} and ${VAR:-default} from `env`, like the shell did."""
    def one(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        got = env.get(name)
        if got:
            return got
        if default is not None:
            return default
        raise KeyError(f"{name} is referenced by modes.yaml but not set")
    return _ENV_RE.sub(one, value)


def _flatten(items: Any) -> list[str]:
    """modes.yaml lets a hydra list nest (a YAML anchor expands to a list)."""
    out: list[str] = []
    if items is None:
        return out
    if isinstance(items, str):
        return [items]
    for it in items:
        out.extend(_flatten(it))
    return out


def load(path: Path = MODES_FILE) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def mode_names(spec: dict | None = None) -> list[str]:
    return list((spec or load())["modes"])


def overrides(
    mode: str,
    *,
    spec: dict | None = None,
    env: dict[str, str] | None = None,
    config_dir: str = DEFAULT_CONFIG_DIR,
    allow_manual: bool = False,
) -> list[str]:
    """Hydra overrides for `mode`, in the order the driver has always emitted."""
    spec = spec or load()
    env = dict(os.environ if env is None else env)
    modes = spec["modes"]
    if mode not in modes:
        raise KeyError(f"unknown mode {mode!r}; known: {', '.join(modes)}")
    m = modes[mode]
    if m.get("status") == "manual-only" and not allow_manual:
        raise KeyError(
            f"mode {mode!r} is manual-only - see modes.yaml for what is missing. "
            f"Pass --allow-manual to render its overrides anyway."
        )

    args: list[str] = list(spec.get("common", {}).get("hydra", []))
    args.append(
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={m['manager']}"
    )

    if m.get("epp_config"):
        # EPP_CONFIG overrides the mode's default variant without editing the mode.
        variant = env.get("EPP_CONFIG") or m["epp_config"]
        args.append(
            f"+actor_rollout_ref.rollout.custom.epp_config_file={config_dir}/{variant}"
        )
        for name in spec.get("epp_binary_env", []):
            if env.get(name):
                args.append(
                    f"+ray_kwargs.ray_init.runtime_env.env_vars.{name}={env[name]}"
                )

    if m.get("envoy_config"):
        args.append(
            f"+actor_rollout_ref.rollout.custom.envoy_config={config_dir}/{m['envoy_config']}"
        )

    if m.get("rollout_name"):
        args.append(f"actor_rollout_ref.rollout.name={m['rollout_name']}")
    if m.get("external_lib"):
        args.append(f"+actor_rollout_ref.model.external_lib={m['external_lib']}")

    args.extend(_flatten(m.get("hydra")))

    for cond in m.get("when", []):
        if env.get(cond["env"], "") == cond["equals"]:
            args.extend(_flatten(cond.get("hydra")))

    args.extend(spec.get("common", {}).get("hydra_last", []))
    return [_expand(a, env) for a in args]


def markdown(spec: dict | None = None, config_dir: str = DEFAULT_CONFIG_DIR) -> str:
    """The override reference, generated from the same data that emits the args.

    Hand-written tables drifted from the driver in both directions; generating
    them makes that impossible.
    """
    spec = spec or load()
    out = [
        "# verl integration: override reference",
        "",
        "GENERATED from `src/llm_d_rl_verl_integration/modes.yaml` by",
        "`llm-d-rl-verl-overrides --markdown`. Edit the YAML, not this file.",
        "",
        "Each mode below is a set of Hydra overrides you add to your own verl launch",
        "command. Nothing else is required - no patched verl, and no part of the",
        "quickstart. `${VAR:-default}` is read from the environment.",
        "",
        f"EPP config paths assume the configs are mounted at `{config_dir}`; pass",
        "`--epp-config-dir` to change that.",
        "",
    ]
    for name, m in spec["modes"].items():
        status = m.get("status")
        out.append(f"## `--mode {name}`" + (f"  ({status})" if status else ""))
        out.append("")
        out.append(m["summary"] + ".")
        out.append("")
        if status == "manual-only":
            out.append("> Not wired into a launcher. See modes.yaml for what is missing.")
            out.append("")
        try:
            args = overrides(name, spec=spec, env={}, config_dir=config_dir, allow_manual=True)
        except KeyError as e:  # pragma: no cover
            out.append(f"> could not render: {e}")
            out.append("")
            continue
        out.append("```bash")
        out.extend(f"{a} \\" for a in args[:-1])
        out.append(args[-1])
        out.append("```")
        out.append("")
        opt = []
        if m.get("epp_config"):
            opt.append(f"`EPP_CONFIG` selects a different EPP variant (default `{m['epp_config']}`).")
        for cond in m.get("when", []):
            opt.append(f"`{cond['env']}={cond['equals']}` adds {len(_flatten(cond.get('hydra')))} more override(s).")
        if opt:
            out.append("Environment knobs:")
            out.extend(f"- {o}" for o in opt)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", help="mode name; omit with --list")
    ap.add_argument("--list", action="store_true", help="print every mode and its summary")
    ap.add_argument("--markdown", action="store_true",
                    help="print the override reference as markdown (docs/configuration.md)")
    ap.add_argument("--epp-config-dir", default=DEFAULT_CONFIG_DIR,
                    help=f"where the EPP configs are mounted (default {DEFAULT_CONFIG_DIR})")
    ap.add_argument("--modes-file", type=Path, default=MODES_FILE)
    ap.add_argument("--sep", default="\n", help="separator between overrides (default newline)")
    ap.add_argument("--allow-manual", action="store_true",
                    help="render a mode marked manual-only (it is not wired into a launcher)")
    args = ap.parse_args(argv)
    spec = load(args.modes_file)

    if args.markdown:
        sys.stdout.write(markdown(spec, args.epp_config_dir))
        return 0

    if args.list:
        for name, m in spec["modes"].items():
            status = f"  [{m['status']}]" if m.get("status") else ""
            print(f"{name}{status}\n    {m['summary']}")
        return 0

    if not args.mode:
        ap.error("give a mode name, or --list")
    try:
        out = overrides(args.mode, spec=spec, config_dir=args.epp_config_dir,
                        allow_manual=args.allow_manual)
    except KeyError as e:
        print(str(e).strip("'"), file=sys.stderr)
        return 2
    sys.stdout.write(args.sep.join(out) + ("\n" if args.sep == "\n" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
