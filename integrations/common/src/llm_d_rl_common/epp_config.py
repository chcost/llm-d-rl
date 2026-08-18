"""Compose an EPP config from a chassis, one routing profile and any modifiers.

The shipped configs used to be nine near-identical files (mean pairwise
similarity 71.4%, two pairs above 98%), so a plugin-level change had to be
hand-repeated up to nine times with nothing to catch a miss. They are now
base.yaml + profiles/<one> + modifiers/<any>, listed per variant in
variants.yaml, merged here.

Two merge rules, which is all this schema needs:

  * maps merge recursively; a later layer wins on scalar leaves.
  * lists of objects carrying an identity key ("name", else "pluginRef") merge
    BY THAT KEY - an item whose key exists is deep-merged into it, an item whose
    key is new is appended. Every list in this schema qualifies.
  * any other list (featureGates) is replaced by the later layer.

The by-key rule is the whole point: modifiers/spread.yaml is three lines that
change maxPerReplica and leave the burst producer's other five parameters alone.
kustomize cannot do this for an unregistered CRD - strategic-merge-patch takes
its list merge keys from Go struct tags or an OpenAPI x-kubernetes-patch-merge-key,
and with no schema it treats lists as atomic, so the same overlay would drop
every other plugin.

${...} placeholders (EPP_PARSER) survive the merge as plain scalars and are
substituted afterwards, the way deploy.sh already does it.

CLI:
    llm-d-rl-epp-config list
    llm-d-rl-epp-config render epp-config-p2p-spread.yaml [-o out.yaml]
    llm-d-rl-epp-config render-all <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
EPP_DIR = _HERE / "configs" / "epp"

_IDENTITY_KEYS = ("name", "pluginRef")


def _identity(item: Any) -> str | None:
    """The key an object in a list is merged on, or None if it has none."""
    if not isinstance(item, dict):
        return None
    for k in _IDENTITY_KEYS:
        if k in item:
            return f"{k}={item[k]}"
    return None


def _merge(base: Any, over: Any) -> Any:
    """Merge `over` onto `base` per the rules in the module docstring."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(out[k], v) if k in out else v
        return out

    if isinstance(base, list) and isinstance(over, list):
        # Merge by identity only when BOTH sides are wholly identity-bearing;
        # a plain scalar list (featureGates) is replaced instead.
        if base and over and all(_identity(i) for i in base + over):
            out = list(base)
            index = {_identity(i): n for n, i in enumerate(out)}
            for item in over:
                key = _identity(item)
                if key in index:
                    out[index[key]] = _merge(out[index[key]], item)
                else:
                    index[key] = len(out)
                    out.append(item)
            return out
        return over

    return over


def _load(path: Path) -> dict:
    with path.open() as f:
        doc = yaml.safe_load(f)
    if doc is None:
        raise ValueError(f"{path} is empty")
    return doc


def layers(variant: str, epp_dir: Path = EPP_DIR) -> list[Path]:
    """Files that compose `variant`, in merge order, starting with the chassis."""
    spec = _load(epp_dir / "variants.yaml")["variants"]
    if variant not in spec:
        raise KeyError(
            f"unknown variant {variant!r}; known: {', '.join(sorted(spec))}"
        )
    out = [epp_dir / "base.yaml"]
    for name in spec[variant]:
        profile, modifier = epp_dir / "profiles" / f"{name}.yaml", epp_dir / "modifiers" / f"{name}.yaml"
        if profile.is_file():
            out.append(profile)
        elif modifier.is_file():
            out.append(modifier)
        else:
            raise FileNotFoundError(
                f"variant {variant!r} names {name!r}, which is neither "
                f"profiles/{name}.yaml nor modifiers/{name}.yaml"
            )
    return out


def render(variant: str, epp_dir: Path = EPP_DIR) -> str:
    """The merged config for `variant`, as YAML text with a provenance header."""
    paths = layers(variant, epp_dir)
    doc: dict = {}
    for p in paths:
        doc = _merge(doc, _load(p))
    provenance = " + ".join(p.relative_to(epp_dir).as_posix() for p in paths)
    header = (
        f"# GENERATED - do not edit.\n"
        f"# {variant} = {provenance}\n"
        f"# Edit those and re-render: llm-d-rl-epp-config render {variant}\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def variant_names(epp_dir: Path = EPP_DIR) -> list[str]:
    return sorted(_load(epp_dir / "variants.yaml")["variants"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epp-dir", type=Path, default=EPP_DIR,
                    help="directory holding base.yaml, profiles/, modifiers/, variants.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print every known variant")
    r = sub.add_parser("render", help="print one merged variant")
    r.add_argument("variant")
    r.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    ra = sub.add_parser("render-all", help="write every variant into a directory")
    ra.add_argument("outdir", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for v in variant_names(args.epp_dir):
            print(f"{v}: {' + '.join(p.relative_to(args.epp_dir).as_posix() for p in layers(v, args.epp_dir))}")
        return 0

    if args.cmd == "render":
        text = render(args.variant, args.epp_dir)
        if args.output:
            args.output.write_text(text)
        else:
            sys.stdout.write(text)
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    for v in variant_names(args.epp_dir):
        (args.outdir / v).write_text(render(v, args.epp_dir))
        print(f"wrote {args.outdir / v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
