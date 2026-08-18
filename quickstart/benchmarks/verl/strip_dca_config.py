"""Strip `dual_chunk_attention_config` from a HuggingFace checkpoint's config.json.

This vLLM nightly registers no dual-chunk-attention backend, so the key makes
the engine die on an unsupported `layer_idx` kwarg before loading the model.
Safe to strip for <262144-token runs. Idempotent; keeps a one-time
`config.json.dca-backup`.

Usage:
    strip_dca_config.py <model_dir_or_config_json> [more...]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_KEY = "dual_chunk_attention_config"


def strip_one(target: Path) -> bool:
    """Strip the key from one model dir / config.json. True if the file changed."""
    config_path = target / "config.json" if target.is_dir() else target
    if not config_path.is_file():
        # MODEL_PATH may be a bare HF repo id, not a local dir.
        print(f"strip_dca_config: no config.json at {config_path} - skipping")
        return False

    with config_path.open() as fh:
        config = json.load(fh)

    if _KEY not in config:
        print(f"strip_dca_config: {config_path} already has no {_KEY} - no change")
        return False

    backup = config_path.with_suffix(".json.dca-backup")
    if not backup.exists():
        shutil.copy2(config_path, backup)
        print(f"strip_dca_config: saved original to {backup}")

    del config[_KEY]
    with config_path.open("w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    print(f"strip_dca_config: removed {_KEY} from {config_path}")
    return True


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for raw in argv:
        strip_one(Path(raw).expanduser())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
