#!/usr/bin/env bash
# Provision verl on one Ray node.   usage: verl.sh <head|worker>
#
# Everything framework-specific about a verl cluster is in this file: the clone,
# the editable install, and the model prefetch. The manifest knows only which
# image to run and that it should run this script.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ROLE="${1:?usage: verl.sh <head|worker>}"
: "${VERL_COMMIT:?not set - declared in integrations/verl/environments.env}"
ENGINE_PY_MODULE="${ENGINE_PY_MODULE:-vllm}"
# Shared across nodes via the PVC, so every node installs the SAME commit rather
# than cloning independently and possibly landing on different ones.
VERL_SRC="${VERL_SRC:-/tmp/verl/code/verl}"
PREFETCH_MODEL="${PREFETCH_MODEL:-Qwen/Qwen3-4B}"
MODEL_DIR="/tmp/verl/models/${PREFETCH_MODEL##*/}"

llmd_pep668

# --- verl itself -------------------------------------------------------------
# The head clones or fetches into the shared PVC path; workers install from what
# is already there. deploy.sh orders the two steps.
if [[ "$ROLE" == "head" ]]; then
  if [[ -d "$VERL_SRC/.git" ]]; then
    llmd_log "fetching verl in $VERL_SRC"
    git -C "$VERL_SRC" fetch --quiet origin
  else
    llmd_log "cloning verl into $VERL_SRC"
    mkdir -p "$(dirname "$VERL_SRC")"
    git clone --quiet https://github.com/volcengine/verl.git "$VERL_SRC"
  fi
  git -C "$VERL_SRC" checkout --quiet "$VERL_COMMIT"
fi
[[ -d "$VERL_SRC" ]] || llmd_fatal "$VERL_SRC does not exist - run the head step first"
RESOLVED="$(git -C "$VERL_SRC" rev-parse HEAD)"
llmd_log "verl at $RESOLVED (requested $VERL_COMMIT)"
[[ "$RESOLVED" == "$VERL_COMMIT"* ]] || llmd_log "NOTE: resolved SHA differs from the requested ref (branch or tag)"
pip install --no-deps -e "$VERL_SRC" >/dev/null
llmd_require_module verl

# --- the integration ---------------------------------------------------------
llmd_install common
llmd_require_module llm_d_rl_common
llmd_install verl
llmd_require_module llm_d_rl_verl_integration

# --- the benchmark harness ----------------------------------------------------
# Separate package (llm-d-rl-verl-bench) holding the research modes, the native
# control arm, the trace player and the in-process sidecar stand-ins. Provisioning
# installs it because this cluster exists to run benchmarks; an adopter installing
# only the integration never pulls it. LLMD_BENCH=0 skips it.
if [[ "${LLMD_BENCH:-1}" != "0" ]]; then
  llmd_install_path="quickstart/benchmarks/verl"
  if [[ "${LLMD_SOURCE:-git}" == "local" ]]; then
    pip install --no-deps --no-cache-dir --force-reinstall "$LLMD_LOCAL_SRC/$llmd_install_path"
  else
    pip install --no-deps --no-cache-dir --force-reinstall \
      "git+$LLMD_REPO@$LLMD_REPO_REF#subdirectory=$llmd_install_path"
  fi
  llmd_require_module llm_d_rl_verl_bench
fi

# --- the engine the image is supposed to provide -----------------------------
llmd_require_module "$ENGINE_PY_MODULE"
eval "MIN=\${ENGINE_${ENGINE_PY_MODULE}_MIN_VERSION:-}"
llmd_require_version "$ENGINE_PY_MODULE" "$MIN"

# --- model prefetch ----------------------------------------------------------
# One shared PVC directory, so this runs once from the head rather than racing
# between nodes - which is why the flock the postStart hook needed is gone.
mkdir -p /tmp/verl/hf_cache/datasets
if [[ "$ROLE" == "head" ]]; then
  llmd_log "prefetching $PREFETCH_MODEL into $MODEL_DIR"
  python3 - "$PREFETCH_MODEL" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], local_dir_use_symlinks=False)
PY
fi

llmd_write_marker verl "$RESOLVED" "$ENGINE_PY_MODULE" "$ROLE"
llmd_log "verl provisioning complete on $ROLE"
