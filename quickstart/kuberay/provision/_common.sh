#!/usr/bin/env bash
# Shared provisioning helpers, sourced by provision/<framework>.sh on a Ray node.
#
# Provisioning runs AFTER the cluster is up, not from a postStart hook. That is
# deliberate:
#   * changing a framework ref no longer needs a pod recreate, which on a shared
#     cluster risks losing the GPU allocation to a contending namespace;
#   * re-running it installs a local working tree (LLMD_SOURCE=local), replacing
#     the habit of kubectl cp'ing patched files into site-packages;
#   * a failure is an exit code on your terminal, not a restart loop with the
#     reason buried in a log file inside the pod.
#
# The cost is that pod Ready no longer implies "framework installed". That is
# replaced by a provenance marker every script writes and `deploy.sh check`
# verifies across the whole cluster - which is strictly more than before, when
# nothing detected a worker that came back from a restart with no framework.

set -euo pipefail

LLMD_REPO="${LLMD_REPO:-https://github.com/llm-d-incubation/llm-d-rl.git}"
LLMD_REPO_REF="${LLMD_REPO_REF:-main}"
# Where kubectl cp puts a local checkout when LLMD_SOURCE=local.
LLMD_LOCAL_SRC="${LLMD_LOCAL_SRC:-/tmp/llmd-src}"
LLMD_MARKER="${LLMD_MARKER:-/tmp/llmd-provisioned.json}"

llmd_log() { echo "[provision] $*"; }
llmd_fatal() { echo "[provision] FATAL: $*" >&2; exit 1; }

# PEP 668: Ubuntu 24.04 refuses system-wide pip installs without this. The vLLM
# image sets it in its own Dockerfile; the stock verlai/verl:sgl* image does not.
llmd_pep668() { export PIP_BREAK_SYSTEM_PACKAGES=1; }

# Install one integration package. --no-deps on purpose: the environment image
# already carries the framework's dependency closure, and letting pip resolve it
# would try to move torch.
llmd_install() {
  local subdir="$1"
  if [[ "${LLMD_SOURCE:-git}" == "local" ]]; then
    local path="$LLMD_LOCAL_SRC/integrations/$subdir"
    [[ -d "$path" ]] || llmd_fatal "LLMD_SOURCE=local but $path is missing"
    llmd_log "installing $subdir from local tree $path"
    pip install --no-deps --no-cache-dir --force-reinstall "$path"
  else
    llmd_log "installing $subdir from $LLMD_REPO@$LLMD_REPO_REF"
    pip install --no-deps --no-cache-dir --force-reinstall \
      "git+$LLMD_REPO@$LLMD_REPO_REF#subdirectory=integrations/$subdir"
  fi
}

# Turn a silent empty install into a hard failure.
llmd_require_module() {
  local mod="$1"
  python3 -c "import $mod" 2>/dev/null || llmd_fatal "$mod is not importable after install"
  llmd_log "$mod importable"
}

llmd_require_command() {
  command -v "$1" >/dev/null || llmd_fatal "$1 is not on PATH after install"
  llmd_log "$1 on PATH"
}

llmd_module_version() {
  python3 -c "import $1,sys; print(getattr($1,'__version__','unknown'))" 2>/dev/null || echo unknown
}

# Assert the image satisfies what the framework declared it needs. This is the
# check that was missing when a verl pin declared vllm<=0.12.0 against an image
# shipping 0.26.1rc1 and nothing complained, because the install is --no-deps.
llmd_require_version() {
  local mod="$1" min="${2:-}"
  local got; got="$(llmd_module_version "$mod")"
  if [[ -z "$min" ]]; then
    llmd_log "$mod $got (no minimum declared)"
    return 0
  fi
  if [[ "$got" == unknown ]]; then
    llmd_fatal "$mod has no __version__; cannot check against declared minimum $min"
  fi
  python3 - "$got" "$min" <<'PY' || llmd_fatal "$mod $got is older than the declared minimum $min"
import sys, re
def key(v):
    return [int(x) for x in re.findall(r"\d+", v)[:3]] or [0]
sys.exit(0 if key(sys.argv[1]) >= key(sys.argv[2]) else 1)
PY
  llmd_log "$mod $got satisfies declared minimum $min"
}

# Provenance, not a done-flag: deploy.sh check compares these across every pod
# and refuses to launch a run when they disagree or one is missing.
llmd_write_marker() {
  local framework="$1" framework_ref="$2" engine="$3" role="$4"
  local integration_source engine_version
  if [[ "${LLMD_SOURCE:-git}" == "local" ]]; then integration_source="local"; else integration_source="$LLMD_REPO_REF"; fi
  engine_version="$(llmd_module_version "$engine")"
  python3 - "$LLMD_MARKER" "$framework" "$framework_ref" "$integration_source" \
            "$engine" "$engine_version" "$role" <<'PY'
import json, sys, datetime
path, framework, fref, isrc, engine, eversion, role = sys.argv[1:8]
json.dump({
    "framework": framework,
    "framework_ref": fref,
    "integration_source": isrc,
    "engine": engine,
    "engine_version": eversion,
    "node_role": role,
    "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}, open(path, "w"), indent=2, sort_keys=True)
print(f"[provision] wrote {path}")
PY
}
