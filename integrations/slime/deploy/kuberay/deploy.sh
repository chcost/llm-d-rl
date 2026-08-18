#!/usr/bin/env bash
# Deploy (or tear down) the slime KubeRay cluster using the single config in
# deploy.env. Renders ray-cluster.yaml.tmpl with the image refs and namespace,
# builds the llmd-epp-configs-slime ConfigMap from the configs shipped in llm-d-rl-common (envoy + burst
# EPP with EPP_PARSER=sglanghttp-parser) plus this directory's run script.
#
# Usage:
#   ./deploy.sh                  # create ConfigMap + apply the cluster
#   ./deploy.sh apply            # same
#   ./deploy.sh delete           # delete the cluster (leaves the ConfigMap)
#   ./deploy.sh configmap        # (re)create the ConfigMap only
#   ./deploy.sh render           # print the rendered manifest to stdout (no kubectl)
#
# Requires: envsubst (GNU gettext) and kubectl on PATH.
set -euo pipefail

ACTION="${1:-apply}"
cd "$(dirname "$0")"

# The routing stack's own versions ship beside the configs they belong to, so an
# EPP config and the build that can load it stay one unit. Sourced FIRST, so
# deploy.env below can override when a framework must (see versions.env).
set -a
# shellcheck disable=SC1091
. ../../../common/src/llm_d_rl_common/configs/versions.env
IMG_EPP="${IMG_EPP:-$LLMD_EPP_IMAGE}"
IMG_ENVOY="${IMG_ENVOY:-$LLMD_ENVOY_IMAGE}"
IMG_SIDECAR="${IMG_SIDECAR:-$LLMD_SIDECAR_IMAGE}"
# shellcheck disable=SC1091
. ./deploy.env
set +a

# NAMESPACE is per-user and comes from the environment, not deploy.env.
: "${NAMESPACE:?not set - export NAMESPACE=<your-namespace>}"
# EPP image must include sglanghttp-parser; deploy.env ships a placeholder.
# Skip on delete: kubectl matches by resource name, not image.
if [[ "$ACTION" != "delete" ]]; then
  case "${IMG_EPP:-}" in
    ""|*REPLACE*|*placeholder*|"<epp-image-with-sglanghttp-parser>")
      echo "IMG_EPP is a placeholder — set it in deploy.env to an EPP image that includes sglanghttp-parser" >&2
      exit 2
      ;;
  esac
fi
# Burst EPP parser. Default is vllmhttp-parser; slime overrides in deploy.env.
# Must be exported: envsubst only substitutes exported variables, otherwise
# ${EPP_PARSER} becomes empty and EPP refuses to start (plugin '' missing type).
export EPP_PARSER="${EPP_PARSER:-vllmhttp-parser}"

COMMON_CONFIGS="$(cd ../../../common/src/llm_d_rl_common/configs && pwd)"
# Source tree of llm-d-rl-common, so the EPP config merger runs without the
# package being installed on whoever is deploying.
COMMON_SRC="$(cd ../../../common/src && pwd)"

render() {
  # Explicit var list prevents envsubst from expanding shell $-vars inside
  # the postStart script (e.g. $! from background process management).
  envsubst '${NAMESPACE} ${IMG_SLIME} ${IMG_CRANE} ${IMG_EPP} ${IMG_ENVOY}' \
    < ray-cluster.yaml.tmpl
}

create_configmap() {
  # epp-config.yaml is the burst variant, merged from base.yaml +
  # profiles/burst.yaml (configs/epp/variants.yaml); EPP_PARSER is substituted
  # afterwards, as before.
  local rendered
  rendered="$(mktemp)"
  trap 'rm -f "$rendered"' RETURN
  PYTHONPATH="$COMMON_SRC" python3 -m llm_d_rl_common.epp_config render epp-config.yaml \
    | envsubst '${EPP_PARSER}' > "$rendered"
  kubectl create configmap llmd-epp-configs-slime \
    --from-file=epp-config.yaml="$rendered" \
    --from-file=envoy.yaml="$COMMON_CONFIGS/envoy-shim.yaml" \
    --from-file=run-qwen3-4B.sh=run-qwen3-4B.sh \
    --namespace "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
}

case "$ACTION" in
  render)    render ;;
  configmap) create_configmap ;;
  apply)     create_configmap; render | kubectl apply -f - ;;
  delete)    render | kubectl delete --ignore-not-found -f - ;;
  *) echo "Unknown action: $ACTION (use apply | delete | configmap | render)" >&2; exit 2 ;;
esac
