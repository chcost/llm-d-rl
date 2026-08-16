#!/usr/bin/env bash
# Deploy (or tear down) the slime KubeRay cluster using the single config in
# deploy.env. Renders ray-cluster.yaml.tmpl with the image refs and namespace,
# builds the llmd-epp-configs-slime ConfigMap from common/deploy (envoy + burst
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

set -a
# shellcheck disable=SC1091
. ./deploy.env
set +a

# NAMESPACE is per-user and comes from the environment, not deploy.env.
: "${NAMESPACE:?not set - export NAMESPACE=<your-namespace>}"
# Burst EPP parser. Default is vllmhttp-parser; slime overrides in deploy.env.
# Must be exported: envsubst only substitutes exported variables, otherwise
# ${EPP_PARSER} becomes empty and EPP refuses to start (plugin '' missing type).
export EPP_PARSER="${EPP_PARSER:-vllmhttp-parser}"

COMMON_DEPLOY="$(cd ../../../common/deploy && pwd)"

render() {
  # Explicit var list prevents envsubst from expanding shell $-vars inside
  # the postStart script (e.g. $! from background process management).
  envsubst '${NAMESPACE} ${IMG_SLIME} ${IMG_CRANE} ${IMG_EPP} ${IMG_ENVOY}' \
    < ray-cluster.yaml.tmpl
}

create_configmap() {
  # Burst EPP + Envoy live in integrations/common/deploy/. Render the parser
  # name here (scoped envsubst — do not pass the RayCluster var list).
  local rendered
  rendered="$(mktemp)"
  trap 'rm -f "$rendered"' RETURN
  envsubst '${EPP_PARSER}' < "$COMMON_DEPLOY/epp-config-burst.yaml" > "$rendered"
  kubectl create configmap llmd-epp-configs-slime \
    --from-file=epp-config.yaml="$rendered" \
    --from-file=envoy.yaml="$COMMON_DEPLOY/envoy-shim.yaml" \
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
