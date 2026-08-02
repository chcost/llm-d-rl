#!/usr/bin/env bash
# Deploy (or tear down) the slime KubeRay cluster using the single config in
# deploy.env. Renders ray-cluster.yaml.tmpl with the image refs and namespace,
# builds the llmd-epp-configs ConfigMap from the standalone config files, and
# applies both — images and namespace are defined in exactly one place.
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

render() {
  # Explicit var list prevents envsubst from expanding shell $-vars inside
  # the postStart script (e.g. $! from background process management).
  envsubst '${NAMESPACE} ${IMG_SLIME} ${IMG_CRANE} ${IMG_EPP} ${IMG_ENVOY}' \
    < ray-cluster.yaml.tmpl
}

create_configmap() {
  # Include router_shim.py so the postStart hook can use it directly from the
  # ConfigMap mount (/etc/llmd-configs/router_shim.py) without a git clone.
  kubectl create configmap llmd-epp-configs \
    --from-file=epp-config.yaml=../epp-config.yaml \
    --from-file=envoy.yaml=../envoy.yaml \
    --from-file=router_shim.py=../../shim/router_shim.py \
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
