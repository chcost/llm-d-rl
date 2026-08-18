#!/usr/bin/env bash
# Supply the EPP to the running Ray head pod from a SEPARATE source - a freshly
# built binary or a separate EPP image - without rebuilding the verl image.
#
# The verl image no longer contains the EPP. ray-cluster.yaml.tmpl mounts an emptyDir
# at /opt/llm-d-bins and sets VERL_EPP_BINARY=/opt/llm-d-bins/epp, so LlmdActor
# launches whatever binary lands there. This script puts it there. The EPP is
# started once per training job, so run this BEFORE launching a run (and again
# after any pod recreation, since the emptyDir is wiped).
#
# Usage:
#   ./push-epp.sh                         # build cmd/epp from the scheduler repo
#   ./push-epp.sh --from-image REF        # extract /app/epp from a separate image
#                                         # (works on distroless: host-side copy)
#
# Override via env:
#   SCHEDULER_REPO     scheduler checkout (default: $HOME/llm-d-inference-scheduler)
#   CONTAINER_RUNTIME  docker|podman for --from-image (default: docker)
#   NAMESPACE          k8s namespace (REQUIRED; export NAMESPACE=<your-namespace>)
#   HEAD_CONTAINER     container in the head pod (default: ray-head)
#   EPP_DEST           path in the pod (default: /opt/llm-d-bins/epp; match VERL_EPP_BINARY)
set -euo pipefail

SCHEDULER_REPO="${SCHEDULER_REPO:-$HOME/llm-d-inference-scheduler}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
# Namespace is per-user and comes from the environment. Mandatory, no default.
: "${NAMESPACE:?not set - export NAMESPACE=<your-namespace>}"
HEAD_CONTAINER="${HEAD_CONTAINER:-ray-head}"
EPP_DEST="${EPP_DEST:-/opt/llm-d-bins/epp}"
LOCAL_BIN="$(mktemp -t epp.XXXXXX)"
trap 'rm -f "$LOCAL_BIN"' EXIT

FROM_IMAGE=""
if [ "${1:-}" = "--from-image" ]; then
    FROM_IMAGE="${2:-}"
    [ -n "$FROM_IMAGE" ] || { echo "ERROR: --from-image needs an image reference" >&2; exit 2; }
fi

if [ -n "$FROM_IMAGE" ]; then
    echo "==> Extracting /app/epp from separate image: $FROM_IMAGE"
    # Host-side copy out of the image filesystem - no shell/cp needed in the
    # image, so this works even though the EPP image is distroless.
    "$CONTAINER_RUNTIME" pull "$FROM_IMAGE"
    cid="$("$CONTAINER_RUNTIME" create "$FROM_IMAGE")"
    trap '"$CONTAINER_RUNTIME" rm "$cid" >/dev/null 2>&1; rm -f "$LOCAL_BIN"' EXIT
    "$CONTAINER_RUNTIME" cp "$cid:/app/epp" "$LOCAL_BIN"
else
    echo "==> Building EPP from $SCHEDULER_REPO (branch: $(git -C "$SCHEDULER_REPO" branch --show-current))"
    # CGO disabled -> static binary that runs on the verl image's glibc unchanged.
    ( cd "$SCHEDULER_REPO" && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
        go build -ldflags="-s -w" -o "$LOCAL_BIN" ./cmd/epp )
fi
echo "    EPP binary: $(du -h "$LOCAL_BIN" | cut -f1)"

echo "==> Finding Ray head pod in namespace $NAMESPACE"
HEAD_POD="$(kubectl get pod -n "$NAMESPACE" -l ray.io/node-type=head \
    -o jsonpath='{.items[0].metadata.name}')"
[ -n "$HEAD_POD" ] || { echo "ERROR: no head pod (label ray.io/node-type=head)" >&2; exit 1; }
echo "    head pod: $HEAD_POD"

echo "==> Copying to $HEAD_POD:$EPP_DEST ($HEAD_CONTAINER)"
kubectl cp "$LOCAL_BIN" "$NAMESPACE/$HEAD_POD:$EPP_DEST" -c "$HEAD_CONTAINER"
kubectl exec -n "$NAMESPACE" "$HEAD_POD" -c "$HEAD_CONTAINER" -- \
    sh -c "chmod +x '$EPP_DEST' && ls -la '$EPP_DEST'"

cat <<EOF

==> Done. EPP is in place at $EPP_DEST (no verl rebuild, no pod recreation).
    Start / restart the verl run to pick it up (LlmdActor launches EPP once per
    job). EPP config lives in the ConfigMap - rebuild it with kuberay/deploy.sh
    configmap and restart the run to change it.
EOF
