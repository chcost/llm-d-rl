#!/usr/bin/env bash
# Deploy (or tear down) the KubeRay example using the single config in
# deploy.env. Renders ray-cluster.yaml.tmpl with the image refs and namespace
# from deploy.env, builds the llmd-epp-configs ConfigMap from the standalone
# EPP config files, and applies both - so the namespace and images are defined
# in exactly one place (deploy.env).
#
# Usage:
#   ./deploy.sh                  # create ConfigMap + apply the cluster
#   ./deploy.sh apply            # same
#   ./deploy.sh delete           # delete the cluster (leaves the ConfigMap)
#   ./deploy.sh configmap        # (re)create the ConfigMap only
#   ./deploy.sh render           # print the rendered manifest to stdout (no kubectl)
#   ./deploy.sh retriever        # apply the BM25 searchr1 retriever (Deployment+Service)
#   ./deploy.sh retriever-delete # delete the retriever
#   ./deploy.sh render-retriever # print the rendered retriever manifest (no kubectl)
#
# Rollout engine (one manifest, one cluster name, per-engine values from deploy.env):
#   ./deploy.sh apply                     # vLLM (default)
#   ./deploy.sh apply --engine sglang     # SGLang
#   ./deploy.sh render --engine sglang    # inspect before applying
#
# The two engines are deliberately mutually exclusive within a namespace: they
# render to the same RayCluster name, so applying one replaces the other. Every
# script here resolves pods with `-l ray.io/node-type=head` + `items[0]` (see
# benchmarks/scripts/run_on_head.sh, utils/push-epp.sh, rl_orchestrate.sh), which
# cannot tell two clusters apart - two live clusters would mean silently
# benchmarking, or pushing an EPP binary into, whichever one came back first.
#
# Requires: envsubst (GNU gettext) and kubectl on PATH.
set -euo pipefail

ACTION="apply"
ENGINE="vllm"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="${2:?--engine needs a value}"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) ACTION="$1"; shift ;;
  esac
done

cd "$(dirname "$0")"

# deploy.env provides the IMG_* refs (envsubst reads them from the environment).
# The routing stack's own versions ship beside the configs they belong to, so an
# EPP config and the build that can load it stay one unit. Sourced FIRST, so
# deploy.env below can override when a framework must (see versions.env).
set -a
# shellcheck disable=SC1091
. ../../../common/src/llm_d_rl_common/configs/versions.env
IMG_EPP="${IMG_EPP:-$LLMD_EPP_IMAGE}"
IMG_ENVOY="${IMG_ENVOY:-$LLMD_ENVOY_IMAGE}"
IMG_SIDECAR="${IMG_SIDECAR:-$LLMD_SIDECAR_IMAGE}"
# Which environment image and framework ref go together - framework knowledge,
# so it is declared with the integration rather than here.
# shellcheck disable=SC1091
. ../../environments.env
# shellcheck disable=SC1091
. ./deploy.env
set +a

# NAMESPACE is per-user and comes from the environment, not deploy.env. Mandatory,
# no default: :? fails fast (before any kubectl) on empty OR unset.
: "${NAMESPACE:?not set - export NAMESPACE=<your-namespace>}"

# An empty VERL_COMMIT would render a bare `git checkout`, which exits 0 and
# leaves the clone on its default branch - a silently wrong verl version.
: "${VERL_COMMIT:?not set - define it in deploy.env}"
# EPP parser for common/src/llm_d_rl_common/configs/epp-config-burst.yaml.
# Default is vllmhttp-parser; only slime overrides this.
export EPP_PARSER="${EPP_PARSER:-vllmhttp-parser}"

COMMON_CONFIGS="$(cd ../../../common/src/llm_d_rl_common/configs && pwd)"
# Source tree of llm-d-rl-common, so the EPP config merger runs without the
# package being installed on whoever is deploying.
COMMON_SRC="$(cd ../../../common/src && pwd)"

# Resolve the per-engine column from deploy.env into the names the manifest uses.
# Fails fast on an engine with no column rather than rendering blank values into a
# manifest that would then fail confusingly at pod start.
select_engine() {
  local img py cpus alloc
  img="ENGINE_${ENGINE}_IMAGE"; py="ENGINE_${ENGINE}_PY_MODULE"
  cpus="ENGINE_${ENGINE}_HEAD_NUM_CPUS"; alloc="ENGINE_${ENGINE}_ALLOC_CONF"
  if [[ -z "${!img:-}" ]]; then
    echo "ERROR: unknown engine '${ENGINE}' - add an ENGINE_${ENGINE}_* block to deploy.env" >&2
    exit 2
  fi
  export IMG_TRAINER="${!img}"
  export ENGINE_PY_MODULE="${!py}"
  export ENGINE_HEAD_NUM_CPUS="${!cpus}"
  # May legitimately be empty (SGLang), so no :- guard and no emptiness check.
  export ENGINE_ALLOC_CONF="${!alloc-}"
}

render() {
  select_engine
  # Explicit var list keeps envsubst from touching the container-runtime
  # $EPP_IMAGE / $ENVOY_IMAGE in the crane args and the shell $ in postStart.
  envsubst '${NAMESPACE} ${IMG_TRAINER} ${IMG_CRANE} ${IMG_EPP} ${IMG_ENVOY} ${IMG_SIDECAR} ${ENGINE_PY_MODULE} ${ENGINE_HEAD_NUM_CPUS} ${ENGINE_ALLOC_CONF} ${VERL_COMMIT}' \
    < ray-cluster.yaml.tmpl
}

render_retriever() {
  # BM25 retriever Deployment+Service (searchr1 workload only). Scoped var list so the
  # $-quoted init-container script is left untouched.
  envsubst '${NAMESPACE} ${IMG_RETRIEVER}' < ../../benchmarks/workloads/searchr1/retriever/retriever.yaml.tmpl
}

create_configmap() {
  # Every shipped EPP variant is base.yaml + one profiles/ file + any modifiers/
  # (see configs/epp/variants.yaml); the merger renders them, then EPP_PARSER is
  # substituted the same way it always was. Adding a variant is a line in
  # variants.yaml - this function does not change.
  local dir
  dir="$(mktemp -d)"
  trap 'rm -rf "$dir"' RETURN
  PYTHONPATH="$COMMON_SRC" python3 -m llm_d_rl_common.epp_config render-all "$dir" >/dev/null
  local epp_args=()
  local f
  for f in "$dir"/*.yaml; do
    envsubst '${EPP_PARSER}' < "$f" > "$f.sub" && mv "$f.sub" "$f"
    epp_args+=(--from-file="$(basename "$f")=$f")
  done
  kubectl create configmap llmd-epp-configs \
    "${epp_args[@]}" \
    --from-file=envoy.yaml="$COMMON_CONFIGS/envoy.yaml" \
    --from-file=searchr1_tool_config.yaml=../../benchmarks/workloads/searchr1/tool_config.yaml \
    --from-file=trace_player_agent_loop.yaml=../../benchmarks/workloads/weka/trace_player_agent_loop.yaml \
    --namespace "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
}

case "$ACTION" in
  render)            render ;;
  configmap)         create_configmap ;;
  apply)             create_configmap; render | kubectl apply -f - ;;
  delete)            render | kubectl delete -f - ;;
  retriever)         render_retriever | kubectl apply -f - ;;
  retriever-delete)  render_retriever | kubectl delete -f - ;;
  render-retriever)  render_retriever ;;
  apply-sglang|delete-sglang|render-sglang)
    echo "ERROR: '$ACTION' is gone - one manifest now serves both engines." >&2
    echo "       Use: ./deploy.sh ${ACTION%-sglang} --engine sglang" >&2
    exit 2 ;;
  *) echo "Unknown action: $ACTION (use apply | delete | configmap | render | retriever | retriever-delete | render-retriever, with optional --engine vllm|sglang)" >&2; exit 2 ;;
esac
