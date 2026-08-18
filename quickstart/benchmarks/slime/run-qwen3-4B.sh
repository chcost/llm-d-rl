#!/usr/bin/env bash
# Submit a Qwen3-4B GRPO training job on the running slime KubeRay cluster.
# Run this from inside the head pod after the cluster is ready.
#
# Usage:
#   bash run-qwen3-4B.sh                       EPP + Envoy routing (llm-d), default
#   bash run-qwen3-4B.sh --mode llm-d          same
#   bash run-qwen3-4B.sh --mode native         slime's built-in sglang-router, no EPP
#   bash run-qwen3-4B.sh --steps 6             short run (6 training steps); default: 500
#   bash run-qwen3-4B.sh --force-download      re-download model/data even if present
set -euo pipefail

MODE=llm-d
STEPS=500
FORCE_DOWNLOAD=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#--mode=}"; shift ;;
    --steps) STEPS="$2"; shift 2 ;;
    --steps=*) STEPS="${1#--steps=}"; shift ;;
    --force-download) FORCE_DOWNLOAD=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  llm-d|native) ;;
  *) echo "Unknown --mode: $MODE (use llm-d or native)" >&2; exit 2 ;;
esac

MODEL_DIR="/tmp/slime/models/${MODEL_NAME:-Qwen3-4B}"
TORCH_DIST_DIR="/tmp/slime/models/${MODEL_NAME:-Qwen3-4B}_torch_dist"
DATA_DIR="/tmp/slime/data"

# Same MODEL_ARGS as slime's own example; extras (router, batch, paths) stay below.
SLIME_MODEL_SCRIPT="/tmp/slime-src/scripts/models/qwen3-4B.sh"
if [ ! -f "$SLIME_MODEL_SCRIPT" ]; then
  echo "Missing $SLIME_MODEL_SCRIPT — wait for postStart to finish cloning slime ( /tmp/slime_ready.txt )." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$SLIME_MODEL_SCRIPT"

# --- 1. Download model weights ---
if [ ! -d "$MODEL_DIR" ] || [ "$FORCE_DOWNLOAD" = true ]; then
  echo "=== Downloading model ==="
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID:-Qwen/Qwen3-4B}',
    local_dir='$MODEL_DIR', local_dir_use_symlinks=False)
"
else
  echo "=== Model already present, skipping download ==="
fi

# --- 2. Download dataset ---
if [ ! -d "$DATA_DIR" ] || [ "$FORCE_DOWNLOAD" = true ]; then
  echo "=== Downloading dataset ==="
  mkdir -p "$DATA_DIR"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${DATASET_ID:-zhuzilin/dapo-math-17k}', repo_type='dataset',
    local_dir='$DATA_DIR', local_dir_use_symlinks=False)
"
else
  echo "=== Dataset already present, skipping download ==="
fi

# --- 3. Convert HF weights to Megatron torch_dist format (required by --ref-load) ---
if [ ! -d "$TORCH_DIST_DIR" ] || [ "$FORCE_DOWNLOAD" = true ]; then
  echo "=== Converting weights to torch_dist format ==="
  PYTHONPATH=/tmp/pyfix:/tmp/slime-src:/tmp/Megatron-LM \
    python3 /tmp/slime-src/tools/convert_hf_to_torch_dist.py \
      --hf-checkpoint "$MODEL_DIR" \
      --save "$TORCH_DIST_DIR" \
      "${MODEL_ARGS[@]}" \
      --attention-dropout 0.0 \
      --hidden-dropout 0.0
else
  echo "=== torch_dist checkpoint already exists, skipping conversion ==="
fi

# --- 4. Submit training job ---
echo "=== Submitting training job (mode: $MODE, steps: $STEPS) ==="

# llm-d: route through Envoy+EPP (prefix-cache-aware).
# native: omit router args; slime manages its own sglang-router.
ROUTER_ARGS=()
if [[ "$MODE" == "llm-d" ]]; then
  ROUTER_ARGS+=(--sglang-router-ip "$MY_POD_IP" --sglang-router-port 8081)
fi

ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars":{"PYTHONPATH":"/tmp/pyfix:/tmp/slime-src:/tmp/Megatron-LM","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}' \
  -- python3 /tmp/slime-src/train.py \
    "${ROUTER_ARGS[@]}" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --rollout-num-gpus 4 \
    --rollout-num-gpus-per-engine 1 \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$MODEL_DIR" \
    --ref-load "$TORCH_DIST_DIR" \
    --prompt-data "$DATA_DIR/${DATASET_FILE:-dapo-math-17k.jsonl}" \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --num-rollout "$STEPS" \
    --rollout-batch-size 32 \
    --n-samples-per-prompt 4 \
    --rollout-max-response-len 8192 \
    --rollout-temperature 1 \
    --global-batch-size 128 \
    --balance-data \
    --rm-type deepscaler \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --kl-loss-type low_var_kl \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --entropy-coef 0.00 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 9216 \
    --sglang-mem-fraction-static 0.7 \
    --sglang-enable-metrics \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash
