#!/usr/bin/env bash
# Submit a Qwen3-4B training job on the running vime KubeRay cluster.
# Run this from inside the head pod after the cluster is ready.
#
# Usage:
#   bash /etc/llmd-configs/run-qwen3-4B.sh --native
#       vime's built-in vLLM router
#   bash /etc/llmd-configs/run-qwen3-4B.sh --llmd
#       llm-d EPP + Envoy routing (--vllm-router-ip/-port point vime at Envoy)
#   bash /etc/llmd-configs/run-qwen3-4B.sh --native --steps 6
#       short run (6 training steps); default: 500
#
# Both modes run the same engine layout - 4 engines at TP=1
# (--rollout-num-gpus 4 --rollout-num-gpus-per-engine 1, set unconditionally
# below) - so the only difference between the two arms is who picks the endpoint.
set -euo pipefail

MODE=""
STEPS=500
FORCE_DOWNLOAD=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --native)         MODE=native; shift ;;
    --llmd)           MODE=llmd; shift ;;
    --steps)          STEPS="$2"; shift 2 ;;
    --steps=*)        STEPS="${1#--steps=}"; shift ;;
    --force-download) FORCE_DOWNLOAD=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$MODE" ]; then
  echo "Usage: $0 --native | --llmd [--steps N] [--force-download]" >&2
  exit 1
fi

MODEL_DIR="/tmp/vime/models/${MODEL_NAME:-Qwen3-4B}"
MEGATRON_DIR="/tmp/vime/models/${MODEL_NAME:-Qwen3-4B}_megatron"
DATASET_DIR="/tmp/vime/data/${DATASET_NAME:-dapo-math-17k}"

cd /tmp/vime
source scripts/models/qwen3-4B.sh

# --- 1. Download model + dataset (skipped if already present) ---
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

if [ ! -d "$DATASET_DIR" ] || [ "$FORCE_DOWNLOAD" = true ]; then
  echo "=== Downloading dataset ==="
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${DATASET_ID:-zhuzilin/dapo-math-17k}', repo_type='dataset',
    local_dir='$DATASET_DIR', local_dir_use_symlinks=False)
"
else
  echo "=== Dataset already present, skipping download ==="
fi

# --- 2. Convert HF weights to Megatron format (skipped if already done) ---
if [ ! -d "$MEGATRON_DIR" ]; then
  echo "=== Converting weights to Megatron format ==="
  PYTHONPATH=/tmp/pyfix:/tmp/Megatron-LM python3 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$MODEL_DIR" \
    --save "$MEGATRON_DIR"
else
  echo "=== Megatron weights already exist, skipping conversion ==="
fi

# --- 3. Submit training ---
echo "=== Submitting training job (mode: $MODE, steps: $STEPS) ==="

EXTRA_ARGS=()
if [ "$MODE" = "llmd" ]; then
  EXTRA_ARGS=(
    --vllm-router-ip "${MY_POD_IP}"
    --vllm-router-port 8081
  )
fi

ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars":{"PYTHONPATH":"/tmp/pyfix:/tmp/Megatron-LM","CUDA_DEVICE_MAX_CONNECTIONS":"1","TORCHINDUCTOR_CACHE_DIR":"/tmp/torch_inductor_cache","VLLM_TARGET_DEVICE":"cuda","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}}' \
  -- python3 /tmp/vime/train.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$MODEL_DIR" \
    --load "$MEGATRON_DIR" \
    --ref-load "$MEGATRON_DIR" \
    --prompt-data "$DATASET_DIR/${DATASET_NAME:-dapo-math-17k}.jsonl" \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rm-type deepscaler \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --recompute-activations \
    --rollout-num-gpus 4 \
    --rollout-num-gpus-per-engine 1 \
    --rollout-batch-size 32 \
    --num-rollout "$STEPS" \
    --n-samples-per-prompt 4 \
    --global-batch-size 128 \
    --rollout-max-response-len 8192 \
    --rollout-temperature 1 \
    --balance-data \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.00 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 9216 \
    --vllm-gpu-memory-utilization 0.7 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    "${EXTRA_ARGS[@]}"
