#!/usr/bin/env bash
# run_test.sh  --mode <native|epp|epp-inflight|epp-fc|epp-p2p|epp-sglang|wave-admission|wave-admission-p2p|llm-d>  [options]
#
# Usage examples:
#   bash run_test.sh --mode native
#   bash run_test.sh --mode epp
#   bash run_test.sh --mode epp --steps 20 --tp 2 --n 4
#   bash run_test.sh --mode epp-fc --task weka   # EPP routing + per-endpoint concurrency CAP (flow-control queue)
#   bash run_test.sh --mode epp-p2p              # EPP routing + P2P KV-cache sharing (every replica pulls/serves)
#   bash run_test.sh --mode epp-sglang           # EPP direct-gRPC routing, SGLang replicas instead of vLLM
#   bash run_test.sh --mode wave-admission --task weka   # estimation-gated admission, no EPP (see wave_admission/)
#   bash run_test.sh --mode wave-admission-p2p --task weka   # wave-admission migration + P2P KV pull (near-free migration)
#   bash run_test.sh --mode llm-d          # (not yet implemented)
#
# Options:
#   --mode   native | epp | epp-inflight | epp-fc | epp-p2p | epp-sglang | wave-admission | wave-admission-p2p | llm-d (required)
#   --steps  total_training_steps          (default: 40)
#   --tp     tensor-parallel size          (default: 1)
#   --n      rollout group size            (default: 8)
#   --task   any folder under workloads/ (gsm8k | hotpotqa | musique | quality |
#            searchr1 | scotus_xl | arxiv | geo3k)   (default: gsm8k)
#   --name   override experiment name      (default: auto-generated)
#   --reqlog enable per-request JSONL log  (default: on for all modes)

set -euo pipefail

# -- defaults -----------------------------------------------------------------
MODE=""
STEPS=40
TP=1
N=8
CUSTOM_NAME=""
REQLOG=""          # empty = auto (on for non-native modes)
TASK="gsm8k"       # name of a folder under workloads/ (each has a task.env)

# -- arg parsing ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)   MODE="$2";        shift 2 ;;
    --steps)  STEPS="$2";       shift 2 ;;
    --tp)     TP="$2";          shift 2 ;;
    --n)      N="$2";           shift 2 ;;
    --task)   TASK="$2";        shift 2 ;;   # any folder name under workloads/
    --name)   CUSTOM_NAME="$2"; shift 2 ;;
    --reqlog) REQLOG="$2";      shift 2 ;;   # "on" or "off"
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "ERROR: --mode is required  (native | epp | epp-inflight | epp-fc | epp-p2p | epp-sglang | wave-admission | wave-admission-p2p | llm-d)"
  exit 1
fi

# -- per-mode overrides -------------------------------------------------------
# The mode matrix is data in the integration package (modes.yaml), not a case
# statement here: an adopter running their own launcher needs the same contract,
# and when it lived only in this driver the documented reference drifted from it
# in both directions. This driver now consumes it.
#
# Prefer the installed package (that is what runs on the pod); fall back to the
# source tree so the driver works from a checkout too.
MODES_PY=(python3 -m llm_d_rl_verl_integration.modes)
if ! python3 -c "import llm_d_rl_verl_integration.modes" 2>/dev/null; then
  SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../src" 2>/dev/null && pwd || true)"
  if [[ -n "$SRC_DIR" ]]; then
    MODES_PY=(env "PYTHONPATH=$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m llm_d_rl_verl_integration.modes)
  else
    echo "ERROR: cannot find llm_d_rl_verl_integration.modes (not installed, no ../../src)" >&2
    exit 1
  fi
fi

# EPP_CONFIG is read by modes.py, so it needs no plumbing here.
if ! MODE_OVERRIDES="$("${MODES_PY[@]}" "$MODE" 2>&1)"; then
  echo "ERROR: $MODE_OVERRIDES" >&2
  echo "       available modes:" >&2
  "${MODES_PY[@]}" --list >&2
  exit 1
fi
mapfile -t MODE_ARGS <<< "$MODE_OVERRIDES"

# Experiment name: the mode with its hyphens dropped, matching the names already
# on disk in verl-results. "native" kept its historical "baseline" slug.
if [[ "$MODE" == "native" ]]; then SLUG="baseline"; else SLUG="${MODE//-/}"; fi
DEFAULT_NAME="qwen3_4b_grpo_${SLUG}_tp${TP}_n${N}_${STEPS}s"
[[ -z "$REQLOG" ]] && REQLOG="on"

EXPERIMENT_NAME="${CUSTOM_NAME:-$DEFAULT_NAME}"

# -- reqlog override -----------------------------------------------------------
# Where per-request JSONL goes. A driver concern (the integration only reads
# VERL_REQLOG_DIR), so it is not in modes.yaml.
if [[ "$REQLOG" == "on" ]]; then
  MODE_ARGS+=(+ray_kwargs.ray_init.runtime_env.env_vars.VERL_REQLOG_DIR=/tmp/verl/reqlog)
fi

# -- task config: sourced from workloads/<task>/task.env ------------------------
# Adding a workload means adding a folder; this driver does not change.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve the workloads dir: explicit WORKLOADS_DIR override, else the repo layout
# (benchmarks/scripts -> ../workloads), else /tmp/workloads (where run_on_head.sh copies
# the selected workload folder alongside run_test.sh on the head pod).
WORKLOADS_DIR="${WORKLOADS_DIR:-}"
if [[ -z "$WORKLOADS_DIR" ]]; then
  if [[ -d "$SCRIPT_DIR/../workloads" ]]; then
    WORKLOADS_DIR="$(cd "$SCRIPT_DIR/../workloads" && pwd)"
  elif [[ -d /tmp/workloads ]]; then
    WORKLOADS_DIR=/tmp/workloads
  fi
fi
TASK_ENV="$WORKLOADS_DIR/$TASK/task.env"
if [[ ! -f "$TASK_ENV" ]]; then
  echo "ERROR: no task.env for --task '$TASK' (looked at: $TASK_ENV)"
  echo "       available workloads: $(ls -1 "$WORKLOADS_DIR" 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
TASK_OVERRIDES=()
# shellcheck disable=SC1090
source "$TASK_ENV"

TRAIN_RESOLVED=${TRAIN_FILE:-$DEF_TRAIN}
TEST_RESOLVED=${TEST_FILE:-$DEF_TEST}
MODEL_RESOLVED=${MODEL_PATH:-$DEF_MODEL}

# -- checkpoint compatibility patch --------------------------------------------
# Strip dual_chunk_attention_config from Qwen 1M checkpoints; this vLLM nightly
# dies on it. Idempotent. SKIP_DCA_STRIP=1 disables. Falls back to /tmp/utils,
# where run_on_head.sh ships the script.
DCA_STRIP=""
for cand in "$SCRIPT_DIR/utils/strip_dca_config.py" /tmp/utils/strip_dca_config.py; do
  [[ -f "$cand" ]] && DCA_STRIP="$cand" && break
done
if [[ "${SKIP_DCA_STRIP:-0}" != "1" && -n "$MODEL_RESOLVED" ]]; then
  if [[ -n "$DCA_STRIP" ]]; then
    python3 "$DCA_STRIP" "$MODEL_RESOLVED" || {
      echo "ERROR: strip_dca_config.py failed for $MODEL_RESOLVED"
      exit 1
    }
  else
    echo "WARNING: strip_dca_config.py not found - skipping the DCA compatibility patch." >&2
    echo "         A Qwen 1M-context checkpoint will crash vLLM on an unsupported layer_idx kwarg." >&2
  fi
fi

# Optional extra hydra overrides, appended LAST so they win over the per-task defaults
# (e.g. raise ppo/log_prob token budgets for a bigger max_prompt). Space-separated;
# values must not contain spaces. Empty by default.
read -r -a EXTRA_OV <<< "${EXTRA_OVERRIDES:-}"

# -- launch --------------------------------------------------------------------
cd /tmp/verl/verl/examples/grpo_trainer

# -- vLLM /metrics scraper ------------------------------------------------------
# Always on, every mode: a run costs GPU time and a re-run costs it again, while
# this is a 1.5s poll with no measurable cost - and missing metrics have already
# forced re-runs here. It tolerates an endpoints file that does not exist yet
# (catches the read and loops), so starting it before the engines are up is safe.
# Set VLLM_SCRAPE_HOST when vLLM binds loopback (nosidecar P2P).
SCRAPE_PID=""
for cand in "$SCRIPT_DIR/vllm_scrape.py" /tmp/utils/vllm_scrape.py; do
  if [[ -f "$cand" ]]; then
    rm -f "${VLLM_SCRAPE_OUT:-/tmp/vllm_metrics.csv}"
    nohup python3 "$cand" > /tmp/vllm_scrape.log 2>&1 &
    SCRAPE_PID=$!
    echo "==> vLLM scraper started (pid $SCRAPE_PID) -> ${VLLM_SCRAPE_OUT:-/tmp/vllm_metrics.csv}"
    break
  fi
done
[[ -z "$SCRAPE_PID" ]] && echo "WARNING: vllm_scrape.py not found - no /metrics will be captured for this arm" >&2
# Stop it however the run ends, so it never bleeds into the next arm's CSV.
trap '[[ -n "$SCRAPE_PID" ]] && kill "$SCRAPE_PID" 2>/dev/null' EXIT

# TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE are env-overridable: workloads with fewer
# prompts than the default batch (e.g. weka replays N<256 conversations) must set
# them to the conversation count, or the dataloader cannot fill a step.
ROLLOUT_N=$N ROLLOUT_TP=$TP NGPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128} \
MODEL_PATH=$MODEL_RESOLVED \
TRAIN_FILE=$TRAIN_RESOLVED \
TEST_FILE=$TEST_RESOLVED \
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$DEF_MAXP} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$DEF_MAXR} \
SAVE_FREQ=-1 PROJECT_NAME=${PROJECT_NAME:-$DEF_PROJECT} \
EXPERIMENT_NAME=$EXPERIMENT_NAME \
bash "$FSDP_SCRIPT" \
  trainer.logger='["console","file"]' \
  trainer.total_training_steps=$STEPS \
  trainer.default_local_dir=/tmp/checkpoints \
  trainer.rollout_data_dir=/tmp/verl/generations/train \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.n=$N \
  ${TASK_OVERRIDES[@]+"${TASK_OVERRIDES[@]}"} \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true \
  ${EXTRA_OV[@]+"${EXTRA_OV[@]}"} \
  hydra.run.dir=/tmp/hydra-outputs \
  "${MODE_ARGS[@]}"
