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

# -- per-mode config -----------------------------------------------------------
# Each branch only picks the manager class / EPP config; hydra overrides are
# assembled once after the case statement.
AGENT_LOOP_MANAGER_CLASS=""
EPP_CONFIG_FILE=""
EPP_REPORT_COMPLETION=""
ROLLOUT_NAME=""        # only epp-p2p sets this (registers a non-default rollout backend)
EXTERNAL_LIB=""        # only epp-p2p sets this (model.external_lib import hook)
P2P_ENGINE_HYDRA=""    # only epp-p2p sets this (OffloadingConnector engine_kwargs)

# -- EPP config selection ------------------------------------------------------
# One EPP_CONFIG picks the scorer variant for every EPP-bearing mode; each mode
# below supplies its own default. EPP_CAP_CONFIG (epp-fc) and EPP_P2P_CONFIG
# (epp-p2p) were the old per-mode names: still honoured so existing sweep drivers
# keep running unchanged, but deprecated. EPP_CONFIG wins if both are set.
for _legacy in EPP_CAP_CONFIG EPP_P2P_CONFIG; do
  if [[ -n "${!_legacy:-}" ]]; then
    echo "WARNING: $_legacy is deprecated - use EPP_CONFIG (value honoured for now)" >&2
    : "${EPP_CONFIG:=${!_legacy}}"
  fi
done

# -- shared P2P engine config (--mode epp-p2p and wave-admission-p2p) ----------
# spec_name + secondary_tiers are BOTH mandatory or there is no P2P tier and
# remote_kv_source is silently ignored. Size cpu_bytes_to_use >= the per-replica
# GPU KV cache, capped by /dev/shm (see ray-cluster.yaml.tmpl's dshm sizeLimit).
P2P_ENGINE_BASE="
      +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=llm_d_rl_verl_integration.register_p2p \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=64 \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=OffloadingConnector \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.offload_prompt_only=false \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.cpu_bytes_to_use=${P2P_CPU_BYTES_TO_USE:-4294967296} \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.spec_name=TieringOffloadingSpec \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.secondary_tiers=[{type:p2p}]"

case "$MODE" in
  native)
    DEFAULT_NAME="qwen3_4b_grpo_baseline_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    # Stock native routing, plus reqlog + endpoints YAML for the scraper.
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.native_logging.agent_loop_manager.NativeLoggingAgentLoopManager"
    ;;

  epp)
    DEFAULT_NAME="qwen3_4b_grpo_epp_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager"
    # EPP_CONFIG selects a scorer/producer variant from the llmd-epp-configs
    # ConfigMap without touching the mode; default is the burst profile.
    EPP_CONFIG_FILE="${EPP_CONFIG:-epp-config.yaml}"
    ;;

  epp-inflight)
    # EPP routing on the in-flight counter, no cap. epp_report_completion keeps
    # the stream open so the counter stays honest.
    DEFAULT_NAME="qwen3_4b_grpo_eppinflight_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager"
    EPP_CONFIG_FILE="${EPP_CONFIG:-epp-config-inflight.yaml}"
    EPP_REPORT_COMPLETION="true"
    ;;

  epp-fc)
    # EPP routing + a per-endpoint concurrency cap (flow control queues over-cap
    # requests). Sweep the cap via EPP_CONFIG.
    DEFAULT_NAME="qwen3_4b_grpo_eppfc_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager"
    EPP_CONFIG_FILE="${EPP_CONFIG:-epp-config-inflight-cap.yaml}"
    EPP_REPORT_COMPLETION="true"
    ;;

  wave-admission)
    # Estimation-gated admission (no EPP), sticky after admit. Tunables via
    # actor_rollout_ref.rollout.custom.wave_admission_* - see wave_admission/.
    DEFAULT_NAME="qwen3_4b_grpo_waveadmission_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.wave_admission.agent_loop_manager.WaveAdmissionAgentLoopManager"
    ;;

  wave-admission-p2p)
    # Wave-admission on the P2P backend: a migration pulls the resident's KV.
    # The P2P tier deadlocks engine sleep(), so pass
    # free_cache_engine=false via EXTRA_OVERRIDES to EVERY arm of a comparison.
    DEFAULT_NAME="qwen3_4b_grpo_waveadmissionp2p_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.wave_admission.agent_loop_manager.WaveAdmissionAgentLoopManager"
    ROLLOUT_NAME="vllm-llmd-p2p"
    EXTERNAL_LIB="llm_d_rl_verl_integration.register_p2p"
    P2P_ENGINE_HYDRA="${P2P_ENGINE_BASE} \
      +actor_rollout_ref.rollout.custom.wave_admission_p2p_kv_available=true \
      +actor_rollout_ref.rollout.custom.wave_admission_reserve_mode=size \
      +actor_rollout_ref.rollout.custom.wave_admission_reserve_z=1.5 \
      +actor_rollout_ref.rollout.custom.wave_admission_max_wait_s=${WAVE_ADMISSION_MAX_WAIT_S:-20}"
    # Skip the sidecar and POST to vLLM's native endpoint. "enabled" not "true":
    # Ray's runtime_env.env_vars rejects the bool Hydra infers from "true".
    if [[ "${WAVE_ADMISSION_P2P_NOSIDECAR:-false}" == "true" ]]; then
      P2P_ENGINE_HYDRA="${P2P_ENGINE_HYDRA} \
      +ray_kwargs.ray_init.runtime_env.env_vars.VERL_P2P_NOSIDECAR=enabled \
      +actor_rollout_ref.rollout.custom.wave_admission_p2p_nosidecar=true \
      +actor_rollout_ref.rollout.custom.wave_admission_p2p_port=${WAVE_ADMISSION_P2P_PORT:-7777}"
    fi
    ;;

  epp-p2p)
    # P2P KV-cache sharing with EPP routing: a per-replica sidecar turns EPP's
    # source header into kv_transfer_params. Same sleep() caveat as above.
    DEFAULT_NAME="qwen3_4b_grpo_eppp2p_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager"
    EPP_CONFIG_FILE="${EPP_CONFIG:-epp-config-p2p.yaml}"
    ROLLOUT_NAME="vllm-llmd-p2p"
    EXTERNAL_LIB="llm_d_rl_verl_integration.register_p2p"
    P2P_ENGINE_HYDRA="${P2P_ENGINE_BASE}"
    ;;

  epp-sglang)
    # EPP direct-gRPC routing with SGLang replicas instead of vLLM. rollout.name=sglang
    # is a verl BUILT-IN backend (no register_pd.py-style import hook needed, unlike
    # epp-p2p's vllm-llmd-p2p above). No PD/P2P for SGLang in this mode - see
    # llmd_epp_sglang/agent_loop_manager.py.
    DEFAULT_NAME="qwen3_4b_grpo_eppsglang_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.llmd_epp_sglang.agent_loop_manager.SglangEPPRouterAgentLoopManager"
    EPP_CONFIG_FILE="${EPP_CONFIG:-epp-config.yaml}"
    ROLLOUT_NAME="sglang"
    ;;

  llm-d)
    echo "ERROR: --mode llm-d is not yet implemented"
    exit 1
    ;;

  *)
    echo "ERROR: unknown mode '${MODE}'. Choose: native | epp | epp-inflight | epp-fc | epp-p2p | epp-sglang | wave-admission | wave-admission-p2p | llm-d"
    exit 1
    ;;
esac

# Overrides common to every mode; epp_config_file only when the mode set one.
# trainer.use_v1=true is mandatory: every llm-d manager subclasses verl's
# AgentLoopManagerTQ, which only the v1 trainer drives correctly.
EXTRA_HYDRA="
  trainer.use_v1=true \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=${AGENT_LOOP_MANAGER_CLASS}"
if [[ -n "$EPP_CONFIG_FILE" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/${EPP_CONFIG_FILE}"
  # LlmdActor (started by whichever Ray actor calls _on_servers_ready - verl's
  # own unpinned TaskRunnerV1 driver, so this can be a GPU worker, not just the
  # head) reads VERL_EPP_BINARY/VERL_ENVOY_BINARY/VERL_SIDECAR_BINARY from
  # os.environ at import time. Ray's job-level runtime_env only guarantees
  # env_vars explicitly listed here reach actors wherever they run - it does
  # NOT fall back to the pod's own container env for actors spawned under a
  # job that already set an explicit runtime_env. Forward whichever of these
  # are actually set on this pod (empty/unset ones are omitted rather than
  # forwarded as blank, so llmd_actor.py's own os.environ.get(...) defaults
  # still apply when a binary genuinely isn't provided).
  for v in VERL_EPP_BINARY VERL_ENVOY_BINARY VERL_SIDECAR_BINARY; do
    if [[ -n "${!v:-}" ]]; then
      EXTRA_HYDRA="${EXTRA_HYDRA} \
  +ray_kwargs.ray_init.runtime_env.env_vars.${v}=${!v}"
    fi
  done
fi
if [[ -n "$EPP_REPORT_COMPLETION" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  +actor_rollout_ref.rollout.custom.epp_report_completion=${EPP_REPORT_COMPLETION}"
fi
if [[ -n "$ROLLOUT_NAME" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  actor_rollout_ref.rollout.name=${ROLLOUT_NAME}"
fi
if [[ -n "$EXTERNAL_LIB" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  +actor_rollout_ref.model.external_lib=${EXTERNAL_LIB}"
fi
if [[ -n "$P2P_ENGINE_HYDRA" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  ${P2P_ENGINE_HYDRA}"
fi
EXTRA_HYDRA="${EXTRA_HYDRA} \
  +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml"

EXPERIMENT_NAME="${CUSTOM_NAME:-$DEFAULT_NAME}"

# -- reqlog override -----------------------------------------------------------
if [[ "$REQLOG" == "on" ]]; then
  EXTRA_HYDRA="
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_REQLOG_DIR=/tmp/verl/reqlog${EXTRA_HYDRA}"
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
  hydra.run.dir=/tmp/hydra-outputs${EXTRA_HYDRA}
