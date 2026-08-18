#!/usr/bin/env bash
# Launch a training run on the RayCluster head from your laptop.
#
# Resolves the head pod by its Ray label (namespace from $NAMESPACE), copies
# run_test.sh AND the selected workloads/<task>/ folder into it, then runs it
# there. All arguments except this script's own --fg flag are passed straight
# through to run_test.sh.
#
# Usage:
#   scripts/run_on_head.sh --mode epp                 # background on pod + tail the log
#   scripts/run_on_head.sh --mode epp --steps 20 --tp 2
#   scripts/run_on_head.sh --fg --mode native         # run attached (foreground)
#
# Modes / options are run_test.sh's: --mode native|epp, --steps, --tp, --n, --name, --reqlog.
#
# Execution model:
#   default  - nohup run_test.sh on the head into /tmp/train.log, then tail -f it.
#              The run survives a laptop disconnect; Ctrl-C only detaches the tail.
#   --fg     - run attached; output streams live, but dropping the connection
#              kills the run. Fine for short interactive tests.
#
# Requires: kubectl on PATH with a valid context (set KUBECONFIG as needed) and
# the target namespace exported: export NAMESPACE=<your-namespace>.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_LOG="/tmp/train.log"

# Namespace is per-user and comes from the environment. Mandatory, no default.
NS="${NAMESPACE:?NAMESPACE not set - export NAMESPACE=<your-namespace>}"

# Split out our own --fg flag; everything else is forwarded to run_test.sh. We also
# peek at --task (still forwarded) so we know which workloads/<task>/ folder to ship.
FG=0
PASS=()
TASK="gsm8k"
want_task=0
for arg in "$@"; do
  if [ "$want_task" -eq 1 ]; then TASK="$arg"; want_task=0; PASS+=("$arg"); continue; fi
  case "$arg" in
    --fg) FG=1 ;;
    --task) want_task=1; PASS+=("$arg") ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) PASS+=("$arg") ;;
  esac
done

HEAD="$(kubectl get pod -n "$NS" -l ray.io/node-type=head \
  -o jsonpath='{.items[0].metadata.name}')"
[ -n "$HEAD" ] || { echo "ERROR: no head pod (label ray.io/node-type=head) in namespace $NS" >&2; exit 1; }
echo "==> head pod: $HEAD (namespace $NS)"

echo "==> copying run_test.sh to $HEAD:/tmp/run_test.sh"
kubectl cp "$SCRIPT_DIR/run_test.sh" "$NS/$HEAD:/tmp/run_test.sh"

# Ship the selected workload folder so run_test.sh can source workloads/<task>/task.env
# on the pod (it falls back to /tmp/workloads when run from /tmp/run_test.sh).
WLDIR="$SCRIPT_DIR/../workloads/$TASK"
if [ -d "$WLDIR" ]; then
  echo "==> copying workload '$TASK' to $HEAD:/tmp/workloads/$TASK"
  kubectl exec -n "$NS" "$HEAD" -- mkdir -p /tmp/workloads
  kubectl cp "$WLDIR" "$NS/$HEAD:/tmp/workloads/$TASK"
else
  echo "WARNING: no workload folder at $WLDIR - run_test.sh will error on the pod for --task $TASK" >&2
fi

# Ship helper scripts run_test.sh calls on the pod (it falls back to /tmp/utils).
if [ -d "$SCRIPT_DIR/utils" ]; then
  echo "==> copying scripts/utils to $HEAD:/tmp/utils"
  kubectl exec -n "$NS" "$HEAD" -- mkdir -p /tmp/utils
  kubectl cp "$SCRIPT_DIR/utils/strip_dca_config.py" "$NS/$HEAD:/tmp/utils/strip_dca_config.py"
fi
# run_test.sh always starts the /metrics scraper; ship it to the same fallback dir.
if [ -f "$SCRIPT_DIR/vllm_scrape.py" ]; then
  kubectl exec -n "$NS" "$HEAD" -- mkdir -p /tmp/utils
  kubectl cp "$SCRIPT_DIR/vllm_scrape.py" "$NS/$HEAD:/tmp/utils/vllm_scrape.py"
fi

# Forward the env knobs run_test.sh reads. kubectl exec does NOT inherit the
# caller's environment, so without this every override is silently dropped and
# the run quietly uses task.env defaults - a wrong model or context length that
# still produces a plausible-looking result. Values may contain spaces
# (EXTRA_OVERRIDES is a whole hydra argument list), so they travel as separate
# argv elements and are re-expanded by `env` on the pod, never re-split by a
# shell. Add a name here when run_test.sh learns a new knob.
FORWARD=(MODEL_PATH TRAIN_FILE TEST_FILE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH
         TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE EXTRA_OVERRIDES CUSTOM_NAME
         PROJECT_NAME SKIP_DCA_STRIP WORKLOADS_DIR
         EPP_CAP_CONFIG EPP_P2P_CONFIG EPP_CONFIG P2P_CPU_BYTES_TO_USE
         WAVE_ADMISSION_MAX_WAIT_S WAVE_ADMISSION_P2P_NOSIDECAR WAVE_ADMISSION_P2P_PORT
         VLLM_SCRAPE_OUT VLLM_SCRAPE_ENDPOINTS VLLM_SCRAPE_HOST)
ENVV=()
for v in "${FORWARD[@]}"; do
  [ -n "${!v:-}" ] && ENVV+=("$v=${!v}")
done
if [ "${#ENVV[@]}" -gt 0 ]; then
  echo "==> forwarding to the pod:"
  printf '      %s\n' "${ENVV[@]}"
fi

if [ "$FG" -eq 1 ]; then
  echo "==> running attached (foreground): run_test.sh ${PASS[*]}"
  exec kubectl exec -it -n "$NS" "$HEAD" -- env "${ENVV[@]}" bash /tmp/run_test.sh "${PASS[@]}"
fi

echo "==> launching in background on the pod: run_test.sh ${PASS[*]}"
kubectl exec -n "$NS" "$HEAD" -- bash -c \
  'nohup env "$@" > "'"$REMOTE_LOG"'" 2>&1 & echo "launched pid $!"' \
  _ "${ENVV[@]}" bash /tmp/run_test.sh "${PASS[@]}"

cat <<EOF
==> streaming $REMOTE_LOG (Ctrl-C detaches the tail; the run keeps going on the pod)
    reattach later:  kubectl exec -n $NS $HEAD -- tail -f $REMOTE_LOG
    stop the run:    kubectl exec -n $NS $HEAD -- pkill -f main_ppo
EOF
exec kubectl exec -n "$NS" "$HEAD" -- tail -f "$REMOTE_LOG"
