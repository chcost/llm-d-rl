#!/bin/bash
# Autonomous A/B orchestrator: wait Run A (EPP, already running) -> collect ->
# launch Run B (baseline) -> collect. One auto-retry per run on failure.
set -uo pipefail
# Namespace is per-user and comes from the environment. Mandatory, no default.
NS="${NAMESPACE:?NAMESPACE not set - export NAMESPACE=<your-namespace>}"
BASE="${RESULTS_DIR:-./verl-results}"
mkdir -p "$BASE"
say(){ echo "[$(date '+%H:%M:%S')] $*"; }

hpod(){ kubectl get pod -n $NS -l ray.io/node-type=head   -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }
wpod(){ kubectl get pod -n $NS -l ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }

# wait until no main_ppo process on head (run finished/exited)
wait_exit(){
  while true; do
    local H; H=$(hpod)
    [ -z "$H" ] && { sleep 20; continue; }
    if ! kubectl exec -n $NS "$H" -- pgrep -f main_ppo >/dev/null 2>&1; then break; fi
    sleep 20
  done
}
# success if the log shows final validation or step 40
ok40(){
  local lf=$1 H; H=$(hpod)
  kubectl exec -n $NS "$H" -- grep -qE 'Final validation|global_step:40' "$lf" 2>/dev/null
}
clean_reqlog(){
  # pre-launch prep: wipe only the ACCUMULATING dirs (pid-named reqlogs, experiment-named
  # jsonl logs) so they don't pile up. generations/ is keyed by step number and is overwritten
  # by the next run, so we leave it.
  local H W; H=$(hpod); W=$(wpod)
  kubectl exec -n $NS "$H" -- bash -c 'rm -rf /tmp/verl/reqlog /tmp/verl/logs; mkdir -p /tmp/verl/reqlog' 2>/dev/null
  kubectl exec -n $NS "$W" -- bash -c 'rm -rf /tmp/verl/reqlog; mkdir -p /tmp/verl/reqlog' 2>/dev/null
}
# ACCUMULATING files -> md5-verified DELETE. Pod *.jsonl in poddir whose md5 matches local copy.
vdel(){  # $1=pod $2=poddir $3=localdir ; echoes "<deleted>/<total>"
  local pod=$1 poddir=$2 localdir=$3 m name del=() tot=0
  while read -r m name; do
    [ -z "$name" ] && continue; tot=$((tot+1))
    if [ -f "$localdir/$name" ] && [ "$(md5sum "$localdir/$name" 2>/dev/null | awk '{print $1}')" = "$m" ]; then
      del+=("$poddir/$name")
    fi
  done < <(kubectl exec -n $NS "$pod" -- bash -c "cd '$poddir' 2>/dev/null && md5sum *.jsonl 2>/dev/null")
  [ ${#del[@]} -gt 0 ] && kubectl exec -n $NS "$pod" -- rm -f "${del[@]}" 2>/dev/null
  echo "${#del[@]}/$tot"
}
del1(){  # ACCUMULATING single file -> delete pod $2 iff md5 == local $3
  local H=$1 pf=$2 lf=$3
  [ -f "$lf" ] || return
  [ "$(kubectl exec -n $NS "$H" -- md5sum "$pf" 2>/dev/null | awk '{print $1}')" = "$(md5sum "$lf" 2>/dev/null | awk '{print $1}')" ] \
    && kubectl exec -n $NS "$H" -- rm -f "$pf" 2>/dev/null
}
# OVERWRITTEN files -> md5-VALIDATE only, no delete (next run overwrites them in place).
vcheck(){  # $1=pod $2=poddir $3=localdir ; echoes "<verified>/<total>"
  local pod=$1 poddir=$2 localdir=$3 m name ok=0 tot=0
  while read -r m name; do
    [ -z "$name" ] && continue; tot=$((tot+1))
    [ -f "$localdir/$name" ] && [ "$(md5sum "$localdir/$name" 2>/dev/null | awk '{print $1}')" = "$m" ] && ok=$((ok+1))
  done < <(kubectl exec -n $NS "$pod" -- bash -c "cd '$poddir' 2>/dev/null && md5sum *.jsonl 2>/dev/null")
  echo "$ok/$tot"
}
check1(){  # OVERWRITTEN single file -> validate only; echoes ok|MISMATCH|nolocal
  local H=$1 pf=$2 lf=$3
  [ -f "$lf" ] || { echo nolocal; return; }
  [ "$(kubectl exec -n $NS "$H" -- md5sum "$pf" 2>/dev/null | awk '{print $1}')" = "$(md5sum "$lf" 2>/dev/null | awk '{print $1}')" ] && echo ok || echo MISMATCH
}
collect(){
  local name=$1 scr=$2 lf=$3 d="$BASE/$1" H W; H=$(hpod); W=$(wpod)
  mkdir -p "$d"
  # stop any scraper first so the CSV is final before copying
  [ "$scr" = yes ] && kubectl exec -n $NS "$H" -- pkill -f vllm_scrape.py 2>/dev/null || true
  kubectl cp "$NS/$H:$lf" "$d/console.log" >/dev/null 2>&1 || true
  kubectl cp "$NS/$H:/tmp/verl/logs" "$d/logs" >/dev/null 2>&1 || true
  kubectl cp "$NS/$H:/tmp/verl/generations" "$d/generations" >/dev/null 2>&1 || true
  kubectl cp "$NS/$H:/tmp/verl/reqlog" "$d/reqlog_head" >/dev/null 2>&1 || true
  kubectl cp "$NS/$W:/tmp/verl/reqlog" "$d/reqlog_worker" >/dev/null 2>&1 || true
  [ "$scr" = yes ] && { kubectl cp "$NS/$H:/tmp/vllm_metrics.csv" "$d/vllm_metrics.csv" >/dev/null 2>&1 || true; }
  say "collected $name -> $d ($(find "$d" -type f 2>/dev/null | wc -l) files)"
  # --- ACCUMULATING (pile up across runs) -> verified delete ---
  local nh nw nl=0
  nh=$(vdel "$H" /tmp/verl/reqlog "$d/reqlog_head")
  nw=$(vdel "$W" /tmp/verl/reqlog "$d/reqlog_worker")
  for pf in $(kubectl exec -n $NS "$H" -- bash -c 'ls /tmp/verl/logs/*/*.jsonl 2>/dev/null'); do
    wl=$(find "$d/logs" -name "$(basename "$pf")" 2>/dev/null | head -1)
    [ -n "$wl" ] && { del1 "$H" "$pf" "$wl"; nl=$((nl+1)); }
  done
  kubectl exec -n $NS "$H" -- bash -c 'rm -rf /tmp/verl/reqlog /tmp/verl/logs' 2>/dev/null || true
  kubectl exec -n $NS "$W" -- bash -c 'rm -rf /tmp/verl/reqlog' 2>/dev/null || true
  # --- OVERWRITTEN (replaced in place next run) -> validate only, NOT deleted ---
  local vg cc cg
  vg=$(vcheck "$H" /tmp/verl/generations/train "$d/generations/train")
  cg=$(check1 "$H" "$lf" "$d/console.log")
  cc=na; [ "$scr" = yes ] && cc=$(check1 "$H" /tmp/vllm_metrics.csv "$d/vllm_metrics.csv")
  say "cleanup: DELETED accumulating reqlog_head=$nh reqlog_worker=$nw logs=$nl | VALIDATED(kept) generations=$vg console=$cg csv=$cc"
}
launch(){  # $1 = run_A.sh|run_B.sh  $2 = logfile
  local H; H=$(hpod)
  kubectl exec -n $NS "$H" -- bash -c "nohup bash /tmp/$1 > $2 2>&1 & echo launched $1 pid \$!"
  sleep 90  # let main_ppo spawn before we start waiting for exit
}

############ RUN A (already running) ############
say "=== waiting for Run A (EPP, /tmp/train-tp1.log) to finish ==="
wait_exit
if ok40 /tmp/train-tp1.log; then
  say "Run A completed (reached step 40)"
else
  say "Run A did NOT reach step 40 -> resubmitting once"
  clean_reqlog
  kubectl exec -n $NS "$(hpod)" -- pkill -f vllm_scrape.py 2>/dev/null || true
  kubectl exec -n $NS "$(hpod)" -- bash -c 'rm -f /tmp/vllm_metrics.csv; nohup python3 /tmp/vllm_scrape.py >/tmp/vllm_scrape.out 2>&1 &' 2>/dev/null || true
  launch run_A.sh /tmp/train-tp1.log
  wait_exit
  ok40 /tmp/train-tp1.log && say "Run A (retry) completed" || say "Run A (retry) STILL failed - collecting whatever exists"
fi
collect run2_epp_tp1_n8_40s yes /tmp/train-tp1.log

# Baseline (Run B) intentionally skipped per user request ("leave the baseline for now").

say "=== DONE: Run A finished and collected under $BASE/run2_epp_tp1_n8_40s ==="
