# Instrumentation runbook (verl + llm-d EPP runs)

How to run an instrumented experiment on a fresh cluster/pod.

Namespace: from the `NAMESPACE` environment variable (export it first). Helpers used throughout:
```bash
NS="${NAMESPACE:?export NAMESPACE=<your-namespace> first}"
H=$(kubectl get pod -n $NS -l ray.io/node-type=head   -o jsonpath='{.items[0].metadata.name}')
W=$(kubectl get pod -n $NS -l ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}')
REPO=~/path/to/llm-d-rl-verl-integration
```
Note: the worker pod has two containers - add `-c ray-worker` to `kubectl exec`/`kubectl cp` for it.

## 0. Prereqs

- Cluster up, both pods `1/1 Running` (`kubectl get pods -n $NS`).
- `kubectl`/`oc` auth valid. OpenShift tokens expire - if you see "Unauthorized", `oc login` again.
  For a multi-hour unattended run, mint a token that outlives the run (~80 min + setup).
- The integration package is installed in the image. Verify:
  ```bash
  kubectl exec -n $NS $H -- python3 -c \
    "import llm_d_rl_verl_integration.llmd_epp.llm_client as m; print(m.__file__)"
  ```

## 1. Ship benchmarks to the pod

Copy the entire `benchmarks/` directory to the head pod once — scripts, workload configs, and
data-prep scripts all land under `/tmp/benchmarks`:

```bash
kubectl cp $REPO/benchmarks $NS/$H:/tmp/benchmarks
```

### Data preparation

| Workload | Data prep script |
|----------|-----------------|
| `gsm8k` | verl `examples/data_preprocess/gsm8k.py` |
| `geo3k` | verl `examples/data_preprocess/geo3k.py` |
| `hotpotqa` | `make_hotpotqa.py` |
| `musique` | `make_musique.py` |
| `quality` | `make_quality.py` |
| `scotus_xl` | `make_scotus.py` |
| `searchr1` | `make_searchr1.py` |

For `gsm8k` and `geo3k`, use verl's own preprocessing scripts.

```bash
# gsm8k
kubectl exec -n $NS $H -- python3 /tmp/verl/verl/examples/data_preprocess/gsm8k.py --local_save_dir /tmp/verl/data/gsm8k
# geo3k
kubectl exec -n $NS $H -- python3 /tmp/verl/verl/examples/data_preprocess/geo3k.py --local_save_dir /tmp/verl/data/geo3k
```

For workloads with a custom script (`hotpotqa`, `musique`, `quality`, `scotus_xl`, `searchr1`),
run it once on the head pod before the first experiment. `--local_dir` is the destination
directory for the processed data - it must match `DEF_TRAIN`/`DEF_TEST` in the workload's `task.env`:

```bash
# replace <workload>, <script>, and <dest_dir> with values from the table
kubectl exec -n $NS $H -- python3 /tmp/benchmarks/workloads/<workload>/<script>.py --local_dir /tmp/verl/data/<dest_dir>
```

For example:
```bash
kubectl exec -n $NS $H -- python3 /tmp/benchmarks/workloads/scotus_xl/make_scotus.py --local_dir /tmp/verl/data/scotus_xl_cot
```

## 2. Per-request JSONL logging (reqlog)

The logging is **built into the package** - no pod patching required.
It is controlled entirely by the `VERL_REQLOG_DIR` environment variable:

- **EPP / llm-d modes** - `run_test.sh` sets `VERL_REQLOG_DIR=/tmp/verl/reqlog` automatically.
- **Native baseline** - reqlog is off by default; pass `--reqlog on` to enable it explicitly.

Each worker process writes `reqlog-<pid>.jsonl` under `VERL_REQLOG_DIR`.
Fields per record: `ts, request_id, endpoint, prompt_hash, prompt_tokens, output_tokens, pick_s, gen_s`.

`prompt_hash` is a BLAKE2b-8 digest of the token IDs - requests sharing the same prompt
(same GRPO group) will have matching hashes across replicas.

## 3. vLLM /metrics scraper - head only

Start the scraper before launching the run (the file is already on the pod from step 1).
It reads `/tmp/epp-endpoints.yaml` (written by EPP at startup) each loop, scrapes every
replica ~1.5s -> `/tmp/vllm_metrics.csv`. EPP-only (baseline writes no endpoints file).
Requires pyyaml (present in the image).

**Important caveat:** vLLM refreshes `prefix_cache_{hits,queries}_total` only every ~70-180s,
so per-step windowed deltas are unusable. Report prefix-cache hit rate as a whole-run aggregate only.

## 4. Launch script - head only

`run_test.sh` and the workload configs are already on the pod at `/tmp/benchmarks/scripts/` and
`/tmp/benchmarks/workloads/` from step 1.

Usage on the pod:
```bash
# EPP run
kubectl exec -n $NS $H -- bash /tmp/benchmarks/scripts/run_test.sh --mode epp
# baseline
kubectl exec -n $NS $H -- bash /tmp/benchmarks/scripts/run_test.sh --mode native
# custom knobs (add --task <name> to use a non-default workload, default: gsm8k)
kubectl exec -n $NS $H -- bash /tmp/benchmarks/scripts/run_test.sh --mode epp --steps 20 --tp 2 --n 4
```

The EPP config (plugins) is loaded from the file set by `rollout.custom.epp_config_file`.
In k8s this is mounted from the `llmd-epp-configs` ConfigMap (built from `common/src/llm_d_rl_common/configs/epp-config-burst.yaml`
or `deploy/epp-config-pd.yaml` - see `deploy/kuberay/README.md` Step 2).

## 5. Clean + start scraper + launch (head)

```bash
# wipe accumulating dirs (reqlog is pid-named, logs is experiment-named)
kubectl exec -n $NS $H --            bash -c 'rm -rf /tmp/verl/reqlog /tmp/verl/logs; mkdir -p /tmp/verl/reqlog'
kubectl exec -n $NS $W -c ray-worker bash -c 'rm -rf /tmp/verl/reqlog; mkdir -p /tmp/verl/reqlog'
# start scraper (truncate old CSV)
kubectl exec -n $NS $H -- bash -c 'rm -f /tmp/vllm_metrics.csv; nohup python3 /tmp/benchmarks/scripts/vllm_scrape.py >/tmp/vllm_scrape.out 2>&1 &'
# launch the run (add --task <name> to use a non-default workload, default: gsm8k)
kubectl exec -n $NS $H -- bash -c 'nohup bash /tmp/benchmarks/scripts/run_test.sh --mode epp > /tmp/train.log 2>&1 & echo launched pid $!'
# confirm it started (after ~60-90s)
kubectl exec -n $NS $H -- pgrep -fa main_ppo
```

## 6. Wait + collect

Use `rl_orchestrate.sh` (waits for main_ppo to exit, checks step count, collects artifacts with
md5-verified cleanup). Set `RESULTS_DIR` for the output location (default `./verl-results`), then plot with your own tooling.

Stop the scraper before collecting:
```bash
kubectl exec -n $NS $H -- pkill -f vllm_scrape.py
```

Collected per run: `console.log`, `logs/<jsonl>`, `generations/train/`, `reqlog_head/`,
`reqlog_worker/`, `vllm_metrics.csv`.

## Teardown (free GPUs)

```bash
bash deploy/kuberay/deploy.sh delete
# or scale down workerGroupSpecs replicas to 0
```
