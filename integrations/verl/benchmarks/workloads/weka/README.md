# weka - Weka CC-traces replay (trace_player)

Replays real Claude-Code agentic conversation *shapes* (per-turn input/output
token counts + inter-turn off-GPU gaps - there is **no prompt text**) through
verl -> vLLM, routed either **native** (verl's `GlobalRequestLoadBalancer`),
**epp** (EPP ext_proc routing), or **epp-fc** (EPP routing + a working in-flight
counter, so a per-endpoint concurrency cap can bind). Used to compare
routing/admission on the weka shape and to reproduce the simulator's
free-migration + per-GPU capacity (C) findings on a real 8-GPU deployment.

The router-agnostic `trace_player` agent loop synthesizes deterministic,
prefix-nesting `prompt_ids` per conversation (length = trace input tokens) and
forces output length (`max_tokens = min_tokens = out, ignore_eos`), so GPU load
matches the trace without the original text. Inter-turn gaps are injected as
non-blocking `asyncio.sleep` (cap via `VERL_TRACE_GAP_CAP_S`, default 10s).

## 1. Get the data (not in this repo)

`traces.jsonl` is ~1.85 GB (393 sessions, ~98k model calls). Download from
HuggingFace:

```bash
# TODO: replace <ORG>/<DATASET> with the published dataset id.
pip install huggingface_hub
huggingface-cli download <ORG>/weka-cc-traces-062126 traces.jsonl \
  --repo-type dataset --local-dir /tmp/weka
```

Or use an existing local copy (e.g. the simulator's at
`~/work/rl-work/agentic/cc-traces-weka-062126/traces.jsonl`). For parity with the
simulator's fixed conversation groups, also grab `preselected_groups.json` and
pass it via `--groups-file` (its group keys are N, values are trace line indices).

## 2. Build the parquet

Smoke test (1-2 small conversations that fit an 8k context):

```bash
python make_weka.py --traces /tmp/weka/traces.jsonl \
  --local_dir /tmp/verl/data/weka --n 2 --max-total-tokens 8192
```

Parity run (simulator's fixed groups, GAP=10, subagents flattened; needs a large
context / raised DEF_MAXP so big turns are not skipped):

```bash
python make_weka.py --traces /tmp/weka/traces.jsonl \
  --groups-file /path/to/preselected_groups.json --n 128 \
  --gap-cap 10 --include-subagents --max-total-tokens 0 --local_dir /tmp/verl/data/weka
```

Notes:
- Default linearizes the **main chain** only; `--include-subagents` flattens
  subagent inner calls into the serial stream by start time (concurrency
  flattened, total work preserved). True subagent concurrency (asyncio.gather)
  is a future extension.
- `conv_id` = trace line index (matches `preselected_groups.json`).

## 3. Deploy config

Mount `trace_player_agent_loop.yaml` into the `/etc/llmd-configs` ConfigMap
(same place as `epp-config.yaml` and the searchr1 tool config) - `deploy.sh`
does this already. `task.env` points `agent_loop_config_path` at it. The
`epp-inflight`/`epp-fc` modes' EPP configs (`epp-config-inflight.yaml`,
`epp-config-inflight-cap.yaml`) are workload-agnostic and live in `deploy/`
alongside `epp-config.yaml`, not here.

## 4. Run (native vs EPP vs EPP+flow-control)

```bash
scripts/run_on_head.sh --mode native --task weka --steps 1 --tp 1 --n 1
scripts/run_on_head.sh --mode epp    --task weka --steps 1 --tp 1 --n 1
scripts/run_on_head.sh --mode epp-fc --task weka --steps 1 --tp 1 --n 1
```

Use `--n 1` for deterministic replay; `--n G` to study GRPO fan-out. Keep the
effective C identical across native/epp (vLLM `--max-num-seqs`) and sweep only the
EPP flow-control cap separately, or routing vs admission is confounded.

## 5. Verify the EPP in-flight counter (epp-fc)

`deploy/epp-config-inflight.yaml` includes `inflight-load-producer` +
`active-request-scorer`. Watch per-endpoint in-flight counts on the EPP metrics
port (9090):

```bash
watch -n0.5 'curl -s <epp-host>:9090/debug/plugins/state | jq .'
```

During a generation the chosen endpoint's `Requests`/`Tokens` rise (increment on
PreRequest); after completion they return to 0 **iff** the decrement fired:
- with `--mode epp-fc` (`epp_report_completion=true`, EPPLLMClient's begin/complete
  path): counts drop promptly;
- with `--mode epp` (`epp_report_completion` unset, EPPLLMClient's pick path):
  counts stay up until the 5-min TTL janitor - the "counter only goes up"
  behaviour the completion path fixes.

Other instruments: the per-request JSONL reqlog (endpoint, pick_s, gen_s, tokens)
and `benchmarks/scripts/vllm_scrape.py` for vLLM-side metrics.

## Files

| File | Purpose |
|---|---|
| `make_weka.py` | traces.jsonl -> normalized trace-player parquet (self-contained) |
| `task.env` | sets `default_agent_loop=trace_player` + `agent_loop_config_path` (loop only; router is `--mode`) |
| `trace_player_agent_loop.yaml` | registers the `trace_player` loop (mount into /etc/llmd-configs) |
| `epp-config-inflight-example.yaml` | example concurrency-cap config; copy and point `EPP_CONFIG` at your own variant to sweep |

The loop itself lives in the importable package at
`src/llm_d_rl_verl_integration/trace_player/`. The in-flight producer +
active-request-scorer EPP config and its concurrency-capped variant are
workload-agnostic and live in `deploy/epp-config-inflight.yaml` /
`deploy/epp-config-inflight-cap.yaml`; completion reporting itself is a flag
(`custom.epp_report_completion`) on the regular `llmd_epp.EPPLLMClient`, not a
separate client class.
