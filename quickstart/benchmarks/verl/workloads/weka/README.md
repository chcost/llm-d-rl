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

## What this measures, and what it does not

**This is a timing harness, not an RL task.** The traces carry token counts and
timing but **no prompt text**, so there is no ground truth and no meaningful
reward: `trace_player/dummy_reward.py` returns a constant, which gives a
degenerate GRPO advantage. The training step still runs end to end, so the
gen/train ratio is real - but nothing here trains a better model. Use it to
compare routing and admission on a realistic conversation *shape*, and use the
other workloads (gsm8k, hotpotqa, arxiv, scotus_xl) when reward matters.

## 1. Get the data (not in this repo)

The source is the public HuggingFace dataset
[`semianalysisai/cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126):
393 sessions, 98,827 model calls (56.8k main turns + 42.0k subagent inner
requests), built from real Claude Code CLI sessions. `traces.jsonl` is ~1.85 GB.

```bash
pip install huggingface_hub
huggingface-cli download semianalysisai/cc-traces-weka-062126 traces.jsonl \
  --repo-type dataset --local-dir /tmp/weka
```

Sibling dated variants exist on the same org (`-061326` 183 traces, `-061526`
233 traces, `-with-subagents-060226/060526/060826`); everything recorded in
HANDOVER.md used `-062126`.

Raw schema, one JSON object per line = one session:

```
{"id", "models", "block_size": 64, "hash_id_scope": "local",
 "requests": [ {"t", "type": "n"|"s", "in", "out", "api_time", "ttft", "hash_ids"}
             | {"type": "subagent", "t", "requests": [ ... ]} ]}
```

`t` is seconds from session start; the off-GPU gap before a call is
`max(0, t - (prev.t + prev.api_time))`. Conversations have no stable id, so
`make_weka.py` uses the **trace line index** as `conv_id`.

An existing local copy on this workstation:
`~/work/rl-work/agentic/cc-traces-weka-062126/traces.jsonl`.

**`preselected_groups.json`** (optional, `--groups-file`) pins which
conversations a given N selects, for parity with the scheduling simulator. Its
keys are N, its values are trace line indices. Ours:
`~/work/rl-work/agentic/analysis/results/preselected_groups.json`. Without it,
selection is deterministic from `--seed` but not identical to the simulator's.

## 2. Build the parquet

```bash
python make_weka.py --traces /tmp/weka/traces.jsonl --local_dir <out-dir> --n <N> [flags]
```

The flags that change what you measure:

| Flag | Effect |
|---|---|
| `--n` | number of conversations |
| `--cap-input-tokens` | **stop** a conversation at the first turn whose input would exceed this; that turn and all later ones are dropped rather than clamped, so replayed turns keep their real distinct sizes |
| `--cap-output-tokens` | truncate each turn's output |
| `--gap-cap` | cap the off-GPU gap before each turn, in seconds (default 10). `0` = back-to-back replay, no think time |
| `--no-gap-cap` | keep the real gaps, uncapped |
| `--max-turns-per-conv` | truncate each conversation to its first K turns; weka sessions average ~145 turns, so this bounds rollout and lets N scale |
| `--include-subagents` | flatten subagent inner calls into the serial stream by start time (concurrency flattened, total work preserved). Default is the main chain only |
| `--threshold` | standalone selection: drop conversations with any input >= this. Set high (e.g. `999999999`) when you are capping instead |
| `--groups-file`, `--seed` | which conversations get picked |

Gap capping applies at **build** time (`--gap-cap`, baked into the parquet) and
again at **run** time (`VERL_TRACE_GAP_CAP_S`), so a dataset built with gaps can
still be replayed back-to-back without rebuilding.

### The datasets HANDOVER.md's results were measured on

These live on the `verl-cache` PVC at `/tmp/verl/data/`. Values below were read
back out of the parquet, not copied from notes:

| Dataset | convs | turns | turns/conv min/med/max | max input | max output | max gap |
|---|---|---|---|---|---|---|
| `weka_ctxc64k_n256` | 256 | **1455** | 1 / 4 / 20 | 64000 | 2048 | 0.0 |
| `weka_ctxc64k_n256_gap15` | 256 | 1455 | 1 / 4 / 20 | 64000 | 2048 | **15.0** |
| `weka_ctxc64k_n32` | 32 | **172** | 1 / 4 / 20 | 64000 | 2048 | 0.0 |

`ctxc64k` means `--cap-input-tokens 64000`. The base sets replay **back-to-back
with zero think time**; only `gap15` preserves real inter-turn gaps, which is why
its arms show much lower GPU occupancy.

```bash
# weka_ctxc64k_n256 - the 1455-turn set most arms use
python make_weka.py --traces /tmp/weka/traces.jsonl \
  --local_dir /tmp/verl/data/weka_ctxc64k_n256 \
  --n 256 --seed 0 --threshold 999999999 \
  --cap-input-tokens 64000 --cap-output-tokens 2048 --gap-cap 0

# weka_ctxc64k_n256_gap15 - same selection, real gaps up to 15s
python make_weka.py --traces /tmp/weka/traces.jsonl \
  --local_dir /tmp/verl/data/weka_ctxc64k_n256_gap15 \
  --n 256 --seed 0 --threshold 999999999 \
  --cap-input-tokens 64000 --cap-output-tokens 2048 --gap-cap 15
```

`weka_ctxc64k_n32` is a **32-conversation subset of the n256 selection,
stratified by turn count** (sort by turns, take every 8th) so it is not biased
toward the 69 single-turn sessions - its turn distribution matches n256 exactly.
It exists to make GRPO groups affordable: 172 turns x `val_kwargs.n=8` = 1376
turns. There is no `make_weka.py` flag for that stratification; HANDOVER.md
records it as a manual selection step, and the exact command was not kept. Verify
any rebuild against the table above before comparing to banked results.

### Turn count is the check that the dataset is what you think

`num_turns` in the run's own metrics must match the table: 1455 for n256, and
1376 (172 x 8) for n32 at `val_kwargs.n=8`. A silently different count is how
six arms once ran at n=1 while reporting "config verified".

## 3. Deploy config

Mount `trace_player_agent_loop.yaml` into the `/etc/llmd-configs` ConfigMap
(same place as `epp-config.yaml` and the searchr1 tool config) - `deploy.sh`
does this already. `task.env` points `agent_loop_config_path` at it. The
`epp-inflight`/`epp-fc` modes' EPP configs are workload-agnostic and are
composed, not stored: `epp-config-inflight.yaml` is `base.yaml` +
`profiles/inflight.yaml`, and `epp-config-inflight-cap.yaml` adds
`modifiers/cap.yaml`. See `integrations/common/src/llm_d_rl_common/configs/epp/variants.yaml`.

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

`profiles/inflight.yaml` supplies `inflight-load-producer` +
`active-request-scorer` (rendered as `epp-config-inflight.yaml`). Watch per-endpoint in-flight counts on the EPP metrics
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
and `../../scripts/vllm_scrape.py` for vLLM-side metrics.

## Files

| File | Purpose |
|---|---|
| `make_weka.py` | traces.jsonl -> normalized trace-player parquet (self-contained) |
| `task.env` | sets `default_agent_loop=trace_player` + `agent_loop_config_path` (loop only; router is `--mode`) |
| `trace_player_agent_loop.yaml` | registers the `trace_player` loop (mount into /etc/llmd-configs) |
| `epp-config-inflight-example.yaml` | a worked concurrency-cap config, kept as a reference for what `modifiers/cap.yaml` renders to. To sweep the cap, add a variant to `variants.yaml` and select it with `EPP_CONFIG` |

The loop itself lives in the importable package at
`quickstart/benchmarks/verl/src/llm_d_rl_verl_bench/trace_player/`. The in-flight producer +
active-request-scorer EPP config and its concurrency-capped variant are
workload-agnostic and are composed from `integrations/common/src/llm_d_rl_common/configs/epp/`
(`profiles/inflight.yaml`, `modifiers/cap.yaml`); completion reporting is a flag
(`custom.epp_report_completion`) on the regular `llmd_epp.EPPLLMClient`, not a
separate client class.
