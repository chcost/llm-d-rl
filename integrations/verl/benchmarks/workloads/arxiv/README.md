# arxiv workload

Large-input arXiv subject-area classification (scientific-domain analog of `scotus_xl`): read a full
arXiv paper and assign exactly one of 11 arXiv subject categories. Single-turn; very large input,
short output. Reward = normalized exact-match (`data_source=searchR1_nq`; `ground_truth.target=[code,
name]`, so the arXiv category code or its descriptive name scores). Builder: `make_arxiv.py`.

## Task and data

- Dataset: `ccdv/arxiv-classification`, config **`no_ref`** - full paper text (title, abstract, body)
  from the Long-Document-Dataset (He et al. 2019). 33k papers: ~28k train / 2.5k val / 2.5k test.
  The `no_ref` config strips in-document class references (e.g. `[cs.LG]` -> `[]`) so the label
  cannot leak into the prompt; the default config does not, so we do not use it.
- Labels: the 11 arXiv categories (math.AC Commutative Algebra, cs.CV Computer Vision and Pattern
  Recognition, cs.AI Artificial Intelligence, cs.SY Systems and Control, math.GR Group Theory,
  cs.CE Computational Engineering Finance and Science, cs.PL Programming Languages, cs.IT Information
  Theory, cs.DS Data Structures and Algorithms, cs.NE Neural and Evolutionary Computing,
  math.ST Statistics Theory). Slightly unbalanced.
- Input sizes: every document is > 4k tokens, with a long tail (raw text ranges from ~2.85k to
  ~2.55M characters). The prompt is capped at 24,576 tokens; the long tail is truncated (see below).
  Exact prompt-token percentiles are printed by the builder's `token_stats` when the data is built.
- Output: brief CoT then a single label; response capped at 2,048 tokens (non-clipping for a short
  label + brief reasoning). Use `--no_cot` for the direct single-label variant.

## Test setup

- Cluster: 1 worker node, 8x NVIDIA H200; vLLM TP=1 -> 8 independent replicas (one per GPU).
- Model: Qwen3-4B (FSDP1 actor/ref). GRPO `rollout.n=8`, `train_batch_size=256`
  -> 256 prompts x 8 samples = 2048 requests/step.
- Sequence budget: `max_prompt_length=24576`, `max_response_length=2048`;
  `filter_overlong_prompts=True` + `truncation=right` drop/clip the long paper tail.
- Routing compared:
  - native = verl least-in-flight load balancer (sticky session, request_id -> replica).
  - EPP as the verl endpoint picker = llm-d burst prefix-cache producer with `balanceBy: tokens`
    and `windowDurationMs: 1000` (covers the per-step rollout arrival burst).

## Results summary

Runs: `arxiv_{native,epp}_10s` (10 steps each, 8 replicas).

<img src="rollout-time-per-step.png" width="75%" alt="arXiv - rollout time per step: Native Verl vs Verl+llm-d-EPP">

Using EPP as the verl endpoint picker (vs native least-in-flight), steady-state over steps 2-10:

1. Mean rollout time per step reduced ~31% (reqlog full-span 148.2s -> 102.2s).
2. Slowest-replica (straggler) generation time reduced ~31% (147.4s -> 101.8s).
3. Per-request generation is faster at every percentile (the straggler tail shrinks).
4. Validation accuracy tracked together (native 0.662 -> 0.782, EPP 0.681 -> 0.764 over 10 steps) -
   no accuracy change from routing.

| metric | native | EPP (token-balanced burst) | diff |
|---|---|---|---|
| mean rollout / step (reqlog full-span) | 148.2 s | 102.2 s | -31.0% |
| mean rollout / step (verl `timing_s/gen`) | 151.7 s | 106.1 s | -30.1% |
| generate_sequences mean / replica | 81.7 s | 61.3 s | -25.0% |
| generate_sequences slowest (straggler) | 147.4 s | 101.8 s | -30.9% |
| straggler ratio (slowest / mean) | 1.80x | 1.66x | -7.8% |
| full step time (`timing_s/step`, training-dominated) | 955.3 s | 910.1 s | -4.7% |
| gen_s p50 / p90 / p99 (per request) | 82 / 132 / 145 s | 59 / 81 / 96 s | tail shrinks |
| prompt_length mean (sanity: same data) | 12,890 | 12,890 | - |
| response_length mean (non-clipping) | 628 | 619 | - |
| val accuracy (step 0 -> 10) | 0.662 -> 0.782 | 0.681 -> 0.764 | parity |

Per-request generation latency (`gen_s`) percentiles - the straggler tail shrinks with EPP:

<img src="latency-percentiles.png" width="75%" alt="arXiv - per-request latency percentiles: Native Verl vs Verl+llm-d-EPP">

Notes: rollout generation is only a fraction of the training-dominated full step (update_actor alone
is ~550 s/step), so the ~31% rollout-generation win nets ~5% on the full step. The vLLM `/metrics`
scrape (prefix-cache hit rate, per-replica KV utilization) was not collected for this run, so those
metrics are omitted here; the rollout-time and latency behaviour is consistent with the large-input
`scotus_xl` classification analog (the long shared paper is co-located and prefilled once instead of
re-prefilled per group sample).

## Reproduce

```bash
# build the data once (no_ref config, CoT), then run each arm
python3 benchmarks/workloads/arxiv/make_arxiv.py --local_dir /tmp/verl/data/arxiv
benchmarks/scripts/run_test.sh --task arxiv --mode <native|epp> --steps 10
```

The EPP arm needs `common/deploy/epp-config-burst.yaml` with `windowDurationMs: 1000` + `balanceBy: tokens` and
the token-balanced EPP binary.
