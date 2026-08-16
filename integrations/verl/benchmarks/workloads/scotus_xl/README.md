# scotus_xl workload

Large-input SCOTUS issue-area classification (LexGLUE SCOTUS): read a full U.S. Supreme Court
opinion and assign exactly one of 13 Spaeth issue-area labels. Single-turn; very large input, short
output. Reward = normalized exact-match (`data_source=searchR1_nq`; `ground_truth.target=[code, name]`,
so the numeric code or the issue-area name scores). Builder: `make_scotus.py`.

## Task and data

- Dataset: LexGLUE SCOTUS (`coastalcph/lex_glue`, config `"scotus"`) - full opinion text from the
  Spaeth Supreme Court Database (~7,800 opinions: 5,000 train / 1,400 val / 1,400 test).
- Labels: the 13 Spaeth issue areas (1 Criminal Procedure, 2 Civil Rights, 3 First Amendment,
  4 Due Process, 5 Privacy, 6 Attorneys, 7 Unions, 8 Economic Activity, 9 Judicial Power,
  10 Federalism, 11 Interstate Relations, 12 Federal Taxation, 13 Miscellaneous).
- Input sizes (measured, prompt cap 24,576 tok): mean 7,740, p50 6,450, p90 16,359, max 24,562,
  min 249. Real opinions vary widely (short per-curiam dismissals to 24k+ full opinions).
- Output sizes (brief CoT, response cap 2,048): mean 603, p50 480, p90 1,077 (non-clipping).

## Test setup

- Cluster: 1 worker node, 8x NVIDIA H200; vLLM TP=1 -> 8 independent replicas (one per GPU).
- Model: Qwen3-4B (FSDP1 actor/ref). GRPO `rollout.n=8`, `train_batch_size=256`
  -> 256 prompts x 8 samples = 2048 requests/step.
- Sequence budget: `max_prompt_length=24576`, `max_response_length=2048`.
- Routing compared:
  - native = verl least-in-flight load balancer (sticky session, request_id -> replica).
  - EPP as the verl endpoint picker = llm-d burst prefix-cache producer with `balanceBy: tokens`
    and `windowDurationMs: 1000` (covers the ~0.86s per-step rollout arrival burst).
- Runs: `scotus_xl_{native,epp}_30s` (30 steps each, 8 replicas). These are the `task.env` defaults
  (brief CoT, `max_response_length=2048`).

## Results summary

<img src="rollout-time-per-step.png" width="75%" alt="SCOTUS - rollout time per step: Native Verl vs Verl+llm-d-EPP">


Using EPP as the verl endpoint picker (vs native least-in-flight), steady-state over steps 2-30:

1. Mean rollout time per step reduced ~31% (88.44s -> 61.08s).
2. Prefix-cache hit rate improved from 12% to 85%.
3. Per-replica KV utilization no longer saturated (native 100% on every replica -> 78% / 63%).
4. Slowest-replica (straggler) generation time reduced ~31% (88.0s -> 60.8s).
5. Validation accuracy converged identically (both reach 0.661) - no accuracy change.

| metric | native | EPP (token-balanced burst) | diff |
|---|---|---|---|
| mean rollout / step (reqlog full-span) | 88.44 s | 61.08 s | -30.9% |
| verl `timing_s/gen` (adds ~3s driver overhead) | 91.79 s | 64.62 s | -29.6% |
| generate_sequences mean / replica | 49.7 s | 36.2 s | -27.2% |
| generate_sequences slowest (straggler) | 88.0 s | 60.8 s | -30.9% |
| straggler ratio (slowest / mean) | 1.77x | 1.68x | -5.1% |
| per-replica peak KV utilization (max / min) | 100% / 100% (saturated) | 78% / 63% | no saturation |
| prefix-cache hit rate (lifetime, 8 replicas) | 12.2% | 84.5% | +72.3 pts |
| val accuracy (step 0 -> 30) | 0.513 -> 0.661 | 0.531 -> 0.661 | parity |

Per-request generation latency (`gen_s`) percentiles - the straggler tail shrinks with EPP:

<img src="latency-percentiles.png" width="75%" alt="SCOTUS - per-request latency percentiles: Native Verl vs Verl+llm-d-EPP">

## Reproduce

```bash
# build the data once (CoT), then run each arm
python3 benchmarks/workloads/scotus_xl/make_scotus.py --local_dir /tmp/verl/data/scotus_xl_cot
benchmarks/scripts/run_test.sh --task scotus_xl --mode <native|epp> --steps 30
```

The EPP arm needs `common/deploy/epp-config-burst.yaml` with `windowDurationMs: 1000` + `balanceBy: tokens` and
the token-balanced EPP binary.
