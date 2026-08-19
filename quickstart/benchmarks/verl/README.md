# llm-d-rl-verl-bench

The benchmark and research harness for the verl integration. **A separate package
on purpose**: it depends on `llm-d-rl-verl-integration`, and nothing in the
integration depends on it, so an adopter installing the integration never pulls
any of this.

| Directory | What it is |
|---|---|
| `epp_dev/` | EPP's own job prototyped in Python - placement, migration, which peer holds a reusable prefix. **Staging**: anything that works here should graduate into an EPP plugin. Today: `wave_admission/`. |
| `inproc_sidecar/` | The routing sidecar's job, done in-process. Our engines are Ray actors sharing one pod, so there is no sidecar container to inject `kv_transfer_params`. `pd_replica.py` launches the real `pd-sidecar` per replica; `p2p_replica.py` does the same for P2P and can also skip it and write the parameters directly; `p2p_addressing.py` gives each replica its own loopback IP so N engines in one pod are distinct peers. **Not expected to graduate** - it exists so PD and P2P can be exercised without the production topology. |
| `native_logging/` | verl's own routing plus a reqlog, so a native arm is comparable with an EPP arm. The A/B control. |
| `trace_player/` | Replays a conversation trace - per-turn input/output token counts and inter-turn gaps, no text - to isolate routing cost from model quality. Dataset-agnostic; see `workloads/weka/`. |
| `tools/` | Per-workload tools (searchr1's retriever client). |
| `modes-bench.yaml` | The research modes. Same schema and renderer as the integration's own table; `run_test.sh` reads both. See [MODES.md](MODES.md). |

## Why the split

In production, EPP decides where a request goes and the sidecar injects the KV
transfer parameters. Measuring whether a *different* decision would be better
means making that decision somewhere first, and doing it in Python is far cheaper
than writing an EPP plugin per idea. That is legitimate research - but it is not
the integration, and shipping it as though it were made the integration look four
times bigger than it is.

## Install

Provisioning does this for you (`kuberay/deploy.sh provision --framework verl`),
because this cluster exists to run benchmarks. Set `LLMD_BENCH=0` to skip it.

```bash
pip install "git+https://github.com/llm-d-incubation/llm-d-rl.git#subdirectory=quickstart/benchmarks/verl"
```

Without it, `run_test.sh` still resolves the integration's own modes (`epp`,
`epp-inflight`, `epp-fc`, `epp-sglang`) and reports the research ones as unknown.

---

## Running the benchmarks

The performance-testing harness for the integration: a native-vs-EPP rollout comparison across a
set of RL workloads.

- [`../scripts/`](../scripts/) - shared, framework-agnostic tooling: `run_test.sh` (in this directory - a thin driver that sources a
  workload's `task.env`), `run_on_head.sh` (laptop launcher), the instrumentation (`rl_orchestrate.sh`,
  `vllm_scrape.py`), and `push-epp.sh`. **Start here to run benchmarks** - see
  [`scripts/README.md`](../scripts/README.md) for the full instrumentation runbook.
  
- [`workloads/`](workloads/) - one self-contained folder per workload (`task.env`, data builder,
  and for Search-R1 its tool config + retriever service). Run one with
  `scripts/run_test.sh --task <name>`.

To bring up a cluster and run these on Kubernetes, see [`../../kuberay/`](../../kuberay/README.md).

Mode reference: [MODES.md](MODES.md) for the research modes, and
[the integration's own](../../../integrations/verl/docs/configuration.md) for the rest.
