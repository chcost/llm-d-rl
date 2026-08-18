# Benchmarks

The performance-testing harness for the integration: a native-vs-EPP rollout comparison across a
set of RL workloads.

- [`scripts/`](scripts/) - the run harness: `run_test.sh` (thin driver, sources a workload's
  `task.env`), `run_on_head.sh` (laptop launcher), the instrumentation (`rl_orchestrate.sh`,
  `vllm_scrape.py`), and `utils/push-epp.sh`. **Start here to run benchmarks** — see
  [`scripts/README.md`](scripts/README.md) for the full instrumentation runbook.
  
- [`workloads/`](workloads/) - one self-contained folder per workload (`task.env`, data builder,
  and for Search-R1 its tool config + retriever service). Run one with
  `scripts/run_test.sh --task <name>`.

To bring up a cluster and run these on Kubernetes, see [`../deploy/kuberay/`](../deploy/kuberay/README.md).

> Suite overview and cross-workload results summary: to be populated later.
