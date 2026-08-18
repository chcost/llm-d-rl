# verl + llm-d integration

Route [verl](https://github.com/volcengine/verl) RL training rollouts through
[llm-d](https://github.com/llm-d/llm-d)'s **Endpoint Picker (EPP)**. During each training step
verl generates completions from a pool of vLLM replicas; this integration replaces verl's default
round-robin replica selection with EPP, which steers each request to the replica most likely to
already have its KV cache warm, so a prompt prefix shared across a GRPO group is not re-prefilled on
every replica.

No verl source changes are required - the whole integration is a single Hydra override.

![verl generate call flow](docs/diagrams/verl-generate-call-flow.png)

## Get started

- **Run it (recommended path):** [`quickstart/kuberay/`](../../quickstart/kuberay/README.md) - a complete,
  runnable end-to-end example on Kubernetes (KubeRay): cluster manifest, configs, and scripts for
  deploy, train, and benchmark.
- **Run it on any Ray cluster:** [`docs/deploying.md`](docs/deploying.md) - the general deployment guide
  (install steps, the Hydra override reference, and env-var reference), with the KubeRay commands
  as the concrete example of each step.
- **Understand it:** [`docs/architecture.md`](docs/architecture.md) - how the integration works,
  the two routing modes, the (tiny) mandatory core vs the optional utilities, and PD disaggregation.

## Two routing modes

- **EPP as the endpoint picker** - verl/Ray call EPP over gRPC to pick a replica, then dispatch to
  it themselves. Fewer moving parts, lower latency. Start here if unsure.
- **llm-d serving** - verl speaks HTTP to a single Envoy endpoint; Envoy + EPP pick a replica and
  forward to it. Closer to a production llm-d serving deployment.

Both need no verl patches, and both support prefill/decode (PD) disaggregation. See
[`docs/architecture.md`](docs/architecture.md).

## Benchmarks

[`quickstart/benchmarks/`](../../quickstart/benchmarks/) is the performance-testing harness for the integration: it benchmarks
rollout routing (native vs EPP) across multiple RL workloads and collects the results.

- [`benchmarks/workloads/`](../../quickstart/benchmarks/verl/workloads/) - one folder per workload, each self-contained: a
  `task.env` with its verl overrides, its data builder, and (for Search-R1) its tool config and
  retriever service.
- [`benchmarks/scripts/`](benchmarks/scripts/) - the run harness; `run_test.sh --task <name>` runs a
  chosen workload in `native` or `epp` mode.
- Result summaries live in each workload's README and in [`benchmarks/README.md`](../../quickstart/benchmarks/verl/README.md)
  (raw run data is kept out of the repo).

---

This package is the `verl` integration of
[llm-d-rl](https://github.com/llm-d-incubation/llm-d-rl), where it lives at
`integrations/verl/`. Its Python distribution is `llm-d-rl-verl-integration` (package
`llm_d_rl_verl_integration`).
