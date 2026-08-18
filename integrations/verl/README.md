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

Both need no verl patches. See [`docs/architecture.md`](docs/architecture.md), and
[`docs/configuration.md`](docs/configuration.md) for the exact overrides each mode
needs (generated from the package's own mode table, so it cannot drift).

### PD and P2P KV-cache sharing are experimental, and not in this package

In a production llm-d deployment **EPP decides** which peer holds a reusable
prefix and **the routing sidecar injects** the resulting `kv_transfer_params`.
This integration does neither: it tells EPP which endpoints are prefill and which
are decode (role-tagged endpoint discovery) and dispatches where EPP says.

What this repo has beyond that is a research harness, not an integration: it
prototypes EPP's placement decisions in Python and stands in for the sidecar
in-process, because our benchmark topology runs many engines as Ray actors inside
one pod and so has no sidecar container. That code lives in
[`quickstart/benchmarks/verl`](../../quickstart/benchmarks/verl/) under
`epp_dev/` and `inproc_sidecar/`, with the modes that use it in
[`MODES.md`](../../quickstart/benchmarks/verl/MODES.md).

Treat PD and P2P as experimental here. If you want them in production, get them
from EPP and the sidecar; if you want to measure or extend them, read the
harness.

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
