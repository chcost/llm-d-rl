# llm-d-rl

Reinforcement-learning rollout infrastructure for the
[llm-d](https://github.com/llm-d) inference serving stack.

RL post-training frameworks (veRL, OpenRLHF, SkyRL, NeMo-RL, Slime) each need to
orchestrate a pool of inference engines during training: route generation
requests efficiently, synchronize updated weights into the engines, and manage
their lifecycle. This repo provides those pieces for llm-d so frameworks do not
have to reimplement them per backend.

The repo has two independent components: framework integrations that plug llm-d
into an existing training loop, and a standalone control plane that orchestrates
the rollout itself. They can be adopted separately.

```
llm-d-rl/
├── integrations/                 # one directory per framework
│   ├── verl/
│   ├── vime/
│   ├── slime/
│   └── common/                   # library used by the three above
└── experimental/
    └── rl-controller/            # control plane (Go + Python); independent of integrations/
```

## integrations/

This section describes how to integrate llm-d into various RL post-training
frameworks. Supported today: [verl](integrations/verl/) (the most complete - EPP
routing, llm-d serving, PD and P2P KV-cache sharing),
[vime](integrations/vime/) (llm-d routing, vLLM engines), and
[slime](integrations/slime/) (llm-d routing, SGLang engines).
Shared code and configs live in [`common/`](integrations/common/).

The common idea is to replace the framework's default round-robin replica
selection with llm-d's **Endpoint Picker Plugin (EPP)**, which scores each
candidate vLLM replica on prefix-cache hit rate, queue depth, and KV utilization,
and steers each request to the replica most likely to already have a warm cache.
For large-group RL workloads (GRPO, PPO with large rollout groups), where many
samples share a prompt prefix, this is a meaningful throughput win over spreading
requests evenly.

There are two integration modes, named by mechanism (EPP is the Endpoint
*Picker* - it scores and selects a replica, it does not proxy):

- **EPP as the endpoint picker** - the framework calls EPP directly over gRPC to
  score replicas, gets back the chosen replica, and dispatches to it itself. Fewer
  moving parts and lower latency; the place to start.
- **llm-d serving** - the framework sends all generation to a single Envoy
  endpoint; Envoy calls EPP to pick the best replica and forwards the request. The
  framework only ever speaks HTTP to one address, closest to a production llm-d
  serving deployment.

Both modes require no framework source changes - they are wired in entirely
through configuration - and both support prefill/decode (PD) disaggregation.

See [`integrations/verl/README.md`](integrations/verl/README.md) for the verl
setup, the config overrides for each mode, PD disaggregation, observability, and
a complete end-to-end KubeRay example.

## experimental/rl-controller

A framework-agnostic **control plane for orchestrating RL tasks** - weight sync
and rollout - exposed over plain HTTP. It handles the primitives a training loop
needs from its inference engine pool: dispatching generation, synchronizing
updated weights into the engines, and managing engine lifecycle
(pause/resume/sleep/wake). The core is a standalone Go binary (the "rollout
controller") with **no Kubernetes dependency** - it talks HTTP to vLLM engines
and runs on Slurm, bare metal, Docker, or Kubernetes. A companion Python package
provides the NCCL weight-sync trainer side.

Control plane (HTTP) and data plane (NCCL/NIXL) stay separate: the controller
orchestrates the lifecycle - pause engines, trigger a weight update, reset caches,
resume - but never proxies weight tensors itself. Generation can be dispatched
directly to a ready engine or forwarded through llm-d's inference router for
prefix-cache-aware, session-affinity routing.

See [`experimental/rl-controller/README.md`](experimental/rl-controller/README.md)
for the HTTP API, CLI flags, quick start, and deployment manifests.

## How the two relate

Both serve the same goal - efficient inference during RL post-training - at
different layers. The framework integrations are drop-ins that add cache-aware
replica selection to an existing training loop; the rollout controller owns the
full rollout lifecycle (weights, sleep/wake, generation) as a standalone service.

Note the two components each carry a similarly named but distinct Python package:
`llm-d-rl-verl-integration` under `integrations/verl` is the EPP routing
integration, while `llmd-verl` under the rollout controller is the NCCL
weight-sync trainer. They are not the same package.

## Status

This is an experimentation / incubation project. The rollout controller runs
end-to-end with simulated lifecycle operations and has manifests for testing with
real vLLM and NCCL weight sync; the verl integration runs on real Ray/KubeRay
clusters. Interfaces may change.

## License

Apache License 2.0
