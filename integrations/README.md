# integrations/ - the delivery

Everything here is something you **install or mount**. If you already have Ray
and an RL framework running, this is all you need; nothing in
[`quickstart/`](../quickstart/) is required.

```
integrations/
  common/    llm-d-rl-common          EPP client, endpoints writer, router
                                      launcher (in-process actor + CLI), reqlog,
                                      and the shipped EPP / Envoy configs
  verl/      llm-d-rl-verl-integration    the verl adapter + its mode contract
             (the benchmark and research harness is a separate package, in
              quickstart/benchmarks/verl - nothing here depends on it)
  vime/      (no package - a flag plus the shared shim is the whole integration)
  slime/     (same)
```

## What the integration actually is

Replace your framework's round-robin replica selection with llm-d's **Endpoint
Picker (EPP)**, which scores each candidate engine on prefix-cache hit rate, queue
depth and KV utilisation, and steers each request to the replica most likely to
already have a warm cache. For large-group RL rollouts (GRPO, PPO with big groups),
where many samples share a prompt prefix, that is a real throughput win over
spreading requests evenly.

No framework source changes: it is configuration plus, for verl, one subclass.

Two ways to route, named by mechanism (EPP is a *picker* - it scores and selects,
it does not proxy):

- **EPP as the endpoint picker** - the framework asks EPP over gRPC which replica
  to use, then dispatches there itself. Fewer moving parts, lower latency. Start here.
- **llm-d serving** - the framework sends all generation to one Envoy endpoint;
  Envoy asks EPP and forwards. Closest to a production llm-d deployment.

Both tell EPP which endpoints are prefill and which are decode, so a PD
deployment routes correctly. Beyond that, PD and P2P KV-cache sharing are
**experimental** and their code is a research harness rather than part of these
packages - see [`quickstart/benchmarks/verl`](../quickstart/benchmarks/verl/).

## Adopting it

1. **Install.** Two packages, and neither needs this repo checked out:

   ```bash
   pip install "git+https://github.com/llm-d-incubation/llm-d-rl.git#subdirectory=integrations/common"
   pip install "git+https://github.com/llm-d-incubation/llm-d-rl.git#subdirectory=integrations/verl"
   ```

   Extras: `[shim]` adds the registration shim's HTTP server (vime, slime need it;
   verl does not), `[ray]` adds Ray for the in-process router actor.

2. **Get EPP and Envoy.** They are separate binaries, not Python. Pull them from
   their published images - the versions these configs are tested against are in
   [`common/src/llm_d_rl_common/configs/versions.env`](common/src/llm_d_rl_common/configs/versions.env):

   ```bash
   crane export --platform linux/amd64 "$LLMD_EPP_IMAGE" - | tar -xO app/epp > /usr/local/bin/epp
   crane export --platform linux/amd64 "$LLMD_ENVOY_IMAGE" - | tar -xO usr/local/bin/envoy > /usr/local/bin/envoy
   chmod +x /usr/local/bin/{epp,envoy}
   ```

3. **Render a routing config.** They ship inside the package, composed from a
   chassis plus a profile plus modifiers:

   ```bash
   llm-d-rl-epp-config list
   llm-d-rl-epp-config render epp-config.yaml -o /etc/llmd-configs/epp-config.yaml
   ```

4. **Start the router.** Either in-process (verl does this itself via a Ray actor)
   or as a plain process alongside your job:

   ```bash
   llm-d-rl-router --epp-config /etc/llmd-configs/epp-config.yaml
   ```

5. **Turn it on.** For verl, a set of Hydra overrides printed by the package
   itself, so it cannot drift from what the code expects:

   ```bash
   llm-d-rl-verl-overrides --list
   llm-d-rl-verl-overrides epp
   ```

## Per framework

| | Read |
|---|---|
| verl | [README](verl/README.md), [adopting](verl/docs/deploying.md), [overrides](verl/docs/configuration.md), [architecture](verl/docs/architecture.md) |
| vime | [README](vime/README.md), [adopting](vime/docs/deploying.md) |
| slime | [README](slime/README.md), [adopting](slime/docs/deploying.md) |
| shared | [common/README.md](common/README.md) |

## Where the line is

`integrations/` holds packages and the configs they ship. Cluster manifests,
provisioning scripts, benchmark workloads, drivers and Dockerfiles live in
[`quickstart/`](../quickstart/) - one worked example, replaceable by your own.

The test we hold that to: **delete `quickstart/` and this page still gets you
running.** If it does not, something in the delivery is missing.
