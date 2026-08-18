# Deploying the verl integration on any Ray cluster

This is the general guide for wiring the integration into any Ray cluster, plus the full Hydra
override and environment-variable reference. For a ready-to-run, end-to-end example, use the
**[KubeRay walkthrough](../../../quickstart/kuberay/README.md)** - it is the concrete instantiation of every step below.

For how the integration works (the two modes, the mandatory core, PD), see
[`../docs/architecture.md`](../docs/architecture.md).

## Prerequisites

- A Ray cluster with at least one GPU worker node.
- verl on a compatible vLLM version (the [official verl images](https://hub.docker.com/r/verlai/verl)
  provide the environment).

## Steps

### 1. Provide verl

The official `verlai/verl` images are **environment** images: they ship the vLLM/CUDA/torch stack
but not verl itself, so verl is installed at runtime. The version is pinned by `VERL_COMMIT` in
[`kuberay/deploy.env`](../../../quickstart/kuberay/deploy.env) - the single place to change it:

```bash
git clone https://github.com/volcengine/verl.git "$VERL_SRC"
cd "$VERL_SRC" && git checkout "$VERL_COMMIT"
pip install --no-deps -e .
```

Do this on **every** node (head and all workers). On KubeRay this runs from the `postStart` hook in
[`kuberay/ray-cluster.yaml.tmpl`](../../../quickstart/kuberay/ray-cluster.head-workers.yaml.tmpl), which `deploy.sh` renders with the
value from `deploy.env`.

To move to a newer verl, edit `VERL_COMMIT` and recreate the pods (`postStart` clones fresh on every
start, so nothing is cached). Prefer a full SHA over a branch name: the head and the workers clone
independently, so a branch can install two different commits in one cluster. Each pod's `postStart`
echoes `verl at <sha> (requested <ref>)` - compare them across pods before trusting a branch run.

#### Nightly-vLLM environment image

[`deploy/Dockerfile.verl.vllm-p2p`](../../../quickstart/images/Dockerfile.verl.vllm-p2p) builds an alternative
environment image for testing verl against a nightly/dev vLLM wheel instead of a pinned stable
release. Built from the verl 0.24 recipe (torch, CUDA 13.0.2, transformers, flash-attn, etc.) but
pins vLLM to a dev wheel off `wheels.vllm.ai` at a specific commit, to pick up two unmerged
upstream PR fixes verl's weight-sync flow needs (see the Dockerfile's own comments for which PRs
and why). Also bakes in NIXL plus the vLLM patches PD and P2P KV-cache sharing both need (formerly
a separate `Dockerfile.pd` layered on top of this one - folded in directly so every mode
(native/EPP/P2P/PD) runs on the same image).

```bash
docker build -f deploy/Dockerfile.verl.vllm-p2p -t <your-registry>/verl:vllm024.devN .
docker push <your-registry>/verl:vllm024.devN
```
Then point `IMG_VERL` at it in [`kuberay/deploy.env`](../../../quickstart/kuberay/deploy.env).

Public image, published for the org: `ghcr.io/llm-d-incubation/llm-d-rl/verl:vllm-p2p`. This is
the tag `deploy.env` points at by default.

Bakes in fixes for problems hit pulling a nightly vLLM wheel on top of the stock verl 0.24 recipe:

| Problem | Fix | In image? |
|---|---|---|
| flash-attn's `.so` built against the wrong torch ABI (`undefined symbol: ...materialize_cow_storage...`) - the vLLM wheel install has no `--no-deps` and silently bumps torch 2.11.0 -> 2.13.0, but apex/TransformerEngine/flash-attn/DeepEP were building *before* that bump | Reordered: vLLM installs first, compiled extensions after | Yes |
| `transformers==5.3.0` too old for this vLLM (`>=5.5.3`) and megatron-bridge (`>=5.8.1,<5.9.0`) | `ARG TRANSFORMERS_VERSION=5.8.1` | Yes |
| `flash_attn`'s `cute` submodule crashes megatron-core's attention import with a bare `AttributeError` (`cutlass.cute.core has no attribute 'ThrMma'`) at `nvidia-cutlass-dsl==4.6.0` | Pin `nvidia-cutlass-dsl==4.5.3` - not a "compatible" version, just one where the failure is a clean `ModuleNotFoundError` that megatron's own `except ImportError:` guard actually catches | Yes |

**Important - this image does not make verl or the integration package "nightly-compatible" by
itself.** Both are still fetched fresh by `postStart` at pod start (step 1 above, and step 2 below)
regardless of which environment image you use - they are never baked into any image, on purpose,
so code changes don't require an image rebuild. Two more compatibility problems surfaced only when
testing this image against real training runs, and *neither* can live in this Dockerfile:

| Problem | Where the fix lives | Why not here |
|---|---|---|
| `LlmdBaseAgentLoopManager` subclasses verl's pre-TransferQueue `AgentLoopManager`; verl's v1 trainer defaults to the newer `AgentLoopManagerTQ`, so any llm-d mode crashes with `AttributeError: 'TensorDict' object has no attribute 'non_tensor_batch'` | `src/llm_d_rl_verl_integration/base_agent_loop_manager.py` (this repo) - **fixed in the working tree, not yet committed/merged** | Installed by `postStart`'s step-2 `pip install git+...llm-d-rl.git`, not this image - the fix needs to reach whatever ref that clones, not a rebuild |
| vLLM renamed `FusedMoE` to `FusedMoEFactory`; verl's `vllm_fp8_utils.py` imports the old name unconditionally | verl's own source - patched by a `postStart` step right after the `git checkout` in step 1 (added to `ray-cluster.yaml.tmpl`, both head and worker) | verl is third-party (`volcengine/verl`), cloned fresh every pod start - we can't bake a fix for it into an image at all |

### 2. Install the integration package

Install on **every** node (head and all workers) - Ray does not propagate a pip install across
nodes:

```bash
pip install "git+https://github.com/llm-d-incubation/llm-d-rl.git#subdirectory=integrations/verl"
```

This pulls in `llm-d-rl-common` (the framework-agnostic EPP client and utilities in
[`integrations/common`](../../common/README.md)) automatically via its declared dependency.

Or add the source to `PYTHONPATH` without installing:

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl.git
export PYTHONPATH=$(pwd)/llm-d-rl/integrations/verl/src:$(pwd)/llm-d-rl/integrations/common/src:$PYTHONPATH
```

### 3. Get the EPP, Envoy, and sidecar binaries

The integration launches these as external processes at runtime; none are baked into the verl image,
so iterating on them never triggers a verl rebuild. Obtain them from the published llm-d images or
build from source, then point the integration at them via env vars (set before Ray starts, or in the
`ray.init` runtime env). On the head, `LlmdActor` launches EPP (and, in serving mode, Envoy); on each
worker, decode replicas launch the sidecar (PD only).

| Env var | Default | Binary | Node | Required for |
|---------|---------|--------|------|--------------|
| `VERL_EPP_BINARY` | `/usr/local/bin/epp` | EPP (endpoint picker) | head | every run |
| `VERL_ENVOY_BINARY` | `/usr/local/bin/envoy` | Envoy proxy | head | llm-d serving mode only |
| `VERL_SIDECAR_BINARY` | `/opt/llm-d-bins/pd-sidecar` | llm-d routing sidecar | workers (decode) | PD disaggregation only |

### 4. Place the config files

The EPP configs ship inside `llm-d-rl-common`, so a pip install is enough - there
is no file to copy out of this repository. Each one is composed from a chassis
plus a routing profile plus optional modifiers, listed in
[`configs/epp/variants.yaml`](../../common/src/llm_d_rl_common/configs/epp/variants.yaml).
Render whichever you want to a path the head node can read:

```bash
llm-d-rl-epp-config list                              # every variant and its layers
llm-d-rl-epp-config render epp-config.yaml -o /etc/llmd-configs/epp-config.yaml
llm-d-rl-epp-config render epp-config-pd.yaml -o /etc/llmd-configs/epp-config-pd.yaml
```

Envoy's config (llm-d serving mode only) is shipped as data rather than composed:

```bash
python3 -c "from llm_d_rl_common import configs; print(configs.path('envoy.yaml'))"
```

To change scoring, edit `configs/epp/profiles/` or `configs/epp/modifiers/` and
re-render - never a rendered file, which carries a generated header saying so. The
EPP build that can load these is pinned alongside them in
[`versions.env`](../../common/src/llm_d_rl_common/configs/versions.env), because a
config can require a plugin or a stability flag only some builds have.

The EPP config's `file-discovery` plugin `path:` must match the
`epp_endpoints_file` override - the router writes the replica list there and EPP
reads it (default `/tmp/epp-endpoints.yaml`).

### 5. Add the Hydra overrides and run

Running the integration is just a few Hydra overrides on your existing verl training command - see
the reference below. The KubeRay walkthrough ([`kuberay/README.md`](../../../quickstart/kuberay/README.md)) shows the
full commands for each mode.

## Hydra override reference

The complete, per-mode reference lives in
[`docs/configuration.md`](../docs/configuration.md), generated from
`src/llm_d_rl_verl_integration/modes.yaml` - the same data the benchmark driver
reads, so the two cannot drift. Print it for any mode without a checkout:

```bash
llm-d-rl-verl-overrides --list          # every mode and what it does
llm-d-rl-verl-overrides epp             # the overrides for one mode
llm-d-rl-verl-overrides --markdown      # regenerate docs/configuration.md
```

`EPP_CONFIG` selects a different EPP variant on any EPP-bearing mode; the shipped
variants are listed in
[`configs/epp/variants.yaml`](../../common/src/llm_d_rl_common/configs/epp/variants.yaml)
and rendered with `llm-d-rl-epp-config render <variant>`.

## Observability

### Per-request JSONL logging (reqlog)

All routing clients (`llmd_epp`, `llmd_serving`, `native_logging`) write a JSONL timing record per
request when `VERL_REQLOG_DIR` is set: `$VERL_REQLOG_DIR/reqlog-<pid>.jsonl`, one file per worker
process, line-buffered. It is a no-op when `VERL_REQLOG_DIR` is unset.

| Field | Description | Modes |
|-------|-------------|-------|
| `ts` | Wall-clock timestamp (Unix seconds) | all |
| `request_id` | verl request ID | all |
| `turn` | 0-based turn index within a multi-turn trajectory | all |
| `endpoint` | Backend pod that served the request (`host:port`) | all |
| `prompt_hash` | BLAKE2b-8 hex digest of the input token IDs | all |
| `prompt_tokens` | Number of input tokens | all |
| `output_tokens` | Number of generated tokens | all |
| `pick_s` | Time spent on the routing decision (EPP gRPC call or load-balancer acquire) | `llmd_epp`, `native_logging` |
| `gen_s` | Generation time — actor call only for `llmd_epp`/`native_logging`; full round-trip (routing + inference) for `llmd_serving` | all |

### Debug logging

All components default to quiet output. Increase verbosity with these env vars:

| Env var | Component | Debug value |
|---------|-----------|-------------|
| `VERL_VLLM_LOG_LEVEL` | vLLM inside replicas | `DEBUG` |
| `VERL_SIDECAR_LOG_LEVEL` | llm-d routing sidecar | `5` |
| `VERL_EPP_VERBOSITY` | EPP subprocess | `5` |
| `VERL_ENVOY_LOG_LEVEL` | Envoy proxy | `debug` |

Log files on the pod: `/tmp/epp.log`, `/tmp/envoy.log`, `/tmp/sidecar-decode-{rank}.log`.

Note: Ray actors are spawned as new processes and do not inherit the launching shell's environment.
Set these in the pod spec `env:` section (KubeRay) or in your `ray.init` runtime env.
