# Deploying the vime integration

General guide for wiring llm-d routing into any vime training setup.

## Prerequisites

- A Ray cluster with GPU nodes.
- [vime](https://github.com/vllm-project/vime) installed in the training environment.
- EPP and Envoy binaries available on the head node (see step 2).

## Steps

### 1. Install the common package

The vime integration is the common router stack plus vime's train flags. Install on the head node:

```bash
pip install "git+https://github.com/llm-d-incubation/llm-d-rl.git#subdirectory=integrations/common"
```

Or add the source to `PYTHONPATH` without installing:

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl.git
export PYTHONPATH=$(pwd)/llm-d-rl/integrations/common/src:$PYTHONPATH
```

### 2. Get the llm-d routing binaries

EPP and Envoy are launched as external processes at runtime; they are not baked into the vime image. Obtain them from the published llm-d images or build from source and place them somewhere on `PATH` (the KubeRay setup extracts them to `/opt/llm-d-bins/` via an initContainer).

### 3. Place the config files

Copy these starting-point configs to any path readable on the head node:

- [`epp-config-burst.yaml`](../../common/configs/epp-config-burst.yaml) - EPP
  scorer pipeline (burst prefix-cache + load-aware). Parser defaults to
  `vllmhttp-parser`.
- [`envoy-shim.yaml`](../../common/configs/envoy-shim.yaml) - Envoy listener
  (inference through EPP, `/workers*` to the shim)

The EPP config's `file-discovery` plugin `path:` must match the `--endpoints-file` passed to `llm-d-router-shim` (default `/tmp/epp-endpoints.yaml`).

### 4. Start llm-d routing

On the head node:

```bash
# EPP + Envoy, via the shared launcher from llm-d-rl-common. It owns the argv for
# both (including --allow-experimental-plugins, which the config's
# burst-prefix-cache-producer requires), waits for each to accept connections, and
# exits if either dies. Point it at the binaries with LLMD_EPP_BINARY /
# LLMD_ENVOY_BINARY if they are not on PATH.
llm-d-rl-router \
  --epp-config /path/to/epp-config.yaml \
  --envoy-config /path/to/envoy.yaml &

# Registration shim (internal only - Envoy proxies /workers* to it)
llm-d-router-shim \
  --engine-type vllm \
  --host 127.0.0.1 \
  --port 3001 \
  --endpoints-file /tmp/epp-endpoints.yaml &
```

### 5. Run training with llm-d routing

Add two flags to your existing `train.py` invocation:

```bash
python3 /tmp/vime/train.py \
  ... \
  --vllm-router-ip <envoy-host> \
  --vllm-router-port 8081
```

When `--vllm-router-ip` is set, vime skips its built-in router entirely ([`rollout.py:1035`](https://github.com/vllm-project/vime/blob/main/vime/ray/rollout.py#L1035)). vLLM engines register themselves via `POST /workers` on startup - Envoy routes this to the shim, which writes `/tmp/epp-endpoints.yaml`. EPP watches that file and starts routing inference requests.

For the full `train.py` command with all training hyperparameters, see [`kuberay/run-qwen3-4B.sh`](kuberay/run-qwen3-4B.sh).

## Observability

| File | Component |
|------|-----------|
| `/tmp/epp.log` | EPP |
| `/tmp/envoy.log` | Envoy |
| `/tmp/router.log` | `llm-d-rl-router` itself (startup, readiness, child exits) |
| `/tmp/shim.log` | Registration shim |

Raise EPP verbosity with `LLMD_EPP_VERBOSITY=5` and Envoy's with
`LLMD_ENVOY_LOG_LEVEL=debug` in the router's environment.
