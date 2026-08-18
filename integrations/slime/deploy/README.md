# Deploying the slime integration

General guide for wiring llm-d routing into any slime training setup.

## Prerequisites

- A Ray cluster with GPU nodes.
- [slime](https://github.com/THUDM/slime) installed in the training environment.
- EPP and Envoy binaries available on the head node (see step 2).

## Steps

### 1. Install the common package

The slime integration is the common router stack and registration shim, activated by training flags. Install on the head node:

```bash
pip install "git+https://github.com/llm-d-incubation/llm-d-rl.git#subdirectory=integrations/common"
```

Or add the source to `PYTHONPATH` without installing:

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl.git
export PYTHONPATH=$(pwd)/llm-d-rl/integrations/common/src:$PYTHONPATH
```

### 2. Get the llm-d routing binaries

The integration launches these as external processes at runtime; none are baked into the slime image.
Obtain them from the published llm-d images or build from source, and put them at the default paths
below. Set the env vars only if the binaries live somewhere else; `llm-d-rl-router` reads them (it
does not search `PATH`). Slime's EPP binary must include `sglanghttp-parser`.

| Env var | Default | Binary |
|---------|---------|--------|
| `LLMD_EPP_BINARY` | `/usr/local/bin/epp` | EPP (endpoint picker) |
| `LLMD_ENVOY_BINARY` | `/usr/local/bin/envoy` | Envoy proxy |
### 3. Place the config files

Copy these starting-point configs to any path readable on the head node:

- [`epp-config-burst.yaml`](../../common/src/llm_d_rl_common/src/llm_d_rl_common/configs/epp-config-burst.yaml) — EPP
  scorer pipeline (burst prefix-cache + load-aware). Render `EPP_PARSER` to
  `sglanghttp-parser`.
- [`envoy-shim.yaml`](../../common/src/llm_d_rl_common/src/llm_d_rl_common/configs/envoy-shim.yaml) — Envoy listener
  (inference through EPP, `/workers*` to the shim)

The EPP config's `file-discovery` plugin `path:` must match the `--endpoints-file` passed to `llm-d-registration-shim` (default `/tmp/epp-endpoints.yaml`).

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

llm-d-registration-shim \
  --engine-type sglang --id-field id \
  --host 127.0.0.1 \
  --port 3001 \
  --endpoints-file /tmp/epp-endpoints.yaml &
```

### 5. Run training with llm-d routing

Add two flags to your existing `train.py` invocation:

```bash
python3 /tmp/slime-src/train.py \
  ... \
  --sglang-router-ip <envoy-host> \
  --sglang-router-port 8081
```

When those flags are set, slime talks to Envoy instead of starting sglang-router. SGLang engines register via `POST /workers` on startup — Envoy routes this to the shim, which writes `/tmp/epp-endpoints.yaml`. EPP watches that file and starts routing inference requests.

## Observability

| File | Component |
|------|-----------|
| `/tmp/epp.log` | EPP |
| `/tmp/envoy.log` | Envoy |
| `/tmp/router.log` | `llm-d-rl-router` itself (startup, readiness, child exits) |
| `/tmp/shim.log` | Registration shim |

Raise EPP verbosity with `LLMD_EPP_VERBOSITY=5` and Envoy's with
`LLMD_ENVOY_LOG_LEVEL=debug` in the router's environment.
