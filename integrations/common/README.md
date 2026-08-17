# llm-d-rl-common

Framework-agnostic utilities for talking to [llm-d](https://github.com/llm-d/llm-d)'s
**Endpoint Picker (EPP)** from an RL training loop. No dependency on any specific RL
framework (verl, Ray, etc.) - only `grpcio` and `pyyaml`.

Used today by [`integrations/verl`](../verl/README.md),
[`integrations/vime`](../vime/README.md), and
[`integrations/slime`](../slime/README.md); intended to be reused by any
other framework integration that needs to talk to EPP.

## Contents

### src

[`src/llm_d_rl_common/`](src/llm_d_rl_common/) is the Python package of
shared components used by every integration.

- `epp_grpc_client.py` - minimal hand-rolled EPP ext-proc gRPC client (`EPPGrpcClient`).
  `route()` is the entry point: it picks between EPP's fire-and-forget and
  tracked-completion protocols and returns a `RoutingResult` with a `.complete()`
  hook that is a no-op unless completion tracking was requested.
- `reqlog.py` - shared per-request JSONL logging helpers.
- `endpoints.py` - EPP file-discovery endpoints YAML writer.
- `router_stack.py` - `RouterStack`: owns the EPP / Envoy argv, binary-path resolution
  (`LLMD_EPP_BINARY`, falling back to `VERL_EPP_BINARY`) and readiness waiting. Stdlib
  only, no ray. verl's `LlmdActor` wraps it in a head-pinned Ray actor.
- `cli.py` - the `llm-d-rl-router` console script: runs the same stack in the
  foreground, for frameworks that start EPP from a pod lifecycle hook instead of from
  inside the training process.
- `registration_shim.py` - `llm-d-registration-shim`: a FastAPI server that
  receives engine registration requests and writes the EPP endpoints file
  (`--engine-type vllm` or `--engine-type sglang --id-field id`).

### configs

[`configs/`](configs/) holds shared Envoy and EPP config. Each integration's `deploy.sh`
points at these files.

- [`configs/envoy.yaml`](configs/envoy.yaml) — Base Envoy config: inference on
  `:8081` (no registration API).
- [`configs/envoy-shim.yaml`](configs/envoy-shim.yaml) — Same inference path as
  the base Envoy config, plus `/workers*` forwarded to
  `llm-d-registration-shim`.
- [`configs/epp-config-burst.yaml`](configs/epp-config-burst.yaml) - burst
  prefix-cache profile. Parser defaults to `vllmhttp-parser`; slime sets
  `EPP_PARSER=sglanghttp-parser` in its `deploy.env`.

## llm-d router stack

Used by the vime and slime integrations.

The llm-d stack is the rollout endpoint for the RL training frameworks.
The stack includes Envoy (the HTTP proxy on `:8081`) and the llm-d
Endpoint Picker (EPP) on `:9002`, which provides the routing
intelligence. The framework sends generation to Envoy; Envoy asks EPP
for a replica, then forwards the request there. The trainer only ever
talks to that one address.

| Component | Port | Role |
|---|---|---|
| EPP | 9002 (gRPC) | Scores replicas and picks the target engine |
| Envoy | 8081 (HTTP) | Forwards inference through EPP (`envoy.yaml`), or also `/workers*` to the shim (`envoy-shim.yaml`) |

vime and slime start this stack with the `llm-d-rl-router` command:

```
llm-d-rl-router --epp-config ... --envoy-config ...
```

## Registration shim

Separate process (`llm-d-registration-shim`). Used by vime and slime so engines can
register over HTTP (verl writes the endpoints file from the trainer instead).

`llm-d-registration-shim` listens on localhost:3001 and writes `/tmp/epp-endpoints.yaml`.
Start it with two flags that select the engine label and the registration
protocol:

- `--engine-type` — written into each endpoints-file entry as
  `llm-d.ai/engine-type` so EPP scrapes the right metrics (`vllm` or `sglang`).
- `--id-field` — which field `POST /workers` returns as the handle, and what
  `DELETE /workers/{ref}` looks up. Set this to match the integration's
  router: vime uses vllm-router (`url`, the default); slime uses
  sglang-router (`id`).

Example:
```
llm-d-registration-shim --engine-type vllm
llm-d-registration-shim --engine-type sglang --id-field id
```

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/workers` | Register. Body `{"url": "http://host:port"}`. Response is `{id_field: key}`. |
| `GET` | `/workers` | List registered engines. |
| `DELETE` | `/workers/{ref}` | Deregister. `{ref}` is the URL (vime) or the id (slime). |

On every change the shim rewrites `/tmp/epp-endpoints.yaml` (or `--endpoints-file`). EPP's `file-discovery` plugin watches that file.
