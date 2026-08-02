# Running slime with EPP on KubeRay

End-to-end guide for running slime GRPO training with EPP-based prefix-cache-aware rollout routing on Kubernetes. Uses Qwen3-4B on a single 8-GPU node as the example.

## Architecture

```
slime → envoy:8081  (single address: --sglang-router-ip/port points here)
             │
    ┌────────┴────────────────────────────┐
    │                                     │
/generate, /health*, ...           /workers*, /add_worker, ...
ext_proc → EPP:9002                registration shim:3001 (internal)
→ chosen sglang engine             writes /tmp/epp-endpoints.yaml
                                   EPP reloads via watchFile
```

**Zero slime code changes.** Slime thinks Envoy is the sglang_router. The only difference from a normal slime run is `--sglang-router-ip $MY_POD_IP --sglang-router-port 8081`.

What each component does:

| Component | Port | Where it runs | Role |
|---|---|---|---|
| EPP | 9002 (gRPC) | head pod | Prefix-cache-aware endpoint picker |
| Envoy | 8081 (HTTP) | head pod | Routes `/generate` via EPP; routes `/workers*` to shim |
| Registration shim | 3001 (HTTP, internal) | head pod | Manages engine registry, writes endpoints YAML |
| sglang engines | dynamic | worker pod | Inference; self-register to Envoy on startup |
| Megatron actors | — | worker pod | Training |

## Prerequisites

- Kubernetes cluster with GPU nodes
- KubeRay operator installed (see [setting-kuberay.md](setting-kuberay.md))
- `envsubst` and `kubectl` on your PATH

## Step 1 — Set namespace and images

Export your namespace (required; not stored in a file):
```bash
export NAMESPACE=<your-namespace>
```

Images are defined in `deploy.env` — edit tags there rather than in the manifest. Defaults:

| Variable | Image | Notes |
|---|---|---|
| `IMG_SLIME` | `slimerl/slime:latest` | Pin to a specific tag in production |
| `IMG_CRANE` | `gcr.io/go-containerregistry/crane@sha256:...` | Pinned by digest; never rebuild |
| `IMG_EPP` | `ghcr.io/llm-d/llm-d-router-endpoint-picker-dev:...` | Bump to change EPP version |
| `IMG_ENVOY` | `docker.io/envoyproxy/envoy:distroless-v1.33.2` | Bump to change Envoy version |

EPP and Envoy are not in the slime image. The `fetch-binaries` init container on the head pod extracts them from their separate public images at pod start using crane. Update a binary without rebuilding by bumping its tag in `deploy.env` and recreating the pod.

Adjust the manifest for your cluster:
- **GPU count** — the worker `resources` block defaults to 8 GPUs; edit to match your node.
- **Node placement** — the worker `nodeAffinity` has a placeholder `excluded-node` in the `NotIn` list; replace with any known-faulty hostnames in your cluster.

## Step 2 — Deploy

`deploy.sh apply` builds the `llmd-epp-configs` ConfigMap from the standalone config files (`../epp-config.yaml`, `../envoy.yaml`) and applies the rendered cluster manifest:

```bash
bash deploy/kuberay/deploy.sh apply
```

Do **not** apply `configmap.yaml` directly — it has no `data:` block; `deploy.sh` builds it from the source files.

Useful sub-commands:
```bash
bash deploy/kuberay/deploy.sh render     # print rendered manifest (no kubectl)
bash deploy/kuberay/deploy.sh configmap  # rebuild ConfigMap only
bash deploy/kuberay/deploy.sh delete     # tear down the cluster
```

Wait for the pod to reach `Running` and finish its `postStart` setup:
```bash
kubectl get pods -n $NAMESPACE -w
```

The `postStart` hook clones slime + Megatron-LM, starts EPP/Envoy/shim, and writes `/tmp/slime_ready.txt` when done. Check progress:
```bash
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/setup_log.txt
```

## Step 3 — Run training

Exec into the pod:
```bash
HEAD=$(kubectl get pod -n "$NAMESPACE" -l ray.io/node-type=head -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep slime | head -1)
kubectl exec -it -n "$NAMESPACE" "$HEAD" -- bash
```

Run the training script. It downloads the model, converts weights to Megatron format, and submits the Ray job:

```bash
bash /etc/llmd-configs/run-qwen3-4B.sh --mode llm-d --steps 6          # llm-d (default)
bash /etc/llmd-configs/run-qwen3-4B.sh --mode native  --steps 6       # slime's built-in sglang-router, no EPP
```

`--mode llm-d` points `--sglang-router-ip`/`--sglang-router-port` at Envoy (EPP picks the engine).
`--mode native` omits those args; slime manages its own sglang-router for a baseline comparison.

**Comparison test** (run sequentially in the pod after model/data are already downloaded):
```bash
bash /etc/llmd-configs/run-qwen3-4B.sh --mode llm-d  --steps 6
bash /etc/llmd-configs/run-qwen3-4B.sh --mode native --steps 6
```

Each step processes 32 prompts × 8 samples = 256 requests + 1 gradient update.
With `--steps 6`: ~1,536 requests total — enough for the prefix-cache scorer to show
KV cache reuse from step 2 onward, and fast enough to turn around both runs in under an hour.

Pass `--force-download` to re-download model/data even if already present.

`$MY_POD_IP` is injected into the head container by the Downward API (see the `env:` block in the manifest). It is the pod's IP — the address where Envoy is listening.

### Colocate mode (single GPU pool for training + rollout)

Replace the separate `--actor-*` / `--rollout-*` node/GPU args with `--colocate`:

```bash
    --sglang-router-ip $MY_POD_IP \
    --sglang-router-port 8081 \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --colocate \
    ...
```

## How EPP routing engages

When slime starts its sglang engines, each one calls `POST /workers` to register itself. Because `--sglang-router-ip/port` points at Envoy, that request hits Envoy → gets routed (without EPP, since ext_proc is disabled on `/workers*`) to the registration shim → shim writes `/tmp/epp-endpoints.yaml`.

EPP is watching that file (`watchFile: true`). Once at least one engine is registered, EPP starts making routing decisions. Every `/generate` from the slime rollout function goes:

```
slime → POST /generate → Envoy:8081
                              ↓ ext_proc gRPC
                           EPP:9002  (prefix-cache-aware pick)
                              ↓ x-gateway-destination-endpoint header
                         Envoy ORIGINAL_DST → chosen sglang engine
```

EPP uses `sglanghttp-parser` to read `input_ids` from slime's `/generate` request body — the native field slime sends, no translation needed. Envoy's `request_body_mode: FULL_DUPLEX_STREAMED` ensures the full request body reaches EPP before the upstream pick is made.

## EPP config

`../epp-config.yaml` is the starting-point config. The key tunable is `windowDurationMs` in `burst-prefix-cache-producer` — it controls how wide a time window EPP uses to co-locate the G group-samples of a GRPO rollout step onto one replica. 1000ms works for most workloads; narrow it if you see unnecessary added latency on fast rollouts.

To update the EPP config on a running cluster:
```bash
bash deploy/kuberay/deploy.sh configmap   # rebuild ConfigMap from ../epp-config.yaml
kubectl rollout restart ... # or just recreate the pod
```

## Logs

| File | Pod | Component |
|---|---|---|
| `/tmp/setup_log.txt` | head + worker | postStart setup output |
| `/tmp/epp.log` | head | EPP subprocess |
| `/tmp/envoy.log` | head | Envoy proxy |
| `/tmp/shim.log` | head | Registration shim |
| `/tmp/ray/session_latest/logs/worker-*.out` | worker | sglang engine logs |

Stream logs from the head pod:
```bash
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/epp.log
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/envoy.log
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/shim.log
```

### Increasing verbosity

Set these in the container `env:` block in `ray-cluster.yaml.tmpl` (uncomment the existing commented blocks):

| Env var | Component | Debug value |
|---|---|---|
| `SLIME_EPP_VERBOSITY` | EPP (`-v`) | `5` |
| `SLIME_ENVOY_LOG_LEVEL` | Envoy (`--log-level`) | `debug` |
