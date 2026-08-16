# Deploying slime on KubeRay

Step-by-step guide for deploying the slime + llm-d cluster and running training. For architecture and how the routing stack works, see [../../README.md](../../README.md).

## Prerequisites

- Kubernetes cluster with GPU nodes (6 GPUs on a single node for the Qwen3-4B example)
- KubeRay operator installed (see [setting-kuberay.md](../../../verl/deploy/kuberay/setting-kuberay.md) — the operator install is framework-agnostic)
- `envsubst` and `kubectl` on your PATH

## Step 1 — Configure

Export your namespace (required — not stored in a file):
```bash
export NAMESPACE=<your-namespace>
```

Images are defined in `deploy.env` — edit tags there rather than in the manifest:

| Variable | Image |
|---|---|
| `IMG_SLIME` | `slimerl/slime:latest` |
| `IMG_CRANE` | `gcr.io/go-containerregistry/crane@sha256:1b1fb24d2b1bb27a9daf81a588157e68463876904e8e537a812edba6284fb252` |
| `IMG_EPP` | `ghcr.io/naomieisen/llm-d-router-epp:sglang-parser` (includes `sglanghttp-parser`) |
| `IMG_ENVOY` | `docker.io/envoyproxy/envoy:distroless-v1.33.2` |

EPP and Envoy are not in the slime image. The `fetch-binaries` init container on the head pod extracts them from their separate public images at pod start using crane. Update a binary without rebuilding by bumping its tag in `deploy.env` and recreating the pod.

If needed, adjust the manifest:
- **GPU count** — `resources.limits.nvidia.com/gpu` defaults to 6; edit to match your node

## Step 2 — Deploy

```bash
bash deploy.sh apply
```

This builds the `llmd-epp-configs-slime` ConfigMap from
[`common/deploy/`](../../../common/deploy/) (`envoy-shim.yaml` and
`epp-config-burst.yaml` rendered with `EPP_PARSER=sglanghttp-parser`) plus this
directory's `run-qwen3-4B.sh`, and applies the rendered cluster manifest. The
shim is `llm-d-registration-shim` from the common package that `postStart` pip-installs.

Useful sub-commands:
```bash
bash deploy.sh render     # print rendered manifest (no kubectl)
bash deploy.sh configmap  # rebuild ConfigMap only
bash deploy.sh delete     # tear down the cluster
```

## Step 3 — Wait for setup

```bash
kubectl get pods -n $NAMESPACE -w

HEAD=$(kubectl get pod -n "$NAMESPACE" -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Check setup log:
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/setup_log.txt

# Setup is done when this exists:
kubectl exec -n $NAMESPACE $HEAD -- test -f /tmp/slime_ready.txt && echo "ready"
```

PostStart completes when slime + Megatron-LM are cloned and llm-d routing is running. Model download and weight conversion happen in the next step.

## Step 4 — Health check (optional)

Verify all three services are up before submitting a training job:

```bash
# EPP - gRPC port open
kubectl exec -n $NAMESPACE $HEAD -- bash -c \
  'echo > /dev/tcp/127.0.0.1/9002 && echo "EPP OK" || echo "EPP DOWN"'

# Envoy - HTTP port open
kubectl exec -n $NAMESPACE $HEAD -- bash -c \
  'echo > /dev/tcp/127.0.0.1/8081 && echo "Envoy OK" || echo "Envoy DOWN"'

# Shim - HTTP port open
kubectl exec -n $NAMESPACE $HEAD -- bash -c \
  'echo > /dev/tcp/127.0.0.1/3001 && echo "Shim OK" || echo "Shim DOWN"'
```

## Step 5 — Run training

Exec into the head pod and run the training script:

```bash
kubectl exec -it -n "$NAMESPACE" "$HEAD" -- bash

# Inside the head pod:

# Run with llm-d routing:
bash /etc/llmd-configs/run-qwen3-4B.sh --mode llm-d

# Run with native with slime's built-in sglang-router
bash /etc/llmd-configs/run-qwen3-4B.sh --mode native
```

`--mode llm-d` points `--sglang-router-ip`/`--sglang-router-port` at Envoy (EPP picks the engine).
`--mode native` omits those args; slime manages its own sglang-router for a baseline comparison.

The script downloads Qwen3-4B, converts weights to Megatron format (once, skipped on re-runs), then submits the Ray job.

To stop a running job:
```bash
ray job list --address=http://127.0.0.1:8265          # find the job ID
ray job stop <job-id> --address=http://127.0.0.1:8265 # graceful stop
```

## Logs

| File | Component |
|---|---|
| `/tmp/setup_log.txt` | postStart setup (slime + Megatron clone, service startup) |
| `/tmp/epp.log` | EPP |
| `/tmp/envoy.log` | Envoy |
| `/tmp/router.log` | `llm-d-rl-router` itself (startup, readiness, child exits) |
| `/tmp/shim.log` | Registration shim |
| `/tmp/ray/session_latest/logs/worker-*.out` | SGLang engine output |

```bash
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/epp.log
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/envoy.log
kubectl exec -n $NAMESPACE $HEAD -- tail -f /tmp/shim.log
```

## EPP config

[`../../../common/deploy/epp-config-burst.yaml`](../../../common/deploy/epp-config-burst.yaml)
is the active config (slime sets `EPP_PARSER=sglanghttp-parser` in `deploy.env`).
To update it on a running cluster:

```bash
bash deploy.sh configmap
```

Then restart EPP to pick up the new config — the ConfigMap is mounted read-only, so the file on disk updates automatically, but EPP only reads it at startup:

```bash
# llm-d-rl-router exits when a child dies, so killing EPP stops the router too;
# start it again the same way postStart does (same flags, one implementation):
kubectl exec -n $NAMESPACE $HEAD -- bash -c 'kill $(pgrep -f llm-d-rl-router) $(pgrep epp)'
kubectl exec -n $NAMESPACE $HEAD -- bash -c '
  nohup llm-d-rl-router \
    --epp-config /etc/llmd-configs/epp-config.yaml \
    --envoy-config /etc/llmd-configs/envoy.yaml \
    >> /tmp/router.log 2>&1 &
'
```

## Increasing verbosity

All components default to quiet logging. Set these env vars in the `env:` block of `ray-cluster.yaml.tmpl`:

| Env var | Component | Default | `info` | `debug` | `trace` |
|---------|-----------|---------|--------|---------|---------|
| `LLMD_EPP_VERBOSITY` | EPP subprocess (`-v`) | `1` | `1`-`3` | `4` | `5` |
| `LLMD_ENVOY_LOG_LEVEL` | Envoy proxy (`--log-level`) | `info` | `info` | `debug` | `trace` |

Both are read by `llm_d_rl_common.router_stack`, which `llm-d-rl-router` uses to build
the EPP and Envoy argv, so they apply to every integration in this repo.
