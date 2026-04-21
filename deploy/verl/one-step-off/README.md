# veRL + llm-d One-step-Off-Policy

End-to-end deployments for running veRL RL training with llm-d as the
rollout backend (or as vanilla veRL baseline for comparison) for one step policy off training.

Built on the following parts:

1. A **KubeRay cluster** that runs the trainer (head pod = driver, worker
   pods = FSDP GPUs).
2. An **llm-d deployment + Go controller** for rollout
   generation — only needed for `*-llmd-*` runs.

---

## Prerequisites

- `kubectl` configured against your cluster.
- Access to a namespace with GPU nodes.
- `helm` (for installing KubeRay).
- For llm-d runs: a running llm-d deployment reachable from the RayCluster
  (service endpoint: `http://<controller-svc>.<namespace>.svc.cluster.local:8090`).

---

## GPU resources

The one-step-off configs are set up for a **2 GPU training / 4 GPU rollout**
split (6 GPUs of accelerator work total, not counting the CPU-only Ray
head). For the setting of verl + llmd rollout, the resoures split between 2 GPUs for the RayWorker, and 4 GPUs for vllm engines (one for each pod, so 4 replicas). For the verl vanilla setup, all the 6 GPUs allocated to the RayWorker, when they are splitted by Ray between 2 GPUs for training, and 4 GPUs for rollout (vllm). That is set with the Hydra settings: **2** FSDP actor GPUs and **4**
rollout GPUs with `tensor_model_parallel_size: 1` (one vLLM engine per
GPU).

### Vanilla veRL baseline (`ray-cluster-verl.yaml`)

| Where | GPUs | Role |
|---|---:|---|
| Ray **head** pod | **0** | Driver / `fit()` / Ray GCS only (CPU). |
| Ray **worker** pod(s) | **6** on the worker template in `ray-cluster-verl.yaml` | Shared by veRL’s two pools on the **same** Ray nodes: **2** for FSDP training (`actor_rollout_ref.actor.n_gpus_per_node` in `verl-config.yaml`) and **4** for in-process vLLM rollout servers (`rollout.n_gpus_per_node`). |

Everything runs **inside the Ray cluster**: training and inference GPUs are
**co-located** on one worker group. The worker `resources.limits` must be
large enough for both pools (here: 2 + 4 = 6× `nvidia.com/gpu` on a single
worker with `replicas: 1`).

### llm-d rollout (`ray-cluster-llmd.yaml` + llm-d engine manifests)

| Where | GPUs | Role |
|---|---:|---|
| Ray **head** pod | **0** | Same as baseline — CPU-only driver. |
| Ray **worker** pod(s) | **2** on the worker in `ray-cluster-llmd.yaml` | **Training only** (FSDP). Rollout does not reserve GPUs in Ray (`gpu_memory_utilization: 0.0` for the rollout slot in `llm-d-config.yaml`). |
| **llm-d** vLLM StatefulSet (see `vllm-engine*.yaml` in the same folder) | **4** (typically **4** pods × **1** GPU each) | **Rollout only** — separate from Ray; the Go controller load-balances HTTP across these engines. |

Training and inference GPUs are on **different** workloads: KubeRay for the
trainer, llm-d for vLLM. For a fair comparison with the vanilla run, keep the
**same** 2 + 4 split and the same model / sampling settings; only the
**placement** of the four rollout GPUs changes (inside Ray vs llm-d pods).

---

## Deploying an experiment

All commands below assume you are in the experiment subfolder, e.g.
`deploy/verl/one-step-off/`.

### 1. Install KubeRay (once per namespace)

```bash
export NAMESPACE=<your-namespace>
bash prereqs.sh
```

Installs the KubeRay operator + CRDs into `$NAMESPACE`. Safe to re-run;
CRD installation is idempotent.

### 2. Deploy llm-d (only for llm-d runs)

Deploy the llm-d Go controller, and epp as described in README-llmd.md. For the vllm engine, seploy the yaml within this folder:

```bash
kubectl apply -f vllm-engine.yaml
```

Wait until vLLM pods are `Ready`:

```bash
kubectl get pods -l app=vllm -w
```
The vllm-engine in this folder are compatible with verl vllm version (vllm 017) and set with the exact arguments as verls vllm.

### 3. Deploy the RayCluster

Two flavors, depending on which training config you'll run:

```bash
# For running verl-config.yaml (vanilla veRL rollout inside Ray workers)
kubectl apply -f ray-cluster-verl.yaml

# For running llm-d-config.yaml (rollout via llm-d pods)
kubectl apply -f ray-cluster-llmd.yaml
```

Both files define:
- 1 head pod (CPU only, runs the driver / `fit()` loop).
- 1 worker pod (GPU, run FSDP).
- Shared env: `LLMD_CONTROLLER_URL`, `WANDB_API_KEY`, `NCCL_DEBUG`,
  `VERL_LOGGING_LEVEL`.

Wait for the cluster to be ready:

```bash
kubectl get raycluster verl-cluster -w
# STATUS = ready
```

### 4. Apply the configs ConfigMap

Training configs live in a `ConfigMap` (`verl-configs-cm.yaml`) containing
both `verl-config.yaml` and `llm-d-config.yaml` inline. The RayCluster
yamls already mount it at `/etc/verl-configs` on both head and worker
pods.

Apply it once (and any time configs change):

```bash
kubectl apply -f verl-configs-cm.yaml
```

Sanity-check that the file is in the pod:

```bash
kubectl exec ${HEAD_POD} -- ls /etc/verl-configs
# -> llm-d-config.yaml  verl-config.yaml
```

Notes:
- The `ConfigMap`'s `namespace:` must match the RayCluster namespace
  (default in the yaml: `llm-d-rl`).
- When you edit `verl-configs-cm.yaml` and re-`apply`, kubelet resyncs
  the mounted files within ~60s. For immediate effect, roll the pods:
  ```bash
  kubectl delete pods -l ray.io/cluster=verl-cluster
  ```
- To regenerate the ConfigMap directly from the on-disk yamls (instead
  of hand-editing the inline copy), use:
  ```bash
  kubectl create configmap verl-configs \
      --from-file=verl-config.yaml \
      --from-file=llm-d-config.yaml \
      --dry-run=client -o yaml | kubectl apply -f -
  ```

### 5. Run training

Exec into the head pod:

```bash
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec -ti ${HEAD_POD} -- bash
```

Then launch training. The command picks up the config from the mounted
ConfigMap at `/etc/verl-configs`:

**llm-d backend** (rollout via vLLM pods):

```bash
python -m verl.experimental.one_step_off_policy.main_ppo \
    --config-path /etc/verl-configs \
    --config-name llm-d-config \
    data.train_files=/tmp/verl/data/gsm8k/train.parquet \
    data.val_files=/tmp/verl/data/gsm8k/test.parquet
```

**Vanilla veRL baseline** (rollout inside Ray workers):

```bash
python -m verl.experimental.one_step_off_policy.main_ppo \
    --config-path /etc/verl-configs \
    --config-name verl-config \
    data.train_files=/tmp/verl/data/gsm8k/train.parquet \
    data.val_files=/tmp/verl/data/gsm8k/test.parquet
```

Logs stream to stdout (visible via `kubectl logs ${HEAD_POD}`) and to
Weights & Biases (project: `verl-benchmark`).

---

## Common operations

### Restart after a killed trainer

If you kill training with Ctrl+C, the vLLM pods may retain stale NCCL
state that blocks the next rendezvous. Restart them:

```bash
kubectl rollout restart statefulset <vllm-statefulset-name>
```

### Change log verbosity

Edit `VERL_LOGGING_LEVEL` in the RayCluster yaml and redeploy:

```yaml
- name: VERL_LOGGING_LEVEL
  value: "DEBUG"    # or INFO / WARN / ERROR
```

Then:

```bash
kubectl apply -f ray-cluster-llmd.yaml
kubectl delete pods -l ray.io/cluster=verl-cluster     # pods get env vars at creation
```

---

## See also

- [`../../python/llmd_verl/README.md`](../../python/llmd_verl/README.md) —
  architecture and class reference for the `llmd_verl` package that
  provides the veRL ↔ llm-d integration.
