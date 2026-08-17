# Running the KubeRay Example

A complete end-to-end example of running verl RL training with the llm-d integration on Kubernetes - single-node GRPO on GSM8K with Qwen3-4B. Includes a KubeRay cluster manifest with all necessary image references (verl, EPP, Envoy, sidecar), config files, and automated scripts for deployment, training, and benchmarking.

The manifest has an **8-GPU** worker option active by default; a 4-GPU option is also provided (commented out). Pick the set of run commands below that matches the worker `resources` block you enabled.

**Rollout engine:** one manifest serves both vLLM and SGLang. `deploy.sh` selects an engine
with `--engine` (default `vllm`) and reads its column from [`deploy.env`](deploy.env):

```bash
bash deploy/kuberay/deploy.sh apply                    # vLLM
bash deploy/kuberay/deploy.sh apply --engine sglang    # SGLang
```

Only four things differ between the two, all values rather than structure: the environment
image (`IMG_VERL` / `IMG_VERL_SGLANG`), the head's `rayStartParams.num-cpus`, which Python
module the `postStart` sanity-check imports, and `PYTORCH_CUDA_ALLOC_CONF`. Adding an engine
means adding an `ENGINE_<name>_*` block to `deploy.env`, not a manifest.

The two engines are **mutually exclusive in a namespace** - they render to the same
RayCluster name, so applying one replaces the other. That is deliberate: every script here
finds pods with `-l ray.io/node-type=head` and `items[0]`, which cannot distinguish two
clusters, so two live clusters would mean silently benchmarking (or `push-epp.sh`-ing into)
whichever one answered first.

SGLang covers the EPP-direct-gRPC routing mode only - no `llmd_serving`/Envoy, no PD, no
P2P. Its cluster still fetches the Envoy and sidecar binaries and still carries the
vLLM-only env vars; they are inert when unused, and keeping them means switching engines
never needs a manifest change. Two SGLang platform quirks the shared manifest encodes, both
verified end-to-end on a real GPU cluster (GRPO/GSM8K, Qwen3-4B, 8 SGLang replicas):

- `num-cpus` is `"0"` for SGLang, `"4"` for vLLM. verl's `TaskRunnerV1` driver has no
  GPU/node pinning, and SGLang's `sgl_kernel` extension (unlike vLLM's Python import)
  hard-requires `libcuda.so.1` just to import `SGLangReplica`, which crashes if the driver
  lands on the GPU-less head.
- The stock `verlai/verl:sgl*` image doesn't set `PIP_BREAK_SYSTEM_PACKAGES=1` the way the
  vLLM env image does in its Dockerfile, so `postStart` sets it before any `pip install`
  (Ubuntu 24.04 PEP 668 guard). It is now set unconditionally - a no-op where the image
  already sets it.

## Prerequisites

- Kubernetes cluster with GPU nodes
- KubeRay CRD and operator installed (see [setting-kuberay.md](setting-kuberay.md) for instructions)

## Step 1 - Set the namespace and images

- **Namespace (required)** - export it in your shell: `export NAMESPACE=<your-namespace>`.
  It is not stored in a file (it is per-user, and keeping it out of `deploy.env` avoids
  committing a personal namespace). `deploy/kuberay/deploy.sh`, `benchmarks/scripts/utils/push-epp.sh`,
  `benchmarks/scripts/rl_orchestrate.sh`, and `benchmarks/scripts/run_on_head.sh` read it from the
  environment and **fail fast** if it is unset.
- **Images** - every runtime image (verl, crane, EPP, Envoy) is defined in `deploy.env`.
  Edit tags there rather than in the manifest; `deploy.sh` substitutes them (and `NAMESPACE`)
  into `ray-cluster.yaml.tmpl` at apply time.

The manifest template itself only needs edits for node/GPU layout:

- **GPU count** - the worker `resources` block ships with the 8-GPU option active and the
  4-GPU option commented out. Enable whichever matches your node.
- **Node placement** - the head co-locates onto the worker's node via `podAffinity`, and the
  worker is anchored to a GPU node by its `nvidia.com/gpu` request. The worker `nodeAffinity`
  has a `NotIn` list excluding known-faulty GPU hosts (e.g. `<faulty-node-name>`) - edit that
  list for your cluster.

None of the EPP, Envoy, or sidecar binaries are baked into the verl image. On the head the
`fetch-binaries` init container extracts the EPP and Envoy (`IMG_EPP`, `IMG_ENVOY`); on the
worker the `fetch-sidecar` init container extracts the sidecar (`IMG_SIDECAR`) - all set in
`deploy.env` and pulled on pod start. Use `benchmarks/scripts/utils/push-epp.sh` to push a new EPP into a
running pod without recreating it. See the [deployment guide](../README.md) for details on the binaries.

## Step 2 - Deploy

`deploy.sh apply` does everything: it builds the `llmd-epp-configs` ConfigMap from
[`common/configs/`](../../../common/configs/) (the rendered burst EPP profile)
plus `common/configs/envoy.yaml` and this tree's variants (`epp-config-pd.yaml`, p2p, inflight) and
applies the rendered cluster manifest into `$NAMESPACE`.

```bash
bash deploy/kuberay/deploy.sh apply
```

Useful sub-commands: `deploy.sh configmap` ((re)create just the ConfigMap),
`deploy.sh render` (print the rendered manifest without applying), `deploy.sh delete`
(tear down the cluster). Each takes an optional `--engine vllm|sglang`; the ConfigMap is
the same either way.

Wait for both pods to be ready:
```bash
kubectl get pods -w
```

The `postStart` hook on each pod installs the integration package with pip install and pre-downloads GSM8K and Qwen3-4B. Training should not start until both pods report `Ready`.

## Step 3 - Run training

### Quick start (native / EPP)

Running a job normally means logging into the head pod and launching training from
there (the [Manual](#manual-all-modes) path below). Two scripts automate that so a
run is a single command from your laptop:

- `benchmarks/scripts/run_on_head.sh` - laptop-side launcher: resolves the head pod by its Ray label, copies `run_test.sh` and the selected `workloads/<task>/` folder onto it, and runs it there (namespace from `$NAMESPACE`).
- `benchmarks/scripts/run_test.sh` - runs on the pod: sources `workloads/<task>/task.env` for the dataset/overrides and wraps verl's `run_qwen3_4b_fsdp.sh` with the right Hydra overrides for the chosen `--mode`.

Where modes (`--mode`):

- `native` - baseline: verl's built-in replica routing, no EPP.
- `epp` - EPP picks the endpoint and verl/Ray dispatch to it (the "EPP as the endpoint picker" mode).
- `llm-d` - Envoy + EPP HTTP stack (the "llm-d serving" mode); not yet implemented in `run_test.sh`.

Other options (`--steps`, `--tp`, `--n`, `--name`, `--reqlog`) and their defaults
are documented in the header comment of [benchmarks/scripts/run_test.sh](../../benchmarks/scripts/run_test.sh).
`benchmarks/scripts/run_on_head.sh --help` covers the launcher's own flags.

Examples:
```bash
benchmarks/scripts/run_on_head.sh --mode epp                    # background on the pod, EPP picks the endpoint, tails the log
benchmarks/scripts/run_on_head.sh --fg --mode native            # run attached (foreground), verl's native routing
benchmarks/scripts/run_on_head.sh --mode epp --steps 20 --tp 2
```

By default `run_on_head.sh` executes on the pod in the background (survives a laptop
disconnect; Ctrl-C just detaches the tail) and streams `/tmp/train.log`; `--fg`
runs it attached. All other args pass straight through to `run_test.sh`.

For Envoy or PD-disaggregated modes, 8-GPU sizing, or to tweak the Hydra
overrides directly, exec in and run the explicit commands below.

### Manual (all modes)

Exec into the head pod, then run one of the commands below. Resolve the head pod
by its Ray label rather than hardcoding the name (the `verl-cluster-head-xxxxx`
suffix changes on every recreation):

```bash
export NAMESPACE=<your-namespace>   # if not already set (see Step 1)
HEAD=$(kubectl get pod -n "$NAMESPACE" -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it -n "$NAMESPACE" "$HEAD" -- bash
cd /tmp/verl/verl/examples/grpo_trainer
```

All commands use verl's own `run_qwen3_4b_fsdp.sh` as the base script and pass the integration overrides via `$@`. `hydra.run.dir` is required because the default `./outputs/` path is read-only in the container.

### EPP - direct gRPC routing

```bash
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_fsdp_8gpu \
bash /tmp/verl/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### EPP - direct gRPC routing, PD disaggregated

```bash
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_pd_fsdp_8gpu \
bash /tmp/verl/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=2 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### EPP - direct gRPC routing (SGLang)

Requires a cluster deployed with `--engine sglang`. `rollout.name=sglang` is a verl
**built-in** backend - no `model.external_lib` registration hook needed, unlike the PD/P2P vLLM
modes above.

```bash
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_sglang_epp_fsdp_8gpu \
bash /tmp/verl/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.name=sglang \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp_sglang.agent_loop_manager.SglangEPPRouterAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### llm-d stack (Envoy + EPP - HTTP proxy routing)

```bash
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_envoy_fsdp_8gpu \
bash /tmp/verl/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_serving.agent_loop_manager.LlmdAgentLoopManager \
    +actor_rollout_ref.rollout.custom.envoy_config=/etc/llmd-configs/envoy.yaml \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### llm-d stack (Envoy + EPP - HTTP proxy routing, PD disaggregated)

```bash
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_envoy_pd_fsdp_8gpu \
bash /tmp/verl/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=2 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_serving.agent_loop_manager.LlmdAgentLoopManager \
    +actor_rollout_ref.rollout.custom.envoy_config=/etc/llmd-configs/envoy.yaml \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

## EPP config

[`../../../common/configs/epp-config-burst.yaml`](../../../common/configs/epp-config-burst.yaml)
(standard, parser defaults to `vllmhttp-parser`) and `../epp-config-pd.yaml`
(PD disaggregated) are the starting-point configs. Customize scorer weights or swap plugins to tune routing for your workload.

The path is passed per run via `+actor_rollout_ref.rollout.custom.epp_config_file=...` (see the commands above). You can point to any file accessible on the head node - mount your own ConfigMap, copy a file to `/tmp`, or use the sample directly in non-k8s environments.

To update a running cluster after editing a config file, re-run the `kubectl create configmap` command from Step 2 above, then recreate the pod.

See the [deployment guide](../README.md) for the full config reference and
[architecture](../../docs/architecture.md) for an overview.


## 4-GPU Option

The same scripts work with a 4-GPU Ray cluster by adjusting a few parameters. Run from inside the head pod (`kubectl exec -it <head-pod> -- bash`).

### EPP - direct gRPC routing

```bash
NGPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=256 \
PPO_MINI_BATCH_SIZE=128 \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_fsdp_4gpu \
bash /tmp/verl/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### EPP - direct gRPC routing, PD disaggregated

```bash
NGPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=256 \
PPO_MINI_BATCH_SIZE=128 \
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
ROLLOUT_GPU_MEM_UTIL=0.6 \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_pd_fsdp_4gpu \
bash /tmp/verl/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=1 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### llm-d stack (Envoy + EPP - HTTP proxy routing)

```bash
NGPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=256 \
PPO_MINI_BATCH_SIZE=128 \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_envoy_fsdp_4gpu \
bash /tmp/verl/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_serving.agent_loop_manager.LlmdAgentLoopManager \
    +actor_rollout_ref.rollout.custom.envoy_config=/etc/llmd-configs/envoy.yaml \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### llm-d stack (Envoy + EPP - HTTP proxy routing, PD disaggregated)

```bash
NGPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=256 \
PPO_MINI_BATCH_SIZE=128 \
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_envoy_pd_fsdp_4gpu \
bash /tmp/verl/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=1 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_serving.agent_loop_manager.LlmdAgentLoopManager \
    +actor_rollout_ref.rollout.custom.envoy_config=/etc/llmd-configs/envoy.yaml \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

## Logs

#### Training logs (verl)

verl's file logger writes per-step training metrics (rewards, loss, timing) to the directory set by `VERL_FILE_LOGGER_ROOT`. In the example commands this is `/tmp/verl/logs` on the **head pod**. Each training step appends a JSON line to a file in that directory - useful for plotting reward curves or diagnosing training instability.

The file path is:
```
<VERL_FILE_LOGGER_ROOT>/<trainer.project_name>/<trainer.experiment_name>.jsonl
```

`trainer.project_name` and `trainer.experiment_name` are Hydra config fields, overridden in the run script via the `PROJECT_NAME` and `EXPERIMENT_NAME` env vars. In the example commands above these are set explicitly, for example:
```bash
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_pd_fsdp_8gpu \
```
which produces:
```
/tmp/verl/logs/verl_grpo_gsm8k_examples/qwen3_4b_grpo_vllm_epp_pd_fsdp_8gpu.jsonl
```

```bash
kubectl exec <head-pod> -- tail -f /tmp/verl/logs/*.jsonl
```

#### Component log files

Each integration component writes its output to a fixed file path on the pod it runs on:

| File | Pod | Component | Contents |
|------|-----|-----------|----------|
| `/tmp/epp.log` | head | EPP subprocess | Endpoint scoring decisions, plugin output, gRPC ext_proc traffic |
| `/tmp/envoy.log` | head | Envoy proxy | HTTP request routing, upstream selection, connection errors |
| `/tmp/sidecar-decode-{rank}.log` | worker | llm-d routing sidecar (one per decode replica) | NIXL V2 protocol - prefill calls, `kv_transfer_params` received, decode forwarding |
| `/tmp/ray/session_latest/logs/worker-*.out` | worker | vLLM prefill and decode engines | vLLM engine logs including NIXL KV transfer traces when `VERL_VLLM_LOG_LEVEL=DEBUG` |

To stream a log live:
```bash
kubectl exec <head-pod> -- tail -f /tmp/epp.log
kubectl exec <worker-pod> -- tail -f /tmp/sidecar-decode-0.log
```

#### Increasing verbosity

All components default to quiet logging. Set these env vars to increase verbosity in the `env:` section of your KubeRay `RayCluster` / `RayJob` container spec.

| Env var | Component | Default | `info` | `debug` | `trace` |
|---------|-----------|---------|--------|---------|---------|
| `VERL_VLLM_LOG_LEVEL` | vLLM inside prefill and decode replicas (`VLLM_LOGGING_LEVEL`) | unset (vLLM default) | `INFO` | `DEBUG` | - |
| `VERL_SIDECAR_LOG_LEVEL` | llm-d routing sidecar (`--zap-log-level`) | `1` | `1`-`3` | `4` | `5` |
| `VERL_EPP_VERBOSITY` | EPP subprocess (`-v`) | `1` | `1`-`3` | `4` | `5` |
| `VERL_ENVOY_LOG_LEVEL` | Envoy proxy (`--log-level`) | `info` | `info` | `debug` | `trace` |

*Note: Ray actors are spawned as new processes on remote nodes and do not inherit the launching shell's environment.*

With *KubeRay* - set in the container spec; vars are present before Ray starts:

```yaml
containers:
  - name: ray-worker
    env:
      - name: VERL_VLLM_LOG_LEVEL
        value: "DEBUG"
      - name: VERL_EPP_VERBOSITY
        value: "5"
```


## Saving Rollout Generations (Optional)

To save the model's generated outputs during training and validation, add these overrides to any command above:

```
trainer.validation_data_dir=/tmp/verl/generations/val \
trainer.rollout_data_dir=/tmp/verl/generations/train \
```

Outputs are written as parquet files to the specified directories on the head node. This is useful for inspecting model behavior or offline reward analysis.
Make sure you have write permission to the destination path.
