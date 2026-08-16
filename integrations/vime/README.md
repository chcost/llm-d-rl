# vime + llm-d

[vime](https://github.com/vllm-project/vime) is a vLLM-based RL training framework. This integration replaces vime's built-in `vllm_router` with **llm-d routing**.

**Zero code changes to vime.** When `--vllm-router-ip` / `--vllm-router-port` is set, vime's `_start_router` returns immediately without starting the built-in router ([`rollout.py:1035`](https://github.com/vllm-project/vime/blob/main/vime/ray/rollout.py#L1035)). All rollout traffic goes to that address. We point it at Envoy.

## Components

The [llm-d router stack](../common/README.md#llm-d-router-stack) (EPP, Envoy) and the
[registration shim](../common/README.md#registration-shim).

This integration starts the shim as:

```
llm-d-router-shim --engine-type vllm
```

## Get started

- **[deploy/kuberay/](deploy/kuberay/README.md)** — end-to-end KubeRay example: cluster, configs, train script
- **[deploy/README.md](deploy/README.md)** — instructions for any Ray cluster
