# slime + llm-d

[Slime](https://github.com/THUDM/slime) is an SGLang-based RL training framework. This integration replaces slime's built-in sglang-router with **llm-d routing**.

**Zero code changes to slime.** The only difference from a standard slime run is pointing `--sglang-router-ip` / `--sglang-router-port` at Envoy. Slime treats that address as the router.

## Components

The [llm-d router stack](../common/README.md#llm-d-router-stack) (EPP, Envoy) and the
[registration shim](../common/README.md#registration-shim).

This integration starts the shim as:

```
llm-d-router-shim --engine-type sglang --id-field id
```

## Get started

- **[quickstart/kuberay/](../../quickstart/kuberay/README.md)** — end-to-end KubeRay example: cluster, configs, train script
- **[docs/deploying.md](docs/deploying.md)** — instructions for wiring llm-d routing into an existing slime training setup.
