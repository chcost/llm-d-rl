# slime + llm-d integration

Route [slime](https://github.com/PRIME-RL/SLIME) GRPO training rollouts through
[llm-d](https://github.com/llm-d/llm-d)'s **Endpoint Picker (EPP)**. During each training step
slime generates completions from a pool of SGLang engines; this integration replaces slime's
built-in sglang-router with EPP, which steers each request to the engine most likely to already
have its KV cache warm, so a prompt prefix shared across a GRPO group is not re-prefilled on
every engine.

No slime source changes are required — the only difference from a standard slime run is pointing
`--sglang-router-ip`/`--sglang-router-port` at Envoy (the EPP-backed proxy) instead of the
built-in router.

## Get started

- **Run it:** [`deploy/kuberay/`](deploy/kuberay/README.md) — a complete, runnable end-to-end
  example on Kubernetes (KubeRay): cluster manifest, configs, and training script.

## How it works

Slime normally connects its rollout engines to a `sglang-router`. This integration substitutes
an Envoy proxy backed by EPP. Slime's `--sglang-router-ip`/`--sglang-router-port` points at
Envoy; every `/generate` request goes through Envoy's ext_proc filter, which calls EPP over
gRPC to pick the best engine based on prefix-cache state, then forwards there directly.

```
slime rollout → POST /generate → Envoy:8081
                                      ↓ ext_proc gRPC (full request body forwarded)
                                   EPP:9002  (sglanghttp-parser reads input_ids,
                                              burst-prefix-cache scorer picks engine)
                                      ↓ x-gateway-destination-endpoint header
                                 Envoy ORIGINAL_DST → chosen SGLang engine
```

Engine registration also flows through Envoy: each SGLang engine calls `POST /workers` to the
Envoy address on startup. Envoy routes those calls (without ext_proc) to a lightweight
registration shim on `localhost:3001`. The shim writes `/tmp/epp-endpoints.yaml`; EPP watches
that file and begins routing once at least one engine is registered.

---

This is the `slime` integration of
[llm-d-rl](https://github.com/llm-d-incubation/llm-d-rl), where it lives at
`integrations/slime/`.
