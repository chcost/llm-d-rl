# Architecture

How the verl + llm-d integration works, the two routing modes, and what is mandatory vs optional.

For how to actually deploy and run it, see the [general deployment guide](deploying.md) and
the [KubeRay walkthrough](../../../quickstart/kuberay/README.md).

## Minimal integration (one Hydra override)

There is no verl fork and no patch. verl already lets you swap the class that manages rollout
generation; this integration ships a drop-in that routes through EPP. You turn it on with a single
Hydra override:

```bash
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager
```

(plus two paths - the EPP config file and where to write the endpoints YAML; see the
[reference tables](configuration.md)).

That one class, and the handful of files it pulls in, is the entire mandatory integration:

```
src/llm_d_rl_verl_integration/
  base_agent_loop_manager.py     starts EPP as a head-node Ray actor, injects the client into workers
  endpoints.py                   writes the endpoints YAML that EPP reads (the replica list)
  llmd_actor.py                  the head-node actor that launches EPP (and, in serving mode, Envoy);
                                 the launching itself lives in llm_d_rl_common.router_stack
  llmd_epp/
    agent_loop_manager.py        entry point - the class named in the override above
    llm_client.py                EPPLLMClient - asks EPP to pick, then dispatches to that replica
    grpc_client.py               the EPP ext_proc gRPC client
  pd_replica.py                  imported at load (PD split factory); harmless when PD is off
```

Everything else in the package is **optional** and only loads if you use it.

### Package map: core vs optional

| Module | Role | Needed for |
|--------|------|------------|
| `base_agent_loop_manager.py`, `endpoints.py`, `llmd_actor.py` | core plumbing | every run |
| `llmd_epp/` | "EPP as the endpoint picker" mode | the basic integration |
| `llmd_epp_sglang/` | "EPP as the endpoint picker" mode, SGLang backend | the SGLang variant of the basic integration |
| `pd_replica.py` | PD prefill/decode split factory (imported at load) | PD runs (inert otherwise) |
| `llmd_serving/` | "llm-d serving" mode (Envoy + EPP HTTP stack) | serving mode only |
| `register_pd.py` | registers the `vllm-llmd-pd` rollout backend | PD runs only |
| `native_logging/` | per-request reqlog + baseline (native) client, for measurement | benchmarking only |
| `tools/search_tool.py` | verl tool that queries the Search-R1 retriever | the searchr1 workload only |

llm-d itself calls EPP the **Endpoint Picker** - the routing "brain" that scores candidate replicas
and is consulted over the ext_proc protocol. This integration uses that brain in two ways.

## How verl drives generation

During each training step, verl generates completions through this component hierarchy:

![verl generate call flow](diagrams/verl-generate-call-flow.png)

`LLMServerClient` is the object `AgentLoopWorker` calls for every generation request. verl's default
implementation uses `GlobalRequestLoadBalancer` to select replicas by least in-flight requests.

The integration replaces two pieces, both via the single Hydra key above, with no verl patches:

- **`AgentLoopManager`** - extended to start EPP (and optionally Envoy) as Ray actors pinned to the
  head node, and to inject a custom `LLMServerClient` into each `AgentLoopWorker`.
- **`LLMServerClient`** - replaced with `EPPLLMClient` (endpoint-picker mode) or `EnvoyLLMClient`
  (serving mode), both of which route through EPP's scoring.

## Mode 1: EPP as the endpoint picker (`llmd_epp/`)

Each generation request is sent to EPP via gRPC ext_proc. EPP scores all available vLLM replicas
(prefix-cache hit rate, queue depth, KV utilization) and returns the chosen backend address.
`EPPLLMClient` then calls that replica's Ray actor directly - the same dispatch path as verl's
built-in client, but with EPP picking the target. **EPP only picks; verl/Ray send the request.**
There is no proxy in the data path.

![epp generate call flow](diagrams/epp-generate-call-flow.png)

Startup, after all vLLM replicas are up:

1. `LlmdRouterAgentLoopManager` spawns `LlmdActor` - a Ray actor pinned to the head node.
2. `LlmdActor` writes the EPP endpoints YAML, starts the EPP subprocess, waits for its health port,
   and returns the gRPC address.
3. `LlmdRouterAgentLoopManager` builds `EPPLLMClient` with that address and replaces
   `self.llm_client` - workers receive it before any generation begins.

### SGLang variant (`llmd_epp_sglang/`)

Same mode, SGLang replicas instead of vLLM (`rollout.name=sglang`, a verl **built-in** backend -
no registration hook needed). `SglangEPPRouterAgentLoopManager` and `SglangEPPLLMClient` mirror
`LlmdRouterAgentLoopManager`/`EPPLLMClient` exactly, with two differences: actor handles are
looked up by SGLang's naming convention (`sglang_server_{rank}_0`, vs vLLM's
`vllm_server_{rank}_0`), and each endpoints-YAML entry is written with an
`llm-d.ai/engine-type: sglang` label so EPP's `core-metrics-extractor` uses the SGLang
Prometheus metric-name mapping instead of the vLLM default. No PD or P2P support for this
backend yet.

## Mode 2: llm-d serving (`llmd_serving/`)

All generation requests go to a single **Envoy** endpoint. Envoy calls EPP via gRPC ext_proc to pick
the best replica, then **Envoy forwards the request to it**. verl workers only ever speak HTTP to one
address; all routing intelligence lives in the Envoy + EPP stack on the head node. In llm-d terms
this is the standalone-mode Router (a self-managed Envoy proxy alongside the EPP).

Startup, after all vLLM replicas are up:

1. `LlmdAgentLoopManager` spawns `LlmdActor` on the head node.
2. `LlmdActor` writes the EPP endpoints YAML, starts EPP, starts Envoy, and returns
   `<head-node-ip>:8081`.
3. `LlmdAgentLoopManager` builds `EnvoyLLMClient` with that address.

### Which mode

| | EPP as the endpoint picker | llm-d serving |
|---|---|---|
| **Data path** | verl/Ray dispatch directly to the picked replica | verl -> Envoy -> replica |
| **Extra processes** | EPP only | EPP + Envoy |
| **Best for** | performance-critical runs; fewer moving parts | closer to a production llm-d serving deployment |
| **PD disaggregation** | yes | yes |

If unsure, start with the endpoint-picker mode - it is simpler and has lower latency.

## PD disaggregation

Both modes support prefill/decode (PD) disaggregation via `rollout.name=vllm-llmd-pd`.

Replicas are split into prefill and decode roles by `PDEngineReplicaFactory`. The first
`prefill_replicas` ranks become prefill replicas; the rest become decode replicas.
`world_size / tp_size` must equal `prefill_replicas + decode_replicas`.

- **Prefill replicas** launch vLLM with NIXL side-channel env vars. They never serve `generate()`
  directly; the decode sidecar pulls KV blocks from them.
- **Decode replicas** launch vLLM with NIXL env vars, then spawn `llm-d-routing-sidecar` alongside.
  The sidecar is the public endpoint: it fetches the KV cache from the prefill replica over NIXL,
  then decodes locally.

Role labels (`llm-d.ai/role: prefill` / `decode`) are written to the EPP endpoints YAML so EPP's
`prefill-filter` and `decode-filter` plugins route correctly.

PD needs a few extra dependencies (NIXL) and vLLM/verl patches that are not in the stock verl
environment image; these are baked directly into
[`deploy/Dockerfile.verl.vllm-p2p`](../../../quickstart/images/Dockerfile.verl.vllm-p2p) - the same image every
other mode (native/EPP/P2P) already uses, so PD needs no separate build or image tag. The PD
Hydra overrides are in the [general deployment guide](deploying.md#pd-disaggregation).

## Endpoints YAML

The endpoints YAML holds the list of available replica addresses (vLLM or SGLang). `LlmdActor`
writes it at startup to the path set by the `epp_endpoints_file` override, and the EPP config's
`file-discovery` plugin has a `path:` that tells EPP where to read it. **Those two paths must
match** (default: `/tmp/epp-endpoints.yaml`). Each entry carries an `llm-d.ai/engine-type` label
(`vllm` by default, `sglang` for the SGLang variant) that EPP's `core-metrics-extractor` uses to
select the right Prometheus metric-name mapping for that replica's backend.
