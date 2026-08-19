# verl integration: override reference

GENERATED from `src/llm_d_rl_verl_integration/modes.yaml` by
`llm-d-rl-verl-overrides --markdown`. Edit the YAML, not this file.

Each mode below is a set of Hydra overrides you add to your own verl launch
command. Nothing else is required - no patched verl, and no part of the
quickstart. `${VAR:-default}` is read from the environment.

EPP config paths assume the configs are mounted at `/etc/llmd-configs`; pass
`--epp-config-dir` to change that.

## `--mode epp`

EPP picks the replica over gRPC; verl dispatches to it directly.

```bash
trainer.use_v1=true \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

Environment knobs:
- `EPP_CONFIG` selects a different EPP variant (default `epp-config.yaml`).

## `--mode epp-inflight`

EPP routing on its in-flight counter, no cap.

```bash
trainer.use_v1=true \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-inflight.yaml \
+actor_rollout_ref.rollout.custom.epp_report_completion=true \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

Environment knobs:
- `EPP_CONFIG` selects a different EPP variant (default `epp-config-inflight.yaml`).

## `--mode epp-fc`

EPP routing plus a per-endpoint concurrency cap (flow-control queue).

```bash
trainer.use_v1=true \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-inflight-cap.yaml \
+actor_rollout_ref.rollout.custom.epp_report_completion=true \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

Environment knobs:
- `EPP_CONFIG` selects a different EPP variant (default `epp-config-inflight-cap.yaml`).

## `--mode epp-sglang`

EPP direct-gRPC routing with SGLang replicas instead of vLLM.

```bash
trainer.use_v1=true \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp_sglang.agent_loop_manager.SglangEPPRouterAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
actor_rollout_ref.rollout.name=sglang \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

Environment knobs:
- `EPP_CONFIG` selects a different EPP variant (default `epp-config.yaml`).

## `--mode llm-d`  (manual-only)

verl speaks HTTP to one Envoy endpoint; Envoy plus EPP pick the replica.

> Not wired into a launcher. See modes.yaml for what is missing.

```bash
trainer.use_v1=true \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_serving.agent_loop_manager.LlmdAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
+actor_rollout_ref.rollout.custom.envoy_config=/etc/llmd-configs/envoy.yaml \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

Environment knobs:
- `EPP_CONFIG` selects a different EPP variant (default `epp-config.yaml`).
