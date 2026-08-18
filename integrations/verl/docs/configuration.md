# verl integration: override reference

GENERATED from `src/llm_d_rl_verl_integration/modes.yaml` by
`llm-d-rl-verl-overrides --markdown`. Edit the YAML, not this file.

Each mode below is a set of Hydra overrides you add to your own verl launch
command. Nothing else is required - no patched verl, and no part of the
quickstart. `${VAR:-default}` is read from the environment.

EPP config paths assume the configs are mounted at `/etc/llmd-configs`; pass
`--epp-config-dir` to change that.

## `--mode native`

stock verl routing, plus reqlog and the endpoints YAML for the scraper.

```bash
trainer.use_v1=true \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.native_logging.agent_loop_manager.NativeLoggingAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

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

## `--mode epp-p2p`

EPP routing plus P2P KV-cache sharing; a per-replica sidecar turns EPP's source header into kv_transfer_params.

```bash
trainer.use_v1=true \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-p2p.yaml \
actor_rollout_ref.rollout.name=vllm-llmd-p2p \
+actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_p2p \
+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=llm_d_rl_verl_integration.register_p2p \
+actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=64 \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=OffloadingConnector \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.offload_prompt_only=false \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.cpu_bytes_to_use=4294967296 \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.spec_name=TieringOffloadingSpec \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.secondary_tiers=[{type:p2p}] \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

Environment knobs:
- `EPP_CONFIG` selects a different EPP variant (default `epp-config-p2p.yaml`).

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

## `--mode wave-admission`

estimation-gated admission with sticky placement, no EPP.

```bash
trainer.use_v1=true \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.wave_admission.agent_loop_manager.WaveAdmissionAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

## `--mode wave-admission-p2p`

wave-admission on the P2P backend, so a migration pulls the resident replica's KV.

```bash
trainer.use_v1=true \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.wave_admission.agent_loop_manager.WaveAdmissionAgentLoopManager \
actor_rollout_ref.rollout.name=vllm-llmd-p2p \
+actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_p2p \
+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=llm_d_rl_verl_integration.register_p2p \
+actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=64 \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=OffloadingConnector \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.offload_prompt_only=false \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.cpu_bytes_to_use=4294967296 \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.spec_name=TieringOffloadingSpec \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.secondary_tiers=[{type:p2p}] \
+actor_rollout_ref.rollout.custom.wave_admission_p2p_kv_available=true \
+actor_rollout_ref.rollout.custom.wave_admission_reserve_mode=size \
+actor_rollout_ref.rollout.custom.wave_admission_reserve_z=1.5 \
+actor_rollout_ref.rollout.custom.wave_admission_max_wait_s=20 \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

Environment knobs:
- `WAVE_ADMISSION_P2P_NOSIDECAR=true` adds 3 more override(s).

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
