"""Router-agnostic trace player.

Importing this package registers the ``trace_player`` agent loop via the
``@register`` side effect. The workload also lists it in an
``agent_loop_config_path`` YAML so registration is guaranteed at AgentLoopWorker
init regardless of routing mode (native / epp / epp-fc).
"""

from llm_d_rl_verl_bench.trace_player import agent_loop  # noqa: F401  (triggers @register)
