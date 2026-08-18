"""The llm-d sidecar's job, done in-process.

In production EPP stamps a KV source header and the routing sidecar turns it into
vLLM's kv_transfer_params. Our benchmark topology has neither: engines are Ray
actors sharing one pod, so there is no per-engine sidecar container.

  pd_replica.py     launches the real pd-sidecar per replica as a subprocess
  p2p_replica.py    the same for P2P, plus a path that skips the sidecar and
                    writes kv_transfer_params directly
  p2p_addressing.py one loopback IP per replica, so N engines in one pod are
                    distinct P2P peers
  register_*.py     the rollout backends the above provide

Unlike epp_dev/, this is not expected to graduate - it exists so P2P and PD can
be exercised without the production topology.
"""
