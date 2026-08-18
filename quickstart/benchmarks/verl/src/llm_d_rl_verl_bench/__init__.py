"""Benchmark and research harness for the verl llm-d integration.

Nothing here is part of the integration. It exists to measure it, and to try
ideas that belong in EPP or the sidecar before they are there:

  epp_dev/         EPP's own job, prototyped in Python (placement, migration,
                   source selection) so it can be evaluated before it becomes an
                   EPP plugin.
  inproc_sidecar/  the sidecar's job, done in-process. Our engines are Ray actors
                   inside one pod, so there is no sidecar container to inject
                   kv_transfer_params - these launch or replace it.
  native_logging/  verl's own routing plus a reqlog, so a native arm is
                   comparable with an EPP arm. The A/B control.
  trace_player/    replay a conversation trace (token counts + gaps, no text) to
                   isolate routing cost from model quality.
  tools/           per-workload tools (searchr1's retriever client).

Depends on llm-d-rl-verl-integration; the reverse must never be true.
"""
