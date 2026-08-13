"""Shared per-request JSONL logging helpers.

All LLM clients write the same reqlog format; this module holds the shared
plumbing so each client only needs to manage its own file handle and timing.

Usage in each client::

    # in __setstate__ (after Ray unpickling):
    self._reqlog_f = open_reqlog()
    self._turn_counts: dict[str, int] = {}

    # in generate():
    log_request(self._reqlog_f, {"ts": ..., "prompt_hash": phash(prompt_ids), ...})
"""

from __future__ import annotations

import hashlib
import json
import os


def phash(prompt_ids) -> str:
    """Return a BLAKE2b-8 hex digest of the token ID list."""
    try:
        b = b",".join(str(int(t)).encode() for t in prompt_ids)
        return hashlib.blake2b(b, digest_size=8).hexdigest()
    except Exception:
        return ""


def open_reqlog():
    """Open the per-process JSONL log file if VERL_REQLOG_DIR is set."""
    d = os.environ.get("VERL_REQLOG_DIR")
    if not d:
        return None
    try:
        os.makedirs(d, exist_ok=True)
        return open(os.path.join(d, f"reqlog-{os.getpid()}.jsonl"), "a", buffering=1)
    except Exception:
        return None


def log_request(f, rec: dict) -> None:
    """Write one JSON record to the reqlog file. No-op if f is None."""
    if f is None:
        return
    try:
        f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def tag_global_steps(out) -> None:
    """Fill min/max_global_steps on a TokenOutput from its global_steps.

    verl's trainer int()s both on every trajectory tag, so a client that leaves
    them unset produces None and a TypeError that kills the run after the first
    rollout. The server side records only global_steps; one generate() call is
    served by one weight version, so min == max == that value.
    """
    fields = getattr(out, "extra_fields", None)
    if not isinstance(fields, dict):
        return
    gs = fields.get("global_steps")
    fields.setdefault("min_global_steps", gs)
    fields.setdefault("max_global_steps", gs)
