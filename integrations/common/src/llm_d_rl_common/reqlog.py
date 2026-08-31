"""Shared per-request JSONL logging helpers.

All LLM clients write the same reqlog format; this module holds the shared
plumbing so each client only needs to manage its own file handle and timing.

Usage in each client::

    # in __setstate__ (after Ray unpickling):
    self._reqlog_f = open_reqlog()
    self._turn_counts: dict[str, int] = {}

    # in generate():
    log_request(self._reqlog_f, {"ts": ..., "prompt_hash": phash(prompt_ids), ...})

`log_request` never touches the filesystem itself: it puts the record on an
in-memory queue and returns immediately (never blocks the calling coroutine).
A single background thread per process (started lazily by the first
open_reqlog() call) drains that queue and writes each record to disk as soon
as it's dequeued -- so data reaches disk in near-real-time without any
synchronous I/O on the hot path.

This replaces two earlier, weaker designs from the same day (2026-08-27,
idle-GPU/long-tail investigation): a pure atexit-flush design lost an entire
run's data (atexit is not called for a Ray actor process torn down via
os._exit()-style termination -- confirmed directly: reqlog files came out at
exactly one threshold-flush's worth of lines, nothing after); a
threshold-flush-every-N-records design shrank but did not eliminate the loss
(files still landed on an exact multiple of N, meaning the last partial batch
before exit was still lost every time). Only a design that writes continuously,
not batched behind a threshold or exit hook, actually closes that gap -- the
only remaining loss window is whatever is unwritten at the exact instant
`os._exit()` fires, which is now bounded by disk-write latency for one record,
not by a batch size or a hook that may never run.

`open_reqlog()`'s return value is an opaque handle to callers; only this module
touches its internals.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading


def phash(prompt_ids) -> str:
    """Return a BLAKE2b-8 hex digest of the token ID list."""
    try:
        b = b",".join(str(int(t)).encode() for t in prompt_ids)
        return hashlib.blake2b(b, digest_size=8).hexdigest()
    except Exception:
        return ""


class _ReqlogHandle:
    """One background writer thread + queue per process."""

    __slots__ = ("path", "queue", "thread")

    def __init__(self, path: str):
        self.path = path
        self.queue: "queue.Queue[str | None]" = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="reqlog-writer")
        self.thread.start()

    def _run(self) -> None:
        # One open file handle for the thread's lifetime; each item is written
        # (and flushed) as soon as it's dequeued -- no batching, so the queue
        # never accumulates more than whatever arrived since the last write.
        try:
            f = open(self.path, "a", buffering=1)
        except Exception:
            f = None
        while True:
            item = self.queue.get()
            if item is None:  # sentinel: explicit shutdown, not used today but future-proof
                break
            if f is not None:
                try:
                    f.write(item)
                except Exception:
                    pass


def open_reqlog():
    """Return a reqlog handle (with its own background writer thread) if
    VERL_REQLOG_DIR is set. Returns None (a no-op handle) if the env var is unset."""
    d = os.environ.get("VERL_REQLOG_DIR")
    if not d:
        return None
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"reqlog-{os.getpid()}.jsonl")
        return _ReqlogHandle(path)
    except Exception:
        return None


def log_request(f, rec: dict) -> None:
    """Enqueue one JSON record for the background writer. No-op if f is None.

    Never touches the filesystem itself -- json.dumps + a queue.put, both fast
    and non-blocking; the background thread does the actual write."""
    if f is None:
        return
    try:
        f.queue.put_nowait(json.dumps(rec) + "\n")
    except Exception:
        pass


def flush_reqlog(f) -> None:
    """No-op, kept only so callers written against the earlier buffered design
    don't break. There is nothing to flush: log_request's queue is drained
    continuously by the background writer thread, not batched."""
    del f


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
