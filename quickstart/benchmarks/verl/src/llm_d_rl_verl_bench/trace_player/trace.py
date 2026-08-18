"""Normalized trace schema + loader for the router-agnostic trace player.

A trace is a list of conversations; each conversation is a list of turns with
input/output token counts and the off-GPU gap that precedes the turn. There is
no prompt text (the weka CC-traces carry only token counts + timing), so the
player synthesizes deterministic prefix-nesting dummy token ids per conversation
and forces the output length. See benchmarks/workloads/weka/make_weka.py for the
producer and README for the schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass
class Turn:
    input_tokens: int
    output_tokens: int
    pre_gap_s: float = 0.0


@dataclass
class ConversationTrace:
    conv_id: str
    turns: list[Turn]

    @staticmethod
    def from_obj(obj: dict) -> "ConversationTrace":
        return ConversationTrace(
            conv_id=str(obj["conv_id"]),
            turns=[
                Turn(
                    input_tokens=int(t["input_tokens"]),
                    output_tokens=int(t["output_tokens"]),
                    pre_gap_s=float(t.get("pre_gap_s", 0.0)),
                )
                for t in obj["turns"]
            ],
        )


def parse_trace_turns(raw) -> ConversationTrace:
    """Accept a JSON string, a dict, or a bare list-of-turns.

    verl passes each ``extra_info`` cell straight through from the parquet, and a
    JSON string round-trips through parquet more robustly than a nested list, so
    make_weka.py stores ``trace_turns`` as a JSON string. Numpy scalar strings
    from non_tensor_batch are handled via ``str()``.
    """
    if raw is None:
        raise ValueError("trace_player: missing 'trace_turns'")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode()
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, list):  # bare list of turns
        raw = {"conv_id": "conv", "turns": raw}
    return ConversationTrace.from_obj(raw)


def _stable_seed(conv_id: str) -> int:
    """Process-stable 64-bit seed from a conversation id.

    IMPORTANT: Python's builtin ``hash()`` is salted per-process
    (PYTHONHASHSEED), so it would produce different token ids for the same
    conversation on different Ray workers, destroying prefix-cache reuse. Use a
    stable digest instead so identical conv_ids map to identical token ids on
    every worker.
    """
    return int.from_bytes(hashlib.blake2b(conv_id.encode(), digest_size=8).digest(), "big")


def nested_prompt_ids(conv_id: str, n: int, vocab_lo: int = 10, vocab_hi: int = 30000) -> list[int]:
    """Deterministic token ids of length ``n`` for a conversation turn.

    Turn k's prompt is a prefix of any longer turn of the same conversation
    (same seed, take-first-n => prefixes nest), which reproduces the append-only
    growing-prefix structure of a real multi-turn conversation and therefore real
    vLLM prefix-cache hits and real EPP prefix-scorer behaviour. Ids are kept in
    ``[vocab_lo, vocab_hi)`` to stay well inside any tokenizer vocab and avoid
    special-token ids.

    A small linear-congruential generator is used (not ``random.Random``) so the
    sequence is cheap and reproducible without allocating the whole stream up
    front for very long prompts.
    """
    n = max(1, int(n))
    span = vocab_hi - vocab_lo
    state = _stable_seed(conv_id) or 1
    out = []
    for _ in range(n):
        # 64-bit LCG (Knuth MMIX constants); high bits are the well-mixed ones.
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append(vocab_lo + (state >> 33) % span)
    return out


def load_traces_jsonl(path: str) -> list[ConversationTrace]:
    """Load a normalized trace-player JSONL (one conversation per line)."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(ConversationTrace.from_obj(json.loads(line)))
    return out
