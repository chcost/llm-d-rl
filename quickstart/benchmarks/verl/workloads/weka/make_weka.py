#!/usr/bin/env python3
"""Build a trace-player parquet from the Weka CC-traces - self-contained.

Reads the raw traces.jsonl (download separately; see README.md), selects N
conversations deterministically, linearizes each into a turn schedule (per-turn
input/output token counts + the off-GPU gap that precedes the turn), and writes
one parquet row per conversation for the ``trace_player`` agent loop. No
dependency on the simulator code under ~/work; the minimal trace parsing is
reimplemented here so this runs on the training pod.

Raw schema (one JSON object per line = one session), per analysis/dataset.py:
  {"requests": [ {t, in, out, ttft, api_time, type, ...}
               | {type: "subagent", t, requests: [{t, in, out, api_time}, ...]} ]}
Conversations have no stable id field, so the trace line index is used as
conv_id (matches preselected_groups.json, whose group lists are line indices).
Gap before a call = max(0, t - prev.obs_end), where obs_end = prev.t + prev.api_time.

Output row schema (consumed by trace_player.trace.parse_trace_turns):
  data_source, prompt (tiny dummy - trace_player builds its own prompt_ids),
  ability, reward_model (dummy), extra_info={"index","conv_id","num_turns",
  "trace_turns": <JSON string of [{input_tokens,output_tokens,pre_gap_s}, ...]>}
"""

from __future__ import annotations

import argparse
import json
import os
import random

import pandas as pd

DATA_SOURCE = "weka_cc_traces"


def _iter_sessions(path):
    with open(path) as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if line:
                yield lineno, json.loads(line)


def _session_max_input(sess) -> int:
    mx = 0
    for r in sess.get("requests", []):
        if r.get("type") == "subagent":
            for inner in r.get("requests", []):
                mx = max(mx, int(inner.get("in", 0)))
        else:
            mx = max(mx, int(r.get("in", 0)))
    return mx


def _collect_calls(sess, include_subagents):
    """Return [(t, in_tok, out_tok, api_time)] for main (and optionally subagent) calls."""
    calls = []
    for r in sess.get("requests", []):
        if r.get("type") == "subagent":
            if not include_subagents:
                continue
            for inner in r.get("requests", []):
                calls.append((float(inner.get("t", 0.0)), int(inner.get("in", 0)),
                              int(inner.get("out", 0)), float(inner.get("api_time", 0.0))))
        else:
            calls.append((float(r.get("t", 0.0)), int(r.get("in", 0)),
                          int(r.get("out", 0)), float(r.get("api_time", 0.0))))
    calls.sort(key=lambda c: c[0])  # by start time (flattens any subagent concurrency)
    return calls


def _linearize(sess, include_subagents, gap_cap, cap_in=0, cap_out=0, max_turns_per_conv=0):
    """Linearize a session into turns.

    cap_in (0 = no cap) STOPS the conversation at the first turn whose natural
    input would exceed the cap (that turn and everything after it is dropped),
    rather than clamping it. Real weka conversations grow context every turn;
    clamping a repeated over-cap value would make nested_prompt_ids(conv_id, n)
    emit the SAME token sequence for every clamped turn (it's deterministic in
    (conv_id, n)), collapsing the tail into an artificially-repeated prompt and
    trivializing/understating cache-affinity effects. Stopping instead preserves
    genuine turn-by-turn growth (and thus real, escalating re-prefill cost) for
    every turn that IS replayed - the conversation is just shorter, as if it hit
    the served model's context/KV budget partway through.
    cap_out (0 = no cap) still CLAMPS each turn's output token count - output is
    freshly generated per turn (not from nested_prompt_ids), so clamping it does
    not create a repeated-content artifact.
    """
    turns = []
    prev_end = None
    for (t, in_tok, out_tok, api) in _collect_calls(sess, include_subagents):
        in_tok = max(1, in_tok)
        out_tok = max(1, out_tok)
        if cap_in and in_tok > cap_in:
            break
        if cap_out:
            out_tok = min(out_tok, cap_out)
        gap = 0.0
        if prev_end is not None:
            gap = max(0.0, t - prev_end)
            if gap_cap is not None:
                gap = min(gap, gap_cap)
        turns.append({"input_tokens": in_tok, "output_tokens": out_tok, "pre_gap_s": round(gap, 4)})
        prev_end = t + api
    if max_turns_per_conv and len(turns) > max_turns_per_conv:
        # Truncate to the first K turns. Weka sessions average ~145 main turns, so
        # replaying them whole is enormous; truncation bounds rollout while keeping
        # the growing-prefix multi-turn structure and letting N scale over many convs.
        turns = turns[:max_turns_per_conv]
    return turns


def _n_main_turns(sess) -> int:
    return sum(1 for r in sess.get("requests", []) if r.get("type") != "subagent")


def _select_indices(path, n, threshold, seed, groups_file, max_turns=0,
                     cap_in=0, cap_out=0, gap_cap=None, include_subagents=False):
    if groups_file:
        with open(groups_file) as fh:
            groups = json.load(fh)["groups"]
        if str(n) not in groups:
            raise SystemExit(f"groups_file has no N={n}; available keys={list(groups)}")
        return list(groups[str(n)])
    if cap_in or cap_out:
        # Eligibility must match what _linearize will actually produce under the
        # cap, or a conversation whose first turn already exceeds cap_in silently
        # drops out of build_rows later (it returns no turns), shrinking the
        # dataset below N unpredictably. Re-run the same truncation here.
        eligible = [lineno for lineno, sess in _iter_sessions(path)
                    if _linearize(sess, include_subagents, gap_cap, cap_in, cap_out, max_turns)]
    else:
        eligible = [lineno for lineno, sess in _iter_sessions(path)
                    if _session_max_input(sess) < threshold
                    and (not max_turns or 0 < _n_main_turns(sess) <= max_turns)]
    random.Random(seed).shuffle(eligible)
    if n > len(eligible):
        raise SystemExit(f"requested N={n} > eligible pool {len(eligible)} (raise --threshold/--max-turns or lower --n)")
    return eligible[:n]


def build_rows(path, indices, include_subagents, gap_cap, max_total_tokens, cap_in=0, cap_out=0, max_turns_per_conv=0):
    want = set(indices)
    by_line = {}
    for lineno, sess in _iter_sessions(path):
        if lineno in want:
            by_line[lineno] = sess
            if len(by_line) == len(want):
                break
    rows, skipped = [], 0
    for i, lineno in enumerate(indices):
        sess = by_line.get(lineno)
        if sess is None:
            continue
        conv_id = str(lineno)
        turns = _linearize(sess, include_subagents, gap_cap, cap_in, cap_out, max_turns_per_conv)
        if not turns:
            continue
        if max_total_tokens:
            worst = max(t["input_tokens"] + t["output_tokens"] for t in turns)
            if worst > max_total_tokens:
                skipped += 1
                continue
        rows.append({
            "data_source": DATA_SOURCE,
            # trace_player builds its own prompt_ids from conv_id; a tiny dummy
            # prompt keeps verl's dataset/tokenizer path happy.
            "prompt": [{"role": "user", "content": "x"}],
            "ability": "trace-replay",
            "reward_model": {"style": "rule", "ground_truth": {"target": ["0"]}},
            "extra_info": {
                "index": i,
                "conv_id": conv_id,
                "num_turns": len(turns),
                # JSON string => robust parquet round-trip; parse_trace_turns accepts it.
                # Embed conv_id so each conversation gets DISTINCT synthetic token ids
                # (and thus distinct prefix-cache behaviour); a bare turns list would
                # make every conversation share the default conv_id "conv".
                "trace_turns": json.dumps({"conv_id": conv_id, "turns": turns}),
            },
        })
    if skipped:
        print(f"note: skipped {skipped} conversations whose worst turn exceeds --max-total-tokens")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build a trace-player parquet from the Weka CC-traces.")
    ap.add_argument("--traces", required=True, help="Path to traces.jsonl (see README for the HuggingFace download).")
    ap.add_argument("--local_dir", default="/tmp/verl/data/weka")
    ap.add_argument("--n", type=int, default=2, help="Number of conversations (smoke test: 1-2).")
    ap.add_argument("--threshold", type=int, default=262144, help="Standalone selection: drop convs with any request input >= this.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups-file", default=None,
                    help="Optional preselected_groups.json for parity with the simulator (group key = N; values are line indices).")
    ap.add_argument("--include-subagents", action="store_true",
                    help="Flatten subagent inner calls into the serial stream by start time (default: main chain only).")
    ap.add_argument("--gap-cap", type=float, default=10.0, help="Cap per-turn off-GPU gap seconds (sim GAP=10). Also gated at runtime by VERL_TRACE_GAP_CAP_S.")
    ap.add_argument("--no-gap-cap", action="store_true", help="Disable the gap cap at build time (GAP=inf).")
    ap.add_argument("--cap-input-tokens", type=int, default=0,
                    help="STOP the conversation at the first turn whose input would exceed this many tokens "
                         "(fit the served model's context/KV budget); that turn and later ones are dropped, "
                         "not clamped, so replayed turns keep their real distinct sizes. 0 = no cap.")
    ap.add_argument("--cap-output-tokens", type=int, default=0,
                    help="TRUNCATE each turn's output to this many tokens. 0 = no cap.")
    ap.add_argument("--max-turns", type=int, default=0,
                    help="Standalone selection: only pick conversations with <= this many main turns (0 = no limit).")
    ap.add_argument("--max-turns-per-conv", type=int, default=0,
                    help="TRUNCATE each conversation to its first K turns (0 = keep all). Weka sessions "
                         "average ~145 turns; truncating bounds rollout and lets N scale over many convs.")
    ap.add_argument("--max-total-tokens", type=int, default=0,
                    help="Skip convs whose worst turn input+output exceeds this AFTER capping (0 disables; prefer --cap-*-tokens).")
    args = ap.parse_args()

    gap_cap = None if args.no_gap_cap else args.gap_cap
    indices = _select_indices(args.traces, args.n, args.threshold, args.seed, args.groups_file, args.max_turns,
                              args.cap_input_tokens, args.cap_output_tokens, gap_cap, args.include_subagents)
    rows = build_rows(args.traces, indices, args.include_subagents, gap_cap,
                      args.max_total_tokens or None, args.cap_input_tokens, args.cap_output_tokens,
                      args.max_turns_per_conv)
    if not rows:
        raise SystemExit("No rows produced; loosen --max-total-tokens / --threshold or check --traces path.")

    os.makedirs(args.local_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    train_p = os.path.join(args.local_dir, "train.parquet")
    test_p = os.path.join(args.local_dir, "test.parquet")
    df.to_parquet(train_p, index=False)
    df.to_parquet(test_p, index=False)  # tiny; reuse for the val split
    tot = sum(r["extra_info"]["num_turns"] for r in rows)
    print(f"wrote {len(rows)} conversations ({tot} turns) -> {train_p}")
    print("conv_ids (trace line indices):", [r["extra_info"]["conv_id"] for r in rows])


if __name__ == "__main__":
    main()
