#!/usr/bin/env python3
"""Build a single-turn, context-in-prompt MuSiQue parquet for verl GRPO.

Real, convergent, deterministic long-prompt/short-answer RLVR task (multi-hop reading
comprehension, NOT agentic search). Same shape as make_hotpotqa.py but with a longer prompt:
MuSiQue ships ~20 candidate paragraphs per example (vs HotpotQA's 10), so prompts run ~2x
longer while the answer stays short.

Each example's prompt embeds the MuSiQue paragraphs + the question; the model reasons inside
<think></think> and emits a short answer inside <answer></answer>. Reward = normalized exact-match via verl's
search_r1_like_qa_em, triggered by data_source="searchR1_musique" (already registered in
verl's reward_score/__init__.py) and reward_model.ground_truth={"target": [answer, *answer_aliases]}.

Source: dgslibisey/MuSiQue (the answerable subset; 19,938 train / 2,417 validation).

Usage: make_musique.py [--local_dir DIR] [--max_train N] [--max_test N]
"""
import argparse
import os

import datasets
import pandas as pd

# Reason step by step inside <think></think>, then give the final short answer inside <answer>.
INSTRUCTION = (
    "You are given several reference paragraphs and a question. Read the paragraphs, reason step "
    "by step inside <think> and </think>, then give the final short answer inside <answer> and "
    "</answer> with no extra words. For example: <answer> Beijing </answer>.\n\n"
)


def _load(split):
    # dgslibisey/MuSiQue is a plain parquet dataset (answerable subset); splits: train, validation.
    return datasets.load_dataset("dgslibisey/MuSiQue", split=split)


def _context_block(paragraphs):
    parts = []
    for p in paragraphs:
        parts.append(f"Title: {p['title']}\n{p['paragraph_text']}")
    return "\n\n".join(parts)


def _targets(ex):
    tgt = [ex["answer"]]
    aliases = ex.get("answer_aliases") or []
    for a in aliases:
        if a and a not in tgt:
            tgt.append(a)
    return tgt


def build(split, limit=None):
    ds = _load(split)
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        user = (
            INSTRUCTION
            + "Reference paragraphs:\n"
            + _context_block(ex["paragraphs"])
            + f"\n\nQuestion: {ex['question']}"
        )
        rows.append(
            {
                "data_source": "searchR1_musique",
                "prompt": [{"role": "user", "content": user}],
                "ability": "qa",
                "reward_model": {"style": "rule", "ground_truth": {"target": _targets(ex)}},
                "extra_info": {"index": i, "question": ex["question"], "answer": ex["answer"], "split": split},
            }
        )
    return pd.DataFrame(rows)


def token_stats(df, model="/tmp/verl/models/Qwen3-4B", n=300):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as e:  # noqa: BLE001
        print(f"[token_stats] tokenizer unavailable ({e}); skipping")
        return
    lens = []
    for i in range(min(len(df), n)):
        msgs = df.iloc[i]["prompt"]
        try:
            ids = tok.apply_chat_template(list(msgs), add_generation_prompt=True, tokenize=True)
        except Exception:
            ids = tok(msgs[0]["content"]).input_ids
        lens.append(len(ids))
    lens.sort()
    m = len(lens)
    print(f"[token_stats] prompt tokens: p50={lens[m // 2]} p90={lens[int(0.9 * m)]} max={lens[-1]} min={lens[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="/tmp/verl/data/musique")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_test", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.local_dir, exist_ok=True)
    for split, out, lim in (("train", "train.parquet", args.max_train),
                            ("validation", "test.parquet", args.max_test)):
        df = build(split, lim or None)
        p = os.path.join(args.local_dir, out)
        df.to_parquet(p, index=False)
        print(f"{split}: wrote {len(df)} rows -> {p}")
        if split == "train":
            token_stats(df)
