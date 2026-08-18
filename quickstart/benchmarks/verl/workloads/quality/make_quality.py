#!/usr/bin/env python3
"""Build a single-turn, article-in-prompt QuALITY parquet for verl GRPO.

Real, convergent, large-input / short-verifiable-output RLVR task (long-context multiple-choice
reading comprehension). QuALITY articles are ~6.8k tokens (p50), the answer is a single option
letter, and reasoning is kept ON (<think>).

Prompt layout is ARTICLE-FIRST: [short global instruction][ARTICLE][question][options], so the
article is the long leading prefix shared by every question about that article. Rows are SORTED BY
ARTICLE and the run sets data.shuffle=False (see the quality task.env) so all questions of an
article land in the same rollout step.

Reward: reuse verl's search_r1_like_qa_em EM via data_source="searchR1_nq" (a reward-routing alias -
no NQ data is involved; it just routes <answer>...</answer> to normalized EM). ground_truth
target=[correct letter]; the model is instructed to emit ONLY the letter, e.g. <answer> B </answer>.
The val metric therefore appears as val-core/searchR1_nq/acc (== QuALITY accuracy).

Source: emozilla/quality (train 2,523 / validation 2,086; 150 articles, ~16.8 questions/article;
answer is a 0-indexed int 0-3 over 4 options).

Usage: make_quality.py [--local_dir DIR] [--max_train N] [--max_test N]
"""
import argparse
import os

import datasets
import pandas as pd

INSTRUCTION = (
    "Read the article and answer the multiple-choice question. Reason step by step inside <think> "
    "and </think>, then give ONLY the letter of the correct option inside <answer> and </answer>, "
    "for example: <answer> B </answer>.\n\n"
)


def _options(ex):
    opts = ex["options"]
    if not isinstance(opts, list):
        opts = list(opts)
    return opts


def _letter(i):
    return chr(ord("A") + i)


def build(split, limit=None):
    ds = datasets.load_dataset("emozilla/quality", split=split)
    rows = []
    for i, ex in enumerate(ds):
        opts = _options(ex)
        ans = int(ex["answer"])          # 0-indexed over the 4 options
        letters = [_letter(k) for k in range(len(opts))]
        optblock = "\n".join(f"{l}) {o}" for l, o in zip(letters, opts))
        user = (
            INSTRUCTION
            + "Article:\n"
            + ex["article"]
            + f"\n\nQuestion: {ex['question']}"
            + "\n\nOptions:\n"
            + optblock
        )
        rows.append(
            {
                "data_source": "searchR1_nq",           # reward-routing alias -> search_r1_like_qa_em EM
                "prompt": [{"role": "user", "content": user}],
                "ability": "qa",
                "reward_model": {"style": "rule", "ground_truth": {"target": [letters[ans]]}},
                "article": ex["article"],
                "extra_info": {"question": ex["question"], "answer_letter": letters[ans],
                               "hard": bool(ex.get("hard", False)), "split": split},
            }
        )
    df = pd.DataFrame(rows)
    # Group each article's questions contiguously so that, with data.shuffle=False, all ~17 questions
    # of an article land in the same rollout step. We RANDOMIZE the article ORDER (fixed seed) rather
    # than the dataset order, so each 256-prompt batch is a random mix of articles instead of
    # dataset-adjacent ones (GRPO's advantage is per-group, so batch composition does not change the
    # learning signal).
    import random as _random
    first = {}
    for a in df["article"]:
        if a not in first:
            first[a] = len(first)
    perm = list(range(len(first)))
    _random.Random(1234).shuffle(perm)
    remap = {a: perm[i] for a, i in first.items()}
    df["article_id"] = [remap[a] for a in df["article"]]
    df = df.sort_values("article_id", kind="stable").reset_index(drop=True)
    # article_id is metadata; fold it into extra_info and drop the bulky article column.
    df["extra_info"] = [dict(ei, article_id=int(a)) for ei, a in zip(df["extra_info"], df["article_id"])]
    df = df.drop(columns=["article", "article_id"])
    if limit:
        df = df.head(limit).reset_index(drop=True)
    return df


def token_stats(df, model="/tmp/verl/models/Qwen3-4B", n=300):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as e:  # noqa: BLE001
        print(f"[token_stats] tokenizer unavailable ({e}); skipping")
        return
    lens = [len(tok(df.iloc[i]["prompt"][0]["content"], add_special_tokens=False).input_ids)
            for i in range(min(len(df), n))]
    lens.sort(); m = len(lens)
    print(f"[token_stats] prompt tokens: p50={lens[m // 2]} p90={lens[int(0.9 * m)]} max={lens[-1]} min={lens[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="/tmp/verl/data/quality")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_test", type=int, default=512)   # cap val to keep validation gen/memory sane
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
