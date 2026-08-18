#!/usr/bin/env python3
"""Build a single-turn arXiv subject-area classification parquet for verl GRPO.

Large-input / short-output classification: a full arXiv paper (ccdv/arxiv-classification;
every document > 4k tokens, long tail into the hundreds of thousands) classified into ONE of 11
arXiv subject categories. Output is a single label. Reward = normalized EM via
data_source="searchR1_nq" (the EM reward alias), reward_model.ground_truth={"target": [code, name]}
so EITHER the arXiv category code (e.g. cs.CV) or its descriptive name (e.g. Computer Vision and
Pattern Recognition) scores.

Scientific-domain analog of scotus_xl (legal -> scientific): same large-shared-prefill,
short-label, single-turn shape (no retriever/tool), so it flows through the same path as scotus_xl.

Uses the "no_ref" config, which strips in-document class references (e.g. "[cs.LG]" -> "[]") so the
label cannot leak into the prompt. Do NOT use the default config for this reason.

Usage: make_arxiv.py [--local_dir DIR] [--max_train N] [--max_test N] [--no_cot]
                     [--min_prompt_tokens LO] [--max_prompt_tokens HI]
"""
import argparse
import os

import datasets
import pandas as pd

DATA_SOURCE = "searchR1_nq"  # routes to the normalized-EM reward

SYSTEM_CONTENT = "You are a scientific expert who classifies arXiv papers by subject area."

# ccdv/arxiv-classification ClassLabel names ARE the arXiv category codes (e.g. "cs.CV"); these are
# the standard arXiv subject-classification names for those codes. Used to offer the model a
# human-readable option and to accept either the code or the name as a correct answer.
ARXIV = {
    "math.AC": "Commutative Algebra",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.AI": "Artificial Intelligence",
    "cs.SY": "Systems and Control",
    "math.GR": "Group Theory",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.PL": "Programming Languages",
    "cs.IT": "Information Theory",
    "cs.DS": "Data Structures and Algorithms",
    "cs.NE": "Neural and Evolutionary Computing",
    "math.ST": "Statistics Theory",
}


def _load(split):
    # ccdv/arxiv-classification is a plain Parquet dataset (no loading script), so no
    # trust_remote_code is needed - and datasets>=4 rejects it.
    return datasets.load_dataset("ccdv/arxiv-classification", "no_ref", split=split)


def build(split, limit=None, no_cot=False):
    ds = _load(split)
    codes = ds.features["label"].names  # arXiv category codes, e.g. ["math.AC", "cs.CV", ...]
    options = "\n".join(f"{c}: {ARXIV.get(c, c)}" for c in codes)
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        code = codes[ex["label"]]
        name = ARXIV.get(code, code)
        if no_cot:
            # Direct single-label classification (the standard formulation): no CoT.
            # /no_think = Qwen3 soft switch to skip the <think> block (it reasons by default), so the
            # model emits only the label and output is naturally short.
            user = (
                "/no_think\n"
                "Read the following arXiv paper and classify it into exactly ONE subject area from "
                "the list below. Answer with ONLY the chosen subject area (its arXiv category code "
                "or its name) inside <answer> and </answer>, e.g. <answer> cs.CV </answer> or "
                "<answer> Computer Vision and Pattern Recognition </answer>. Do not explain.\n\n"
                f"Subject areas:\n{options}\n\nPaper:\n{ex['text']}"
            )
        else:
            user = (
                "Read the following arXiv paper and classify it into exactly ONE subject area from "
                "the list below. Reason briefly inside <think> and </think>, then give ONLY the "
                "chosen subject area (its arXiv category code or its name) inside <answer> and "
                "</answer>, e.g. <answer> cs.CV </answer> or <answer> Computer Vision and Pattern "
                "Recognition </answer>.\n\n"
                f"Subject areas:\n{options}\n\nPaper:\n{ex['text']}"
            )
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [
                    {"role": "system", "content": SYSTEM_CONTENT},
                    {"role": "user", "content": user},
                ],
                "ability": "classification",
                "reward_model": {"style": "rule", "ground_truth": {"target": [code, name]}},
                "extra_info": {"index": i, "split": split, "code": code, "name": name},
            }
        )
    return pd.DataFrame(rows)


def _prompt_len(tok, msgs):
    """Prompt token count as verl sees it (chat template + generation prompt)."""
    try:
        s = tok.apply_chat_template(list(msgs), add_generation_prompt=True, tokenize=False)
        return len(tok(s, add_special_tokens=False).input_ids)
    except Exception:
        return len(tok(msgs[-1]["content"]).input_ids)


def filter_band(df, lo, hi, model="/tmp/verl/models/Qwen3-4B"):
    """Keep only rows whose prompt token length is in [lo, hi]. lo/hi<=0 disable that bound.
    Drops short papers and the extreme tail that busts training memory."""
    if lo <= 0 and hi <= 0:
        return df
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    keep, n0 = [], len(df)
    for i in range(n0):
        L = _prompt_len(tok, df.iloc[i]["prompt"])
        if (lo <= 0 or L >= lo) and (hi <= 0 or L <= hi):
            keep.append(i)
    out = df.iloc[keep].reset_index(drop=True)
    print(f"[filter_band] lo={lo} hi={hi}: kept {len(out)}/{n0} ({100*len(out)/max(n0,1):.1f}%)")
    return out


def token_stats(df, model="/tmp/verl/models/Qwen3-4B", n=300, over=24576):
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
            s = tok.apply_chat_template(list(msgs), add_generation_prompt=True, tokenize=False)
            ids = tok(s, add_special_tokens=False).input_ids
        except Exception:
            ids = tok(msgs[-1]["content"]).input_ids
        lens.append(len(ids))
    lens.sort()
    m = len(lens)
    n_over = sum(1 for x in lens if x > over)
    print(f"[token_stats] prompt tokens: p50={lens[m // 2]} p90={lens[int(0.9 * m)]} "
          f"p99={lens[int(0.99 * m)]} max={lens[-1]} min={lens[0]}  (>{over}: {n_over}/{m})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="/tmp/verl/data/arxiv")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--no_cot", action="store_true", help="direct single-label output, no <think> CoT")
    ap.add_argument("--min_prompt_tokens", type=int, default=0,
                    help="drop papers shorter than this many prompt tokens (0 = no lower bound)")
    ap.add_argument("--max_prompt_tokens", type=int, default=0,
                    help="drop papers longer than this many prompt tokens (0 = no upper bound)")
    args = ap.parse_args()
    os.makedirs(args.local_dir, exist_ok=True)
    for split, out, lim in (("train", "train.parquet", args.max_train),
                            ("test", "test.parquet", args.max_test)):
        df = build(split, lim or None, no_cot=args.no_cot)
        df = filter_band(df, args.min_prompt_tokens, args.max_prompt_tokens)
        p = os.path.join(args.local_dir, out)
        df.to_parquet(p, index=False)
        print(f"{split}: wrote {len(df)} rows -> {p}")
        if split == "train":
            token_stats(df)
