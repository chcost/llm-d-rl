#!/usr/bin/env python3
"""Build a multi-turn *agentic* Search-R1 parquet for verl GRPO.

Unlike make_hotpotqa.py (single-turn, context-in-prompt reading comprehension),
here the prompt is QUESTION-ONLY: the model must decide what to search, call the
`search` tool (served by a Search-R1 retrieval server over the wiki-18 corpus),
read the returned passages, and iterate until it emits a short <answer>. This is
the genuine multi-turn agentic loop.

Reward = normalized exact-match via verl's search_r1_like_qa_em, triggered by
data_source="searchR1_hotpotqa" and reward_model.ground_truth={"target": [answer]}.

The multi-turn ToolAgentLoop needs, per row:
  - prompt: [system, user]  (question only; the tool schema is injected by the
    Qwen3 chat template via tools= at apply_chat_template time, so the user text
    must NOT hardcode a free-text <tool_call> convention - it just asks for
    <think> reasoning, tool use when needed, and a final <answer>).
  - extra_info.need_tools_kwargs = True
  - extra_info.tools_kwargs["search"].create_kwargs = {ground_truth, question, data_source}
    (the key "search" MUST match the tool's function name and the dataset alias).

Source: HotpotQA distractor (question + answer only; distractor context dropped)
so the workload is directly comparable to the single-turn hotpotqa run and its
gold docs live in wiki-18. (Alternative: PeterJinGo/nq_hotpotqa_train.)

Usage: make_searchr1.py [--local_dir DIR] [--max_train N] [--max_test N]
"""
import argparse
import os

import datasets
import pandas as pd

DATA_SOURCE = "searchR1_hotpotqa"

SYSTEM_CONTENT = "You are a helpful and harmless assistant."

# Question-only instruction. Tool-call mechanics (the hermes <tool_call>{...} JSON)
# are handled by the chat template + tools= injection, so we only describe the
# reasoning / when-to-search / answer-format contract here.
USER_PREFIX = (
    "Answer the given question. You must reason inside <think> and </think> every time you "
    "get new information. If you lack knowledge, call the search tool; the top passages will "
    "be returned to you. You may search as many times as needed. When you have enough "
    "information, give the final short answer inside <answer> and </answer> with no extra "
    "words, e.g. <answer> Beijing </answer>.\n\nQuestion: "
)


def _load(split):
    last = None
    for repo in ("hotpotqa/hotpot_qa", "hotpot_qa"):
        try:
            return datasets.load_dataset(repo, "distractor", split=split, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"could not load hotpot_qa distractor {split}: {last}")


def build(split, limit=None):
    ds = _load(split)
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        question = ex["question"]
        answer = ex["answer"]
        ground_truth = {"target": [answer]}
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [
                    {"role": "system", "content": SYSTEM_CONTENT},
                    {"role": "user", "content": USER_PREFIX + question},
                ],
                "ability": "qa",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "index": i,
                    "split": split,
                    "question": question,
                    "answer": answer,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "search": {
                            "create_kwargs": {
                                "ground_truth": ground_truth,
                                "question": question,
                                "data_source": DATA_SOURCE,
                            }
                        }
                    },
                },
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
            # tokenize=False then encode: this tokenizer's apply_chat_template(tokenize=True)
            # path is broken here (returns ~2 ids), while the string render is correct - and
            # tokenize=False is also the path verl's dataset uses for the rollout prompt.
            s = tok.apply_chat_template(list(msgs), add_generation_prompt=True, tokenize=False)
            ids = tok(s, add_special_tokens=False).input_ids
        except Exception:
            ids = tok(msgs[-1]["content"]).input_ids
        lens.append(len(ids))
    lens.sort()
    m = len(lens)
    print(f"[token_stats] prompt tokens (question-only): p50={lens[m // 2]} p90={lens[int(0.9 * m)]} "
          f"max={lens[-1]} min={lens[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="/tmp/verl/data/searchr1")
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
