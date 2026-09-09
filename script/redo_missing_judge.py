"""
Re-run a single judge model, only for rows/sides where its scores are null,
without touching the other judges already recorded in eval_judge_scores.jsonl.

Usage
-----
    python script/redo_missing_judge.py output/eval_hallucinate_citation --judge_model Qwen/Qwen3.5-27B
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sympy import python

from openai import OpenAI

from eval_dpo import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DIMS,
    JUDGE_MAX_TOKENS,
    JUDGE_PROMPT,
    call_judge,
    canonical_judge,
    load_jsonl,
    parse_scores,
)


def judge_with_retry(client: OpenAI, model: str, prompt: str, max_tokens: int, max_attempts: int = 3) -> dict:
    """call_judge already retries once on a fully empty response (reasoning models
    burning their budget on hidden CoT). But a response can also be non-empty and
    still get its JSON object truncated mid-object, which parse_scores can't
    recover from — that case doesn't raise, so call_judge never sees it. Retry
    here with a doubled budget each time parse_scores comes back all-null."""
    sc = {k: None for k in DIMS}
    for attempt in range(max_attempts):
        try:
            raw = call_judge(client, model, prompt, max_tokens)
            sc = parse_scores(raw)
        except Exception as e:
            print(f"    attempt {attempt + 1} error: {e}")
            sc = {k: None for k in DIMS}
        if any(v is not None for v in sc.values()):
            return sc
        max_tokens *= 2
    return sc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--judge_model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--judge_api_base", default="https://api.swissai.svc.cscs.ch/v1")
    args = parser.parse_args()

    answers_file = args.output_dir / "eval_answers.jsonl"
    scores_file = args.output_dir / "eval_judge_scores.jsonl"
    answers = {r["question_id"]: r for r in load_jsonl(answers_file)}
    scored = load_jsonl(scores_file)

    client = OpenAI(
        api_key=os.environ.get("CSCS_SERVING_API"),
        base_url=args.judge_api_base,
    )
    max_tokens = JUDGE_MAX_TOKENS.get(args.judge_model, DEFAULT_JUDGE_MAX_TOKENS)

    n_redone = 0
    for row in scored:
        qid = row["question_id"]
        answer_row = answers.get(qid)
        if answer_row is None:
            print(f"  Skipping {qid}: not found in {answers_file}")
            continue
        meta = answer_row.get("meta", {})

        canon = canonical_judge(args.judge_model)
        for model_key in ("baseline", "dpo"):
            side = row["scores"].setdefault(model_key, {})
            # Find any existing entry for this judge, under any historical alias.
            alias_keys = [k for k in side if canonical_judge(k) == canon]
            sc = side[alias_keys[0]] if alias_keys else None
            if sc is not None and all(sc.get(k) is not None for k in DIMS):
                continue  # already scored, leave as-is

            candidate = answer_row.get(model_key, {}).get("text", "")
            if not candidate:
                continue

            prompt = JUDGE_PROMPT.format(
                question=answer_row["question"],
                topic=meta.get("topic", ""),
                subject_matter=meta.get("subject_matter", ""),
                ground_truth=answer_row["ground_truth"],
                candidate=candidate,
            )
            new_sc = judge_with_retry(client, args.judge_model, prompt, max_tokens)

            # Consolidate onto the canonical key so a row never carries two
            # entries (old-suffix + new-suffix) for what is the same judge.
            for k in alias_keys:
                del side[k]
            side[canon] = new_sc
            n_redone += 1
            print(f"  Redid {qid} [{model_key}]: {new_sc}")

    with scores_file.open("w") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nRe-judged {n_redone} (row, side) entries for {args.judge_model} → {scores_file}")


if __name__ == "__main__":
    main()




# python script/redo_missing_judge.py output/llama/eval_sft --judge_model "CSCS-Inference/google/gemma-4-31B-it"
# python script/redo_missing_judge.py output/llama/eval_sft --judge_model "RCP-AIaaS/zai-org/GLM-5.3-Flash"

# python script/redo_missing_judge.py output/llama/eval_dr_dpo --judge_model "RCP-AIaaS/Qwen/Qwen3.6-35B-A3B"
# python script/redo_missing_judge.py output/llama/eval_dr_dpo --judge_model "RCP-AIaaS/zai-org/GLM-5.3-Flash"

# python script/redo_missing_judge.py output/llama/eval_gr_dpo --judge_model "RCP-AIaaS/zai-org/GLM-5.3-Flash"
# python script/redo_missing_judge.py output/llama/eval_law_swap --judge_model "RCP-AIaaS/zai-org/GLM-5.3-Flash"
# python script/redo_missing_judge.py output/llama/eval_hallucinate_text --judge_model "RCP-AIaaS/zai-org/GLM-5.3-Flash"
# python script/redo_missing_judge.py output/llama/eval_hallucinate_citation --judge_model "RCP-AIaaS/zai-org/GLM-5.3-Flash"
# python script/redo_missing_judge.py output/llama/eval_article_swap --judge_model "RCP-AIaaS/zai-org/GLM-5.3-Flash"
