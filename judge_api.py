"""
LLM-as-a-Judge for EBA/ESMA Regulatory Q&A  (API version)
===========================================================
Same pipeline as judge.py but calls the SwissAI serving API
instead of loading local models — no GPU memory required.

Usage:
    python judge_api.py --n_samples 5

    python judge_api.py \
        --answerer meta-llama/Llama-3.1-70B-Instruct \
        --judge    openai/gpt-oss-120b-evMj \
        --n_samples 50

    python judge_api.py --qa_file output/esma_qa_web.json --n_samples 10
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Model selection ───────────────────────────────────────────────────────────
# Change these two lines to switch models without touching the CLI args.

ANSWERER_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"
JUDGE_MODEL    = "meta-llama/Llama-3.3-70B-Instruct"


# ── Client ────────────────────────────────────────────────────────────────────

_client: OpenAI = OpenAI(
    api_key=os.environ.get("CSCS_SERVING_API"),
    base_url="https://api.swissai.svc.cscs.ch/v1",
)


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Drop messages with None/empty content that some endpoints reject."""
    return [m for m in messages if m.get("content")]


def run_multiturn_inference(model_name: str, messages: list, max_new_tokens: int = 512) -> str:
    """Call the API with a full conversation history and return the response text."""
    response = _client.chat.completions.create(
        model=model_name,
        messages=_sanitize_messages(messages),
        max_tokens=max_new_tokens,
    )
    return response.choices[0].message.content.strip()


def generate(model_name: str, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    messages = [{"role": "user", "content": prompt}]
    response = _client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    msg = response.choices[0].message
    # Some reasoning models (Kimi, DeepSeek-R1) return None for content
    # and put the actual reply in reasoning_content or a custom field.
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        raise ValueError(f"Empty response from {model_name}: {response}")
    return text


# ── Prompts ───────────────────────────────────────────────────────────────────

ANSWERER_PROMPT = """\
You are an expert in EU financial regulation, specializing in EBA and ESMA guidelines.
Answer the following regulatory question accurately and completely.
Where relevant, reference specific articles, guidelines, or regulatory frameworks.

Question: {question}

Answer:"""

JUDGE_PROMPT = """\
You are an expert judge evaluating answers to EU financial regulation questions \
from EBA and ESMA sources.

## Question
{question}

## Official Answer (Ground Truth)
{ground_truth}

## Candidate Answer
{candidate}

## Scoring Rubric
Score each dimension 1-5:
- 1: Completely wrong/missing
- 3: Partially correct/complete
- 5: Fully correct/complete

Dimensions:
- Accuracy: factual correctness vs official answer
- Completeness: coverage of all key points
- Clarity: clear, structured, unambiguous language
- Overall: weighted average (accuracy 50%, completeness 30%, clarity 20%)

## Instructions
1. Write your reasoning FIRST
2. Then assign scores based on your reasoning

Respond in this exact JSON format only:
{{
  "reasoning": "<two sentences evaluating the answer>",
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "clarity": <1-5>,
  "overall": <1-5>
}}"""


# ── Score parsing ─────────────────────────────────────────────────────────────

SCORE_KEYS = {"accuracy", "completeness", "clarity", "overall"}

def parse_scores(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    for block in reversed(re.findall(r"\{[^{}]+\}", text, re.DOTALL)):
        try:
            data = json.loads(block)
            if SCORE_KEYS.issubset(data.keys()):
                return data
        except json.JSONDecodeError:
            continue
    return {"accuracy": None, "completeness": None, "clarity": None,
            "overall": None, "reasoning": text[-300:]}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(qa_items, answerer_model: str, judge_model: str, output_path: str, done_ids: set) -> list[dict]:
    results = []
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    remaining = [item for item in qa_items if item["id"] not in done_ids]
    skipped = len(qa_items) - len(remaining)
    if skipped:
        print(f"Resuming: skipping {skipped} already-processed items.")

    for i, item in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {item['id']}")

        # Step 1: candidate answer
        try:
            candidate = generate(
                answerer_model,
                ANSWERER_PROMPT.format(question=item["question"]),
            )
        except Exception as e:
            print(f"  Answerer error: {e} — skipping item.")
            continue
        print(f"  Candidate: {candidate[:120]}...")

        # Step 2: judge scores
        ground_truth = item.get("final_answer") or item.get("answer", "")
        try:
            judge_out = generate(
                judge_model,
                JUDGE_PROMPT.format(
                    question=item["question"],
                    ground_truth=ground_truth,
                    candidate=candidate,
                ),
                max_new_tokens=1024,
                temperature=0,
            )
            scores = parse_scores(judge_out)
        except Exception as e:
            print(f"  Judge error: {e} — recording null scores.")
            scores = {"accuracy": None, "completeness": None, "clarity": None,
                      "overall": None, "reasoning": str(e)}
        print(f"  Scores → accuracy={scores['accuracy']} completeness={scores['completeness']} "
              f"clarity={scores['clarity']} overall={scores['overall']}")
        print(f"  Reason: {str(scores.get('reasoning', ''))[:100]}")

        result = {
            "id": item["id"],
            "question": item["question"],
            "ground_truth": ground_truth,
            "candidate_answer": candidate,
            "scores": scores,
            "meta": {
                "answerer_model": answerer_model,
                "judge_model": judge_model,
                "status": item.get("status", ""),
                "legal_act": item.get("legal_act", item.get("level1_regulation", "")),
                "topic": item.get("topic", ""),
                "url": item.get("url", ""),
            },
        }
        results.append(result)

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return results


def print_summary(results: list[dict]):
    scored = [r for r in results if r["scores"]["overall"] is not None]
    if not scored:
        print("\nNo valid scores parsed.")
        return

    def avg(key):
        vals = [r["scores"][key] for r in scored if r["scores"][key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 45)
    print(f"SUMMARY  ({len(scored)}/{len(results)} scored)")
    print(f"  Accuracy      {avg('accuracy'):.2f} / 5")
    print(f"  Completeness  {avg('completeness'):.2f} / 5")
    print(f"  Clarity       {avg('clarity'):.2f} / 5")
    print(f"  Overall       {avg('overall'):.2f} / 5")
    print("=" * 45)


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_file(qa_file: str, answerer: str, judge: str, n_samples: int, seed: int) -> list[dict]:
    with open(qa_file) as f:
        all_items = json.load(f)
    answer_key = "final_answer" if "final_answer" in all_items[0] else "answer"
    all_items = [x for x in all_items if x.get("question") and x.get(answer_key)]
    print(f"\nLoaded {len(all_items)} Q&As from {qa_file}")

    random.seed(seed)
    samples = random.sample(all_items, min(n_samples, len(all_items)))

    stem = Path(qa_file).stem.replace("_web", "")
    out_path = Path(f"output/judge_results_{stem}_api.json")

    done_ids: set = set()
    prior_results: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            raw = f.read().strip()
        if raw.startswith("["):
            prior_results = json.loads(raw)
        else:
            prior_results = [json.loads(l) for l in raw.splitlines() if l.strip()]
        done_ids = {r["id"] for r in prior_results}
        print(f"Loaded {len(prior_results)} checkpointed results from {out_path}")

    new_results = run(samples, answerer, judge, str(out_path), done_ids)
    all_results = prior_results + new_results

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Total results: {len(all_results)} ({len(new_results)} new) → {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_files", nargs="+",
                        default=["output/eba_qa_web.json", "output/esma_qa_web.json"],
                        help="One or more Q&A JSON files to process")
    parser.add_argument("--answerer",  default=ANSWERER_MODEL,
                        help="Model name served at the SwissAI endpoint")
    parser.add_argument("--judge",     default=JUDGE_MODEL,
                        help="Judge model name")
    parser.add_argument("--n_samples", type=int, default=9999,
                        help="Samples per file")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    judge_name = args.judge or args.answerer
    print(f"Answerer : {args.answerer}")
    print(f"Judge    : {judge_name}")
    print(f"Endpoint : {_client.base_url}")

    all_results = []
    for qa_file in args.qa_files:
        all_results += run_file(qa_file, args.answerer, judge_name, args.n_samples, args.seed)

    print(f"\n{'='*45}")
    print(f"COMBINED SUMMARY ({len(args.qa_files)} files)")
    print_summary(all_results)


if __name__ == "__main__":
    main()
