"""
LLM-as-a-Judge for EBA/ESMA Regulatory Q&A
============================================
Pipeline:
  1. Answerer LLM  — generates a candidate answer to a regulatory question
  2. Judge LLM     — scores the candidate against the official ground-truth answer

Available models (loaded from $SCRATCH/models/):
    Llama-3.1-8B-Instruct
    gemma-4-E4B-it
    Qwen3.5-4B
    Qwen3.5-9B

Usage:
    python judge.py --n_samples 5 --same_model

    python judge.py \
        --answerer Llama-3.1-8B-Instruct \
        --judge    Qwen3.5-9B \
        --n_samples 50 --load_in_4bit

    python judge.py --qa_file output/esma_qa_web.json --same_model --n_samples 10
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ── Prompts ───────────────────────────────────────────────────────────────────

ANSWERER_PROMPT = """\
You are an expert in EU financial regulation. Answer the following regulatory \
question concisely and precisely.

Question: {question}

Provide your answer directly without preamble."""

JUDGE_PROMPT = """\
You are an expert judge evaluating answers to EU financial regulation questions.

## Question
{question}

## Official Answer (Ground Truth)
{ground_truth}

## Candidate Answer
{candidate}

## Task
Score the candidate answer on three dimensions (each 1-5):
1. Accuracy     — factual correctness compared to the official answer
2. Completeness — coverage of all key points in the official answer
3. Clarity      — clear, well-structured, unambiguous language

Then give an Overall score (1-5) reflecting overall quality.

Respond in this exact JSON format only, no extra text:
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "clarity": <1-5>,
  "overall": <1-5>,
  "reasoning": "<one sentence explaining the overall score>"
}}"""


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_name: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH environment variable is not set.")

    model_path = os.path.join(scratch, "models", model_name)
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    qconfig = None
    if load_in_4bit:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        qconfig = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True,
        torch_dtype=torch.float16, device_map="auto",
        quantization_config=qconfig,
    )
    model.eval()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {used:.1f} / {total:.1f} GB")

    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        # enable_thinking=False disables Qwen3's chain-of-thought output
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for models that don't support enable_thinking
        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            input_text = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip any residual <think>...</think> block (some models leak it)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# ── Score parsing ─────────────────────────────────────────────────────────────

SCORE_KEYS = {"accuracy", "completeness", "clarity", "overall"}

def parse_scores(text: str) -> dict:
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find all JSON-like blocks and return the last one that has the score keys
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

def run(qa_items, answerer, answerer_tok, judge, judge_tok, output_path: str, done_ids: set) -> list[dict]:
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
        candidate = generate(
            answerer, answerer_tok,
            ANSWERER_PROMPT.format(question=item["question"]),
        )
        print(f"  Candidate: {candidate[:120]}...")

        # Step 2: judge scores
        ground_truth = item.get("final_answer") or item.get("answer", "")
        judge_out = generate(
            judge, judge_tok,
            JUDGE_PROMPT.format(
                question=item["question"],
                ground_truth=ground_truth,
                candidate=candidate,
            ),
            max_new_tokens=256,
            temperature=0,
        )
        scores = parse_scores(judge_out)
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
                "status": item.get("status", ""),
                "legal_act": item.get("legal_act", item.get("level1_regulation", "")),
                "topic": item.get("topic", ""),
                "url": item.get("url", ""),
            },
        }
        results.append(result)

        # Append checkpoint to disk after each item
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file",    default="output/eba_qa_web.json")
    parser.add_argument("--answerer",   default="Llama-3.1-8B-Instruct",
                        help="Model name under $SCRATCH/models/")
    parser.add_argument("--judge",      default=None,
                        help="Judge model name (defaults to --answerer)")
    parser.add_argument("--same_model", action="store_true",
                        help="Use one loaded model for both roles (saves memory)")
    parser.add_argument("--output",     default="output/judge_results.json")
    parser.add_argument("--n_samples",  type=int, default=10)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()

    # Load Q&A data
    with open(args.qa_file) as f:
        all_items = json.load(f)
    answer_key = "final_answer" if "final_answer" in all_items[0] else "answer"
    all_items = [x for x in all_items if x.get("question") and x.get(answer_key)]
    print(f"Loaded {len(all_items)} Q&As from {args.qa_file}")

    random.seed(args.seed)
    samples = random.sample(all_items, min(args.n_samples, len(all_items)))

    # Load models
    answerer_model, answerer_tok = load_model(args.answerer, args.load_in_4bit, args.load_in_8bit)

    judge_name = args.judge or args.answerer
    if args.same_model or judge_name == args.answerer:
        judge_model, judge_tok = answerer_model, answerer_tok
        print("Using same model for answerer and judge.")
    else:
        judge_model, judge_tok = load_model(judge_name, args.load_in_4bit, args.load_in_8bit)

    # Load any previously checkpointed results
    out_path = Path(args.output)
    done_ids: set = set()
    prior_results: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    prior_results.append(rec)
                    done_ids.add(rec["id"])
        print(f"Loaded {len(prior_results)} checkpointed results from {args.output}")

    new_results = run(samples, answerer_model, answerer_tok, judge_model, judge_tok,
                      args.output, done_ids)

    all_results = prior_results + new_results
    print(f"\nTotal results: {len(all_results)} ({len(new_results)} new) → {args.output}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
