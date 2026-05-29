"""
EU Financial Regulation Benchmark — Tier 1 Question Generator
==============================================================
Generates single-regulation factual MCQ questions from parsed articles.
Questions have verifiable ground truth directly from the regulation text.

Input:  Parsed article JSONs from parse_regulations.py
Output: JSON file with questions in benchmark format

Usage:
    # Generate from all parsed regulations
    python generate_tier1.py \
        --input_dir output \
        --output output/tier1_questions.json \
        --model meta-llama/Llama-3.1-8B-Instruct

    # Test with 3 articles first
    python generate_tier1.py \
        --input_dir output \
        --output output/tier1_questions.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --max_articles 3

    # Generate from specific regulation only
    python generate_tier1.py \
        --input_dir output \
        --output output/tier1_questions.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --regulations DORA

    # With 4-bit quantisation
    python generate_tier1.py \
        --input_dir output \
        --output output/tier1_questions.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --load_in_4bit

Requirements:
    pip install torch transformers accelerate
    pip install bitsandbytes  # only for --load_in_4bit or --load_in_8bit
"""

import json
import re
import argparse
import os
import gc
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def load_model(model_name, cache_dir=None, hf_token=None,
               load_in_4bit=False, load_in_8bit=False):
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH environment variable is not set.")

    if os.path.isabs(model_name):
        model_path = model_name
    else:
        model_dir = model_name.split("/")[-1]
        model_path = os.path.join(scratch, "models", model_dir)

    print(f"  Loading model from local path: {model_path}")

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    qconfig = None
    if load_in_4bit:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif load_in_8bit:
        qconfig = BitsAndBytesConfig(load_in_8bit=True)

    print("  Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True,
        torch_dtype=torch.float16, device_map="auto",
        quantization_config=qconfig)
    model.eval()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {mem:.1f}GB / {total:.1f}GB")

    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt, temperature=0.5, max_new_tokens=2000):
    messages = [{"role": "user", "content": prompt}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# JSON PARSING (robust for 8B models)
# ─────────────────────────────────────────────
def parse_json_response(text):
    if not text:
        return None

    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    # Try direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        extracted = match.group()
        # Try as-is
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
        # Fix trailing commas
        try:
            fixed = re.sub(r',\s*([}\]])', r'\1', extracted)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        # Fix single quotes
        try:
            fixed = extracted.replace("'", '"')
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Try extracting JSON array
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return {"questions": json.loads(match.group())}
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────
# QUESTION CATEGORIES PER REGULATION
# Maps each regulation to question categories
# so generated questions are well-distributed
# ─────────────────────────────────────────────
QUESTION_CATEGORIES = {
    "DORA": [
        "Incident Reporting", "ICT Risk Management", "Resilience Testing",
        "Third-Party Risk", "Proportionality", "Scope",
        "Information Sharing", "Governance",
    ],
    "GDPR": [
        "Breach Notification", "Data Subject Rights", "Lawful Basis",
        "Supervisory Authority", "Data Protection Officer",
        "Data Processing", "Transfers", "Penalties",
    ],
    "MiFID2": [
        "Investor Protection", "Record Keeping", "Best Execution",
        "Client Classification", "Product Governance", "Transparency",
        "Trading Venues", "Inducements",
    ],
    "MiCA": [
        "Token Classification", "Scope", "White Paper",
        "CASP Authorisation", "Significant Tokens",
        "Stablecoin Requirements", "Market Abuse", "Transitional Provisions",
    ],
    "NIS2": [
        "Scope", "Entity Classification", "Incident Reporting",
        "Risk Management", "Penalties", "Governance",
        "Supply Chain", "Supervisory Authority",
    ],
}


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
def build_prompt(regulation, article, questions_per_article=2):
    categories = QUESTION_CATEGORIES.get(regulation, ["General"])
    categories_str = ", ".join(categories)

    return f"""You are an EU financial regulation expert creating exam questions.

REGULATION: {regulation}
Article {article['article_number']}: {article['title']}

FULL ARTICLE TEXT:
{article['text']}

Generate exactly {questions_per_article} multiple-choice questions from this article.

RULES:
1. Each question must be answerable DIRECTLY from the article text above
2. The correct answer must be a FACT stated in the article (timeline, obligation,
   threshold, authority, entity type, etc.) — not an interpretation
3. Create a realistic scenario for each question (specific entity type, EU country)
4. Each question has 4 options: 1 correct, 3 plausible but wrong
5. Wrong options should be close to correct but factually wrong (wrong number,
   wrong authority, wrong entity type, wrong timeline)
6. Assign a category from: {categories_str}
7. Assign difficulty: "Easy" (direct fact lookup), "Medium" (requires understanding
   context), "Hard" (requires combining multiple paragraphs)
8. Vary the correct answer position across A, B, C, D

Respond ONLY with JSON:
{{"questions": [
  {{
    "regulation": "{regulation}",
    "category": "one of the categories above",
    "difficulty": "Easy",
    "source_article": "{regulation} Art. {article['article_number']}",
    "scenario": "A [entity type] in [EU country] [specific situation]...",
    "question": "What is required under {regulation}?",
    "option_a": "...",
    "option_b": "...",
    "option_c": "...",
    "option_d": "...",
    "correct_answer": "B",
    "explanation": "According to {regulation} Art. {article['article_number']}(X), [exact ground truth from article text]"
  }}
]}}

JSON only:"""


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
def run(args):
    # Load parsed articles
    print(f"\nLoading parsed articles from: {args.input_dir}")
    all_articles = []

    for json_file in sorted(Path(args.input_dir).glob("*_articles.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        reg_name = data["regulation"]

        # Filter by regulation if specified
        if args.regulations and reg_name not in args.regulations:
            continue

        articles = data["articles"]
        for art in articles:
            art["_regulation"] = reg_name
        all_articles.extend(articles)
        print(f"  {reg_name}: {len(articles)} articles")

    if not all_articles:
        print("ERROR: No articles found. Check --input_dir and --regulations")
        return

    # Limit for testing
    if args.max_articles:
        all_articles = all_articles[:args.max_articles]
        print(f"  Truncated to {args.max_articles} articles (--max_articles)")

    print(f"  Total articles to process: {len(all_articles)}")

    # Load model
    print(f"\n{'=' * 60}")
    print("LOADING MODEL")
    print(f"{'=' * 60}")
    model, tokenizer = load_model(
        args.model, args.cache_dir, args.hf_token,
        args.load_in_4bit, args.load_in_8bit)

    # Generate questions
    print(f"\n{'=' * 60}")
    print("GENERATING QUESTIONS")
    print(f"{'=' * 60}")

    all_questions = []
    question_id = 1
    errors = 0

    for i, article in enumerate(all_articles):
        reg = article["_regulation"]
        art_num = article["article_number"]
        title = article["title"]

        print(f"\n  [{i+1}/{len(all_articles)}] {reg} Art. {art_num}: {title[:50]}...")

        prompt = build_prompt(reg, article, args.questions_per_article)

        try:
            response = generate_response(
                model, tokenizer, prompt,
                temperature=0.5, max_new_tokens=2500)
            parsed = parse_json_response(response)
        except Exception as e:
            print(f"    ERROR: {e}")
            errors += 1
            continue

        if parsed and "questions" in parsed:
            for q in parsed["questions"]:
                # Assign ID
                q["id"] = f"T1-{question_id:03d}"
                question_id += 1

                # Normalise field names to match the spreadsheet format
                q.setdefault("regulation", reg)
                q.setdefault("source_article", f"{reg} Art. {art_num}")

                all_questions.append(q)

            print(f"    ✓ {len(parsed['questions'])} questions")
        else:
            errors += 1
            print(f"    ✗ Failed to parse")
            if response:
                print(f"      Raw: {response[:150]}...")

        # Clear cache periodically
        if (i + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    # Also save as the flat benchmark format (easy to load for evaluation)
    benchmark_path = args.output.replace(".json", "_benchmark.json")
    benchmark = []
    for q in all_questions:
        benchmark.append({
            "id": q.get("id", ""),
            "regulation": q.get("regulation", ""),
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "source_article": q.get("source_article", ""),
            "scenario": q.get("scenario", ""),
            "question": q.get("question", ""),
            "option_a": q.get("option_a", ""),
            "option_b": q.get("option_b", ""),
            "option_c": q.get("option_c", ""),
            "option_d": q.get("option_d", ""),
            "correct_answer": q.get("correct_answer", ""),
            "explanation": q.get("explanation", ""),
        })

    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total questions: {len(all_questions)}")
    print(f"  Parse errors:    {errors}")
    print(f"\n  Saved to:")
    print(f"    {args.output}")
    print(f"    {benchmark_path}")

    # Distribution
    reg_counts = {}
    diff_counts = {}
    cat_counts = {}
    correct_pos = {}
    for q in all_questions:
        r = q.get("regulation", "?")
        reg_counts[r] = reg_counts.get(r, 0) + 1
        d = q.get("difficulty", "?")
        diff_counts[d] = diff_counts.get(d, 0) + 1
        c = q.get("category", "?")
        cat_counts[c] = cat_counts.get(c, 0) + 1
        a = q.get("correct_answer", "?")
        correct_pos[a] = correct_pos.get(a, 0) + 1

    print(f"\n  By regulation:")
    for r, c in sorted(reg_counts.items()):
        print(f"    {r}: {c}")

    print(f"\n  By difficulty:")
    for d, c in sorted(diff_counts.items()):
        print(f"    {d}: {c}")

    print(f"\n  By category:")
    for cat, c in sorted(cat_counts.items()):
        print(f"    {cat}: {c}")

    print(f"\n  Correct answer position distribution:")
    for pos, c in sorted(correct_pos.items()):
        pct = c / max(len(all_questions), 1) * 100
        print(f"    {pos}: {c} ({pct:.0f}%)")

    # Check for position bias
    if correct_pos:
        max_pos = max(correct_pos.values())
        min_pos = min(correct_pos.values())
        if max_pos > min_pos * 2:
            print(f"\n  ⚠ WARNING: Correct answer position is unbalanced!")
            print(f"    You should randomise positions before using as benchmark.")

    # Free GPU
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\n  Next steps:")
    print(f"  1. Manually validate each question against EUR-Lex article text")
    print(f"  2. Check correct answers are factually accurate")
    print(f"  3. Check distractors are plausible but wrong")
    print(f"  4. Randomise correct answer positions if biased")
    print(f"  5. Merge with Tier 2 cross-regulation questions")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Tier 1 benchmark questions from parsed regulation articles",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input_dir", default="output",
                        help="Directory with *_articles.json files from parse_regulations.py")
    parser.add_argument("--output", default="output/tier1_questions.json",
                        help="Output JSON path")
    parser.add_argument("--regulations", nargs="*", default=None,
                        help="Only generate for these regulations (e.g., DORA GDPR)")
    parser.add_argument("--questions_per_article", type=int, default=2,
                        help="Questions to generate per article (default: 2)")
    parser.add_argument("--max_articles", type=int, default=None,
                        help="Max articles to process (for testing)")

    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--cache_dir", default=None)

    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--load_in_4bit", action="store_true")
    quant.add_argument("--load_in_8bit", action="store_true")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()