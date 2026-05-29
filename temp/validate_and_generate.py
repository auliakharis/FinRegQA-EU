"""
EU Financial Regulation Benchmark — Pair Validation & Question Generation Pipeline
===================================================================================
(Hugging Face Transformers version — no server needed)

This script takes the article_pairs_for_generation.json from parse_regulations.py
and runs a two-stage LLM pipeline:

  Stage 1 — VALIDATION: Quick check if each pair creates a genuine cross-regulation
            compliance scenario. Filters out weak pairs.

  Stage 2 — GENERATION: For pairs that pass validation (pass=true AND relevance>=3),
            generate full MCQ questions with scenarios, distractors, and citations.

Output:
  - validated_pairs.json        → All pairs with pass/fail status and scores
  - generated_questions.json    → Only passed pairs with their generated questions
  - all_questions_flat.json     → Flat list of all questions for benchmark use
  - pipeline_report.json        → Summary statistics of the pipeline run

Requirements:
    pip install torch transformers accelerate
    pip install bitsandbytes   # only if using --load_in_4bit or --load_in_8bit

Usage:
    # Basic usage with Llama 3.1 8B
    python validate_and_generate.py \
        --input output/article_pairs_for_generation.json \
        --output_dir ./output \
        --model meta-llama/Llama-3.1-8B-Instruct

    # With 4-bit quantisation (saves VRAM, ~8GB needed)
    python validate_and_generate.py \
        --input output/article_pairs_for_generation.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --load_in_4bit

    # Test with 5 pairs first
    python validate_and_generate.py \
        --input output/article_pairs_for_generation.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --max_pairs 5

    # Validate only (skip question generation)
    python validate_and_generate.py \
        --input output/article_pairs_for_generation.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --validate_only

    # Custom HF cache directory (common on clusters)
    python validate_and_generate.py \
        --input output/article_pairs_for_generation.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --cache_dir /scratch/$USER/hf_cache

Notes:
    - Requires HF_TOKEN env variable or --hf_token flag for gated models (Llama)
    - Accept the model license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    - On multi-GPU clusters, use CUDA_VISIBLE_DEVICES to select GPUs
    - 8B model needs ~16GB VRAM in float32, ~8GB in 4-bit quantisation
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
# MODEL LOADING
# ─────────────────────────────────────────────
def load_model(model_name: str, cache_dir: str = None, hf_token: str = None,
               load_in_4bit: bool = False, load_in_8bit: bool = False):
    """
    Load model and tokenizer from a local offline path under $SCRATCH.

    Args:
        model_name: Model directory name under $SCRATCH/models/ (e.g., "Llama-3.1-8B-Instruct")
        cache_dir: Unused (kept for CLI compatibility)
        hf_token: Unused (kept for CLI compatibility)
        load_in_4bit: Use 4-bit quantisation (saves VRAM, needs bitsandbytes)
        load_in_8bit: Use 8-bit quantisation (saves VRAM, needs bitsandbytes)
    """
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH environment variable is not set.")

    # Support both a bare model name and a full path
    if os.path.isabs(model_name):
        model_path = model_name
    else:
        # Strip any HF org prefix (e.g. "meta-llama/Llama-3.1-8B-Instruct" → "Llama-3.1-8B-Instruct")
        model_dir = model_name.split("/")[-1]
        model_path = os.path.join(scratch, "models", model_dir)

    print(f"  Loading model from local path: {model_path}")

    # Tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Set pad token if not set (common for Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantisation config
    quantization_config = None
    if load_in_4bit:
        print("  Using 4-bit quantisation (QLoRA-style)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        print("  Using 8-bit quantisation")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    print("  Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatically distribute across available GPUs
        quantization_config=quantization_config,
        low_cpu_mem_usage = True
    )

    model.eval()

    # Print device allocation
    if hasattr(model, "hf_device_map"):
        devices = set(str(v) for v in model.hf_device_map.values())
        print(f"  Model loaded on: {devices}")
    else:
        print(f"  Model loaded on: {next(model.parameters()).device}")

    # Memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {mem:.1f}GB / {total:.1f}GB used")

    return model, tokenizer


# ─────────────────────────────────────────────
# LLM INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def generate_response(model, tokenizer, prompt: str,
                      temperature: float = 0.3, max_new_tokens: int = 1500) -> str:
    """
    Generate a response from the model.

    Uses the model's chat template if available (Llama 3.1 Instruct has one).
    Falls back to raw prompt if no chat template.
    """
    # Build messages in chat format
    messages = [{"role": "user", "content": prompt}]

    try:
        # Use the model's built-in chat template (Llama 3.1 Instruct supports this)
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback if no chat template
        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generation config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the new tokens (skip the input)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


# ─────────────────────────────────────────────
# JSON PARSING
# ─────────────────────────────────────────────
def parse_json_response(response_text: str) -> dict | None:
    """
    Safely parse JSON from LLM response.
    Handles: markdown fences, trailing commas, extra text around JSON.

    8B models are more likely to produce malformed JSON, so we try
    multiple repair strategies.
    """
    if not response_text:
        return None

    text = response_text.strip()

    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from surrounding text
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            # Try fixing the extracted JSON
            extracted = json_match.group()

            # Fix trailing commas
            try:
                fixed = re.sub(r',\s*([}\]])', r'\1', extracted)
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

            # Fix single quotes → double quotes
            try:
                fixed = extracted.replace("'", '"')
                fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    # Last resort: fix common 8B model issues on original text
    try:
        fixed = re.sub(r',\s*([}\]])', r'\1', text)
        fixed = fixed.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None


# ─────────────────────────────────────────────
# STAGE 1 — VALIDATION PROMPT
# ─────────────────────────────────────────────
def build_validation_prompt(pair: dict) -> str:
    """Build validation prompt. Uses truncated text to fit 8B context window."""
    reg_a = pair["regulation_a"]
    reg_b = pair["regulation_b"]

    # Truncate for validation — keep it short for 8B
    text_a = reg_a["text"][:500].strip()
    text_b = reg_b["text"][:500].strip()

    return f"""You are an EU financial regulation expert.

Evaluate if this article pair creates a real cross-regulation compliance scenario
for EU financial entities (banks, insurers, CASPs, investment firms).

ARTICLE 1: {reg_a['name']} Article {reg_a['article_number']}: {reg_a['title']}
{text_a}...

ARTICLE 2: {reg_b['name']} Article {reg_b['article_number']}: {reg_b['title']}
{text_b}...

Answer these 3 questions:
1. Would a compliance officer need to consider BOTH articles for one specific event?
2. Do the articles create a dual obligation, tension, scope boundary, or authority conflict?
3. Can a question be written where BOTH articles are needed to find the correct answer?

If ALL 3 answers are yes, the pair passes.

Respond ONLY with this exact JSON format, nothing else:
{{"pass": true, "relevance_score": 4, "interaction_type": "dual_obligation", "reasoning": "one sentence why", "suggested_scenario_seed": "brief scenario description"}}

Values:
- pass: true or false
- relevance_score: 1 to 5 (5 = critical overlap)
- interaction_type: "dual_obligation" or "tension" or "scope_boundary" or "authority_conflict" or "none"
- reasoning: one sentence
- suggested_scenario_seed: brief scenario if pass is true, "N/A" if false

JSON only:"""


# ─────────────────────────────────────────────
# STAGE 2 — QUESTION GENERATION PROMPT
# ─────────────────────────────────────────────
def build_question_generation_prompt(pair: dict, scenario_seed: str) -> str:
    """Build generation prompt adapted for 8B model."""
    reg_a = pair["regulation_a"]
    reg_b = pair["regulation_b"]

    return f"""You are an EU financial regulation expert facing real world financial problems, and you want to create questions with realistic scenario and truthful answers based on the regulation.

SCENARIO IDEA: {scenario_seed}

REGULATION A: {reg_a['name']} Article {reg_a['article_number']}: {reg_a['title']}
{reg_a['text']}

REGULATION B: {reg_b['name']} Article {reg_b['article_number']}: {reg_b['title']}
{reg_b['text']}

Generate 3 multiple-choice questions. Rules:
- Each question starts with a realistic scenario (specific company type, EU country, specific event)
- The correct answer REQUIRES knowledge of BOTH {reg_a['name']} and {reg_b['name']}
- Each question has 4 options: 1 correct, 3 wrong
- Wrong answers must be plausible (not obviously wrong)
- One wrong answer should ignore {reg_a['name']}, one should ignore {reg_b['name']}, one should have a factual error
- Cite specific article numbers
- Mix difficulty: 1 easy, 1 medium, 1 hard
- Vary the correct answer position (don't always use B)

Respond ONLY with JSON in this exact format:
{{"questions": [
  {{
    "scenario": "A [entity] in [country] [situation]...",
    "question": "What must the entity do?",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "C",
    "explanation": "Explanation citing {reg_a['name']} Art. X and {reg_b['name']} Art. Y...",
    "source_articles": {{"{reg_a['name']}": ["Art. {reg_a['article_number']}"], "{reg_b['name']}": ["Art. {reg_b['article_number']}"]}},
    "distractor_analysis": {{
      "A": {{"error_type": "ignores_{reg_b['name'].lower()}", "explanation": "..."}},
      "B": {{"error_type": "ignores_{reg_a['name'].lower()}", "explanation": "..."}},
      "D": {{"error_type": "fabricated_rule", "explanation": "..."}}
    }},
    "reasoning_type": "cross_regulation_overlap",
    "difficulty": "medium"
  }}
]}}

JSON only:"""


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(args):
    """Main pipeline: load model → load pairs → validate → generate → save."""

    # ─── GENERATE ONLY: skip validation, load from validated_pairs.json ───
    if args.generate_only:
        validated_path = Path(args.output_dir) / "validated_pairs.json"
        print(f"\nLoading validated pairs from: {validated_path}")
        with open(validated_path, "r", encoding="utf-8") as f:
            validated_pairs = json.load(f)

        # Reconstruct passed_pairs from the validation results
        # We need to merge the article text back from the original input file
        print(f"  Loading original pairs for article text from: {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            original_pairs = json.load(f)
        original_by_id = {p.get("pair_id", f"pair_{i}"): p for i, p in enumerate(original_pairs)}

        passed_pairs = []
        for entry in validated_pairs:
            if entry.get("final_status") == "PASS":
                pair_id = entry["pair_id"]
                orig = original_by_id.get(pair_id, {})
                passed_pairs.append({**orig, "validation": entry["validation"]})

        if args.max_pairs:
            passed_pairs = passed_pairs[:args.max_pairs]
            print(f"  Truncated to {args.max_pairs} pairs (--max_pairs)")

        print(f"  PASS pairs to generate from: {len(passed_pairs)}")

        # Load model and jump straight to Stage 2
        print(f"\n{'=' * 60}")
        print("LOADING MODEL")
        print(f"{'=' * 60}")
        model, tokenizer = load_model(
            model_name=args.model,
            cache_dir=args.cache_dir,
            hf_token=args.hf_token,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )

        stats = {
            "total_pairs": len(passed_pairs),
            "validated": len(validated_pairs),
            "passed": len(passed_pairs),
            "failed": len(validated_pairs) - len(passed_pairs),
            "parse_errors": 0,
            "by_interaction_type": {},
            "by_overlap_zone": {},
            "by_regulation_pair": {},
        }

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.generate_only:
        # Load model
        print(f"\n{'=' * 60}")
        print("LOADING MODEL")
        print(f"{'=' * 60}")
        model, tokenizer = load_model(
            model_name=args.model,
            cache_dir=args.cache_dir,
            hf_token=args.hf_token,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )

        # Load article pairs
        print(f"\nLoading article pairs from: {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
            print(f"  Truncated to {args.max_pairs} pairs (--max_pairs)")

        print(f"  Total pairs to process: {len(pairs)}")

        # ─── STAGE 1: VALIDATION ───
        print(f"\n{'=' * 60}")
        print("STAGE 1: PAIR VALIDATION")
        print(f"{'=' * 60}")

        validated_pairs = []
        passed_pairs = []
        stats = {
            "total_pairs": len(pairs),
            "validated": 0,
            "passed": 0,
            "failed": 0,
            "parse_errors": 0,
            "by_interaction_type": {},
            "by_overlap_zone": {},
            "by_regulation_pair": {},
        }

        for i, pair in enumerate(pairs):
            pair_id = pair.get("pair_id", f"pair_{i}")
            print(f"\n  [{i+1}/{len(pairs)}] Validating: {pair_id}")

            prompt = build_validation_prompt(pair)

            try:
                response_text = generate_response(
                    model, tokenizer, prompt,
                    temperature=0.3, max_new_tokens=300
                )
                validation = parse_json_response(response_text)
            except Exception as e:
                print(f"    ERROR: {e}")
                response_text = None
                validation = None

            if validation is None:
                stats["parse_errors"] += 1
                print(f"    PARSE ERROR")
                if response_text:
                    print(f"    Raw: {response_text[:200]}...")
                validation = {
                    "pass": False,
                    "relevance_score": 0,
                    "interaction_type": "parse_error",
                    "reasoning": "Could not parse model response as JSON",
                    "suggested_scenario_seed": "N/A",
                    "raw_response": (response_text or "")[:500],
                }

            validated_entry = {
                "pair_id": pair_id,
                "overlap_zone": pair.get("overlap_zone", "unknown"),
                "regulation_a": {
                    "name": pair["regulation_a"]["name"],
                    "article_number": pair["regulation_a"]["article_number"],
                    "title": pair["regulation_a"]["title"],
                },
                "regulation_b": {
                    "name": pair["regulation_b"]["name"],
                    "article_number": pair["regulation_b"]["article_number"],
                    "title": pair["regulation_b"]["title"],
                },
                "validation": validation,
            }

            is_passed = (
                validation.get("pass", False) is True
                and validation.get("relevance_score", 0) >= 4
            )
            validated_entry["final_status"] = "PASS" if is_passed else "FAIL"

            icon = "✓" if is_passed else "✗"
            score = validation.get("relevance_score", "?")
            itype = validation.get("interaction_type", "?")
            reason = validation.get("reasoning", "?")[:80]
            print(f"    {icon} score={score} type={itype}")
            print(f"      {reason}")

            validated_pairs.append(validated_entry)
            stats["validated"] += 1

            if is_passed:
                stats["passed"] += 1
                passed_pairs.append({**pair, "validation": validation})
            else:
                stats["failed"] += 1

            # Track stats
            itype_key = validation.get("interaction_type", "unknown")
            stats["by_interaction_type"][itype_key] = stats["by_interaction_type"].get(itype_key, 0) + 1

            zone = pair.get("overlap_zone", "unknown")
            if zone not in stats["by_overlap_zone"]:
                stats["by_overlap_zone"][zone] = {"total": 0, "passed": 0}
            stats["by_overlap_zone"][zone]["total"] += 1
            if is_passed:
                stats["by_overlap_zone"][zone]["passed"] += 1

            rp_key = f"{pair['regulation_a']['name']} × {pair['regulation_b']['name']}"
            if rp_key not in stats["by_regulation_pair"]:
                stats["by_regulation_pair"][rp_key] = {"total": 0, "passed": 0}
            stats["by_regulation_pair"][rp_key]["total"] += 1
            if is_passed:
                stats["by_regulation_pair"][rp_key]["passed"] += 1

        # Save validation results (ALL pairs)
        val_path = Path(args.output_dir) / "validated_pairs.json"
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(validated_pairs, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(validated_pairs)} validated pairs to: {val_path}")

        # Print summary
        print(f"\n{'─' * 40}")
        print(f"VALIDATION SUMMARY")
        print(f"{'─' * 40}")
        print(f"  Total:        {stats['total_pairs']}")
        print(f"  Passed:       {stats['passed']} ({stats['passed']/max(stats['total_pairs'],1)*100:.0f}%)")
        print(f"  Failed:       {stats['failed']}")
        print(f"  Parse errors: {stats['parse_errors']}")
        print(f"\n  By overlap zone:")
        for zone, c in sorted(stats["by_overlap_zone"].items()):
            pct = c['passed'] / max(c['total'], 1) * 100
            print(f"    {zone}: {c['passed']}/{c['total']} passed ({pct:.0f}%)")
        print(f"\n  By regulation pair:")
        for rp, c in sorted(stats["by_regulation_pair"].items()):
            pct = c['passed'] / max(c['total'], 1) * 100
            print(f"    {rp}: {c['passed']}/{c['total']} passed ({pct:.0f}%)")

    # ─── STAGE 2: QUESTION GENERATION ───
    if args.validate_only:
        print(f"\n  --validate_only set, skipping generation")
    elif len(passed_pairs) == 0:
        print(f"\n  No pairs passed — skipping generation")
    else:
        print(f"\n{'=' * 60}")
        print(f"STAGE 2: QUESTION GENERATION ({len(passed_pairs)} pairs)")
        print(f"{'=' * 60}")

        all_generated = []
        generation_errors = 0

        for i, pair in enumerate(passed_pairs):
            pair_id = pair.get("pair_id", f"pair_{i}")
            scenario_seed = pair["validation"].get(
                "suggested_scenario_seed",
                "A financial entity in the EU faces a regulatory compliance scenario."
            )

            print(f"\n  [{i+1}/{len(passed_pairs)}] Generating: {pair_id}")
            print(f"    Seed: {scenario_seed[:80]}...")

            prompt = build_question_generation_prompt(pair, scenario_seed)

            try:
                response_text = generate_response(
                    model, tokenizer, prompt,
                    temperature=0.7,
                    max_new_tokens=3000,
                )
                questions_data = parse_json_response(response_text)
            except Exception as e:
                print(f"    ERROR: {e}")
                response_text = None
                questions_data = None

            if questions_data and "questions" in questions_data:
                num_q = len(questions_data["questions"])
                print(f"    ✓ Generated {num_q} questions")

                for q in questions_data["questions"]:
                    q["source_pair_id"] = pair_id
                    q["overlap_zone"] = pair.get("overlap_zone", "unknown")
                    q["regulation_pair"] = (
                        f"{pair['regulation_a']['name']} × {pair['regulation_b']['name']}"
                    )
                    q["validation_score"] = pair["validation"].get("relevance_score", 0)
                    q["interaction_type"] = pair["validation"].get("interaction_type", "unknown")

                all_generated.append({
                    "pair_id": pair_id,
                    "overlap_zone": pair.get("overlap_zone", "unknown"),
                    "regulation_a": {
                        "name": pair["regulation_a"]["name"],
                        "article_number": pair["regulation_a"]["article_number"],
                        "title": pair["regulation_a"]["title"],
                    },
                    "regulation_b": {
                        "name": pair["regulation_b"]["name"],
                        "article_number": pair["regulation_b"]["article_number"],
                        "title": pair["regulation_b"]["title"],
                    },
                    "validation": pair["validation"],
                    "questions": questions_data["questions"],
                })
            else:
                generation_errors += 1
                print(f"    ✗ Failed")
                if response_text:
                    print(f"      Raw: {response_text[:200]}...")

            # # Clear GPU cache periodically
            # if (i + 1) % 10 == 0 and torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     gc.collect()

        # Save grouped
        gen_path = Path(args.output_dir) / "generated_questions.json"
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(all_generated, f, indent=2, ensure_ascii=False)

        # Save flat
        flat_questions = []
        for entry in all_generated:
            for q in entry.get("questions", []):
                flat_questions.append(q)

        flat_path = Path(args.output_dir) / "all_questions_flat.json"
        with open(flat_path, "w", encoding="utf-8") as f:
            json.dump(flat_questions, f, indent=2, ensure_ascii=False)

        total_q = len(flat_questions)
        stats["total_questions_generated"] = total_q
        stats["generation_errors"] = generation_errors

        print(f"\n{'─' * 40}")
        print(f"GENERATION SUMMARY")
        print(f"{'─' * 40}")
        print(f"  Pairs processed:     {len(passed_pairs)}")
        print(f"  Questions generated: {total_q}")
        print(f"  Errors:              {generation_errors}")
        print(f"\n  Files:")
        print(f"    {gen_path}")
        print(f"    {flat_path}")

        diff_counts = {}
        for q in flat_questions:
            d = q.get("difficulty", "unknown")
            diff_counts[d] = diff_counts.get(d, 0) + 1
        if diff_counts:
            print(f"\n  By difficulty:")
            for d, c in sorted(diff_counts.items()):
                print(f"    {d}: {c}")

    # ─── SAVE REPORT ───
    stats["timestamp"] = datetime.now().isoformat()
    stats["model"] = args.model
    stats["quantisation"] = "4bit" if args.load_in_4bit else ("8bit" if args.load_in_8bit else "float16")

    report_path = Path(args.output_dir) / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Free GPU memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Report: {report_path}")
    print(f"\n  Next steps:")
    print(f"  1. Review validated_pairs.json — check pass/fail decisions")
    print(f"  2. Review generated_questions.json — spot-check quality")
    print(f"  3. Manually validate questions against regulation texts")
    print(f"  4. Build final benchmark dataset")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Validate article pairs and generate benchmark questions (HF Transformers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python validate_and_generate.py \\
      --input output/article_pairs_for_generation.json \\
      --model meta-llama/Llama-3.1-8B-Instruct

  # 4-bit quantisation (less VRAM)
  python validate_and_generate.py \\
      --input output/article_pairs_for_generation.json \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --load_in_4bit

  # Test run
  python validate_and_generate.py \\
      --input output/article_pairs_for_generation.json \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --max_pairs 5

  # Validate only
  python validate_and_generate.py \\
      --input output/article_pairs_for_generation.json \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --validate_only

  # Cluster with custom cache
  python validate_and_generate.py \\
      --input output/article_pairs_for_generation.json \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --cache_dir /scratch/$USER/hf_cache \\
      --load_in_4bit
        """
    )

    parser.add_argument("--input", required=True,
                        help="Path to article_pairs_for_generation.json")
    parser.add_argument("--output_dir", default="output",
                        help="Output directory (default: output)")

    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HF model ID (default: meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--hf_token", default=None,
                        help="Hugging Face token (or set HF_TOKEN env variable)")
    parser.add_argument("--cache_dir", default=None,
                        help="HF model cache directory")

    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--load_in_4bit", action="store_true",
                       help="4-bit quantisation (~8GB VRAM for 8B model)")
    quant.add_argument("--load_in_8bit", action="store_true",
                       help="8-bit quantisation (~10GB VRAM for 8B model)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate_only", action="store_true",
                      help="Only validate, skip question generation")
    mode.add_argument("--generate_only", action="store_true",
                      help="Skip validation; load PASS pairs from output_dir/validated_pairs.json and generate questions")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max pairs to process (for testing)")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()