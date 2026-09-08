"""
Evaluate DPO-trained Qwen-4B vs baseline Qwen-4B on held-out regulatory Q&A.

Three independent phases:

  generate  Load baseline and DPO model locally; run both on test questions.
            Saves output/eval_answers.jsonl.

  judge     Call the vLLM-served judge panel (same 3 models as training data)
            to pointwise-score each generated answer.
            Saves output/eval_judge_scores.jsonl.

  report    Aggregate judge scores → per-dimension means + win rate.
            Compute citation F1 (extracted citations vs ground truth) and
            BERTScore (semantic similarity to ground truth).
            Prints a summary table and saves output/eval_report.json.

Test set: questions in judge_results_train_api.jsonl that are NOT in
dpo_corruption_pairs.jsonl (1,779 of 2,138 total). 100 are sampled by default
(~5.6% of the held-out pool, stratified by random seed). Pass --n_questions -1
to use all held-out questions.

Usage
-----
    # Full pipeline
    python eval_dpo.py --phase all --baseline_model Qwen3.5-4B-Instruct

    # Quick smoke-test: 10 questions, one judge
    python eval_dpo.py --phase all --baseline_model Qwen3.5-4B-Instruct \\
        --n_questions 10 --judge_models Qwen/Qwen3.5-27B

    # Separate phases (e.g. generate on GPU node, report on login node)
    python eval_dpo.py --phase generate --baseline_model Qwen3.5-4B-Instruct
    python eval_dpo.py --phase judge output/eval_hallucinate_text output/eval_hallucinate_citation
    python eval_dpo.py --phase report
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

try:
    from bert_score import score as _bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

try:
    from citation_parser import find_citation_spans
    CITATION_PARSER_AVAILABLE = True
except ImportError:
    CITATION_PARSER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

BASE              = Path("output")
RECORDS_FILE      = BASE / "judge_results_train_api.jsonl"
DPO_PAIRS_FILE    = BASE / "dpo_corruption_pairs.jsonl"

DEFAULT_DPO_MODEL_PATH = Path("/cluster/scratch/arakhmasari/dpo_llama3b")

ALL_JUDGE_MODELS = [
    # "Qwen/Qwen3.5-27B",
    # "google/gemma-4-31B-it",
    # "zai-org/GLM-4.7-Flash",
    "RCP-AIaaS/Qwen/Qwen3.6-35B-A3B",
    "CSCS-Inference/google/gemma-4-31B-it",
    "RCP-AIaaS/zai-org/GLM-5.3-Flash"
]
DIMS = ["accuracy", "completeness", "topic_coherence", "citation_quality"]

JUDGE_MAX_TOKENS = {
    # "Qwen/Qwen3.5-27B":       6144,
    # "google/gemma-4-31B-it":  6144,
    # "zai-org/GLM-4.7-Flash":  8192,
    "RCP-AIaaS/Qwen/Qwen3.6-35B-A3B"        : 6144,
    "CSCS-Inference/google/gemma-4-31B-it"  : 6144,
    "RCP-AIaaS/zai-org/GLM-5.3-Flash"       : 8192,
}
DEFAULT_JUDGE_MAX_TOKENS = 3072

# Lab members have redeployed the same underlying judge models under different
# endpoint names over time (a "-<person>-<tag>" suffix marks who/where it's
# served from). Map every known deployment name to one canonical id so scores
# recorded under different suffixes are treated as the same judge when
# aggregating or checking for existing results.
JUDGE_ALIASES = {
    "google/gemma-4-31B-it-bdoan": "google/gemma-4-31B-it",
    "google/gemma-4-31B-it": "google/gemma-4-31B-it",
    "zai-org/GLM-4.7-Flash-bdoan": "zai-org/GLM-4.7-Flash",
    "zai-org/GLM-4.7-Flash": "zai-org/GLM-4.7-Flash",
    "zai-org/GLM-4.7-Flash-rwindesheim-cfbig": "zai-org/GLM-4.7-Flash",
    "Qwen/Qwen3.5-27B": "Qwen/Qwen3.5-27B",
}


def canonical_judge(name: str) -> str:
    return JUDGE_ALIASES.get(name, name)

# Reused verbatim from judge_api.py
JUDGE_PROMPT = """\
You are an expert judge evaluating answers to EU financial regulation \
questions from EBA and ESMA sources. You will score a candidate answer \
against the official answer on four dimensions.

## Question
{question}

## Topic
{topic}

## Subject Matter
{subject_matter}

## Official Answer (Ground Truth)
{ground_truth}

## Candidate Answer
{candidate}

## Scoring Dimensions

**Accuracy** (factual correctness relative to the official answer):
- 5: All factual claims match the official answer.
- 4: Minor inaccuracies that do not change the substantive conclusion.
- 3: Partially correct; one substantive claim is wrong or unsupported.
- 2: Multiple substantive errors; conclusion is partly incorrect.
- 1: Conclusion contradicts the official answer or is fabricated.

**Completeness** (coverage of key points in the official answer):
- 5: Covers all key points present in the official answer.
- 4: Covers all key points but omits a minor detail.
- 3: Covers the main point but misses one secondary point.
- 2: Misses multiple key points; partial coverage.
- 1: Misses the main point entirely.

**Topic Coherence** (alignment with the specified topic and subject matter):
- 5: Fully addresses the specified topic and subject matter without \
drifting into adjacent regulatory areas.
- 4: Stays on topic but includes minor tangential content, OR addresses \
a broader scope that still fully covers the topic.
- 3: Partially on topic; some content addresses a different but related \
regulatory area.
- 2: Primarily addresses an adjacent topic; only partially relevant to \
the specified subject matter.
- 1: Off-topic or addresses a different regulatory area entirely.

**Citation Quality** (specificity and correctness of legal references):
- 5: All citations are specific (article/paragraph/field level) and \
correctly identify the supporting provision.
- 4: Citations are specific and mostly correct; one minor citation issue.
- 3: Citations are present but partially generic (e.g., "Article 5" \
without specifying the regulation), or one citation is incorrect.
- 2: Citations are mostly generic ("the ITS", "DORA") or several are \
incorrect.
- 1: Citations are missing or fabricated.

## Instructions

1. **Reason before scoring.** Think step-by-step about each dimension \
before assigning the score. Your reasoning should determine the score, \
not the reverse.

2. **Catch plausible-sounding hallucinations.** If the candidate \
contains specific factual claims (numbers, dates, article references, \
requirements, definitions) that are not supported by the official \
answer, treat those as accuracy violations even if the claims sound \
plausible.

3. **Score dimensions independently.** If two dimensions seem to \
conflict, score each on its own merits.

4. **Use the full 1-5 range.** Do not default to 3 when uncertain. \
A shorter candidate that still covers the key point is not a \
completeness penalty.

4b. **Score must match your own reasoning.** If your reasoning \
sentence for a dimension names a specific flaw (a missing point, a \
wrong citation, an unsupported claim, drift off-topic), the score for \
that dimension cannot be 5. Reserve 5 only when your reasoning states \
the candidate fully and correctly meets the criterion with no caveat. \
Do not default to 5 out of leniency.

5. **Citation specificity matters.** A correct citation to "Article 5 \
of Regulation (EU) 2019/2033" is stronger than "Article 5" alone, \
even when both are technically correct.

6. **Be concise.** Do your step-by-step thinking silently. Each \
reasoning field must be ONE short sentence (max ~25 words) stating the \
conclusion, not a transcript of your deliberation. Do not repeat the \
question, the candidate text, or the official answer back. Output the \
JSON object immediately after you have decided the scores — no \
preamble, no text before or after the JSON.

7. **No visible thinking.** Do not output a `<think>` block, chain-of-\
thought, or any reasoning outside the JSON's "reasoning" fields. Your \
entire response must be the JSON object and nothing else.

Respond in this exact JSON format only:
{{
  "reasoning": {{
    "accuracy": "<one sentence>",
    "completeness": "<one sentence>",
    "topic_coherence": "<one sentence>",
    "citation_quality": "<one sentence>"
  }},
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "topic_coherence": <1-5>,
  "citation_quality": <1-5>
}}"""

ANSWERER_PROMPT = """\
You are an expert in EU financial regulation, with deep knowledge of EBA and \
ESMA guidelines, technical standards, and related directives and regulations.

Answer the following regulatory question. Base your answer on the actual \
content of the legal act identified below and support every substantive \
claim with a specific citation in the form [Source, Article/Paragraph]. \
Write 100–400 words of prose, matching the style of official EBA/ESMA Q&A \
responses, without padding or restating the question.

## Context
LEGAL ACT: {legal_act}
TOPIC: {topic}
SUBJECT MATTER: {subject_matter}

## Question
{question}

Answer:
"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model_local(model_path: Path, load_in_4bit: bool = False):
    model_path = Path(model_path)
    print(f"  Loading: {model_path}")

    # Detect LoRA adapter saved by TRL/PEFT (has adapter_config.json but no model weights).
    # Use PEFT's own loading path to avoid transformers' integrations/peft.py which
    # requires a newer _maybe_shard_state_dict_for_tp symbol.
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        with adapter_config_path.open() as f:
            adapter_cfg = json.load(f)
        base_path = Path(adapter_cfg["base_model_name_or_path"])
        print(f"  Detected LoRA adapter — loading base from: {base_path}")
    else:
        base_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    qconfig = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), local_files_only=True,
        torch_dtype=compute_dtype, device_map="auto",
        quantization_config=qconfig,
    )

    if adapter_config_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(model_path))
        model = model.merge_and_unload()

    return model, tokenizer


_LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except ValueError:
            text = _LLAMA3_TEMPLATE.format(prompt=prompt)
    except ValueError:
        text = _LLAMA3_TEMPLATE.format(prompt=prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    terminators = [t for t in terminators if t is not None]

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    out = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip any residual <think> blocks as a safety net.
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
    return out


def select_test_questions(n: int | None, seed: int) -> list[dict]:
    all_records = load_jsonl(RECORDS_FILE)
    if DPO_PAIRS_FILE.exists():
        dpo_qids = {r["question_id"] for r in load_jsonl(DPO_PAIRS_FILE)}
    else:
        dpo_qids = set()
        print(f"Warning: {DPO_PAIRS_FILE} not found — using all questions.")
    questions = [r for r in all_records if r["question_id"] not in dpo_qids]
    if n is not None and n != -1:
        import random
        random.Random(seed).shuffle(questions)
        questions = questions[:n]
    print(f"Test set: {len(questions)} questions  (excluded {len(dpo_qids)} DPO training qids)")
    return questions


# ---------------------------------------------------------------------------
# Judge API helpers (mirrors judge_api.py)
# ---------------------------------------------------------------------------

def _is_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    return any(m in name for m in ("qwen", "glm"))


def _thinking_kwargs(model_name: str) -> dict:
    name = model_name.lower()
    if "qwen" in name:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if "glm" in name:
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def _call_judge(client: OpenAI, model_name: str, prompt: str, max_new_tokens: int) -> str:
    if _is_reasoning_model(model_name):
        prompt = f"{prompt}\n\n/no_think"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.0,
        **_thinking_kwargs(model_name),
    )
    msg = response.choices[0].message
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        raise ValueError(
            f"Empty response from {model_name} (finish_reason={response.choices[0].finish_reason})"
        )
    return text


def call_judge(client: OpenAI, model_name: str, prompt: str, max_new_tokens: int) -> str:
    try:
        return _call_judge(client, model_name, prompt, max_new_tokens)
    except ValueError:
        if not _is_reasoning_model(model_name):
            raise
        # Reasoning models occasionally exhaust their budget on hidden CoT; retry with 2x tokens.
        return _call_judge(client, model_name, prompt, max_new_tokens * 2)


# ---------------------------------------------------------------------------
# Score parsing (mirrors judge_api.py)
# ---------------------------------------------------------------------------

def _find_balanced_objects(text: str) -> list[str]:
    blocks, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start : i + 1])
    return blocks


def parse_scores(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    for block in _find_balanced_objects(text):
        try:
            data = json.loads(block)
            if all(k in data for k in DIMS):
                result = {k: data[k] for k in DIMS}
                if isinstance(data.get("reasoning"), dict):
                    result["reasoning"] = {k: data["reasoning"].get(k) for k in DIMS}
                return result
        except json.JSONDecodeError:
            continue
    return {k: None for k in DIMS}


# ---------------------------------------------------------------------------
# Citation F1
# ---------------------------------------------------------------------------

def _citation_set(text: str) -> set[str]:
    if not CITATION_PARSER_AVAILABLE:
        return set()
    spans = find_citation_spans(text)
    return {text[s:e].lower().strip() for s, e, *_ in spans}


def citation_f1(pred: str, ref: str) -> float:
    pred_c, ref_c = _citation_set(pred), _citation_set(ref)
    if not ref_c and not pred_c:
        return 1.0
    if not ref_c or not pred_c:
        return 0.0
    tp = len(pred_c & ref_c)
    p  = tp / len(pred_c)
    r  = tp / len(ref_c)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Phase 1 · generate
# ---------------------------------------------------------------------------

def generate_phase(args) -> None:
    scratch = os.environ.get("SCRATCH", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Question set and starting answers dict ---
    if args.models == "dpo":
        if args.baseline_answers is None:
            raise ValueError("--baseline_answers is required when --models dpo")
        baseline_rows = load_jsonl(args.baseline_answers)
        answers = {r["question_id"]: r for r in baseline_rows}
        questions = baseline_rows
        print(f"Loaded {len(questions)} questions from {args.baseline_answers}")
    else:
        questions = select_test_questions(args.n_questions, args.seed)
        answers = {
            r["question_id"]: {
                "question_id":  r["question_id"],
                "regulator":    r.get("regulator", ""),
                "question":     r["question"],
                "ground_truth": r.get("ground_truth", ""),
                "meta":         r.get("meta", {}),
                "baseline":     {},
                "dpo":          {},
            }
            for r in questions
        }

    # --- Which models to run ---
    if args.models == "baseline":
        model_configs = [("baseline", Path(scratch) / "models" / args.baseline_model)]
    elif args.models == "dpo":
        model_configs = [("dpo", args.dpo_model_path)]
    else:
        model_configs = [
            ("baseline", Path(scratch) / "models" / args.baseline_model),
            ("dpo",      args.dpo_model_path),
        ]

    for model_key, model_path in model_configs:
        print(f"\n=== Generating [{model_key}] from {model_path} ===")
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        model, tokenizer = load_model_local(model_path, args.load_in_4bit)

        for i, rec in enumerate(questions):
            meta   = rec.get("meta", {})
            prompt = ANSWERER_PROMPT.format(
                legal_act=meta.get("legal_act", ""),
                topic=meta.get("topic", ""),
                subject_matter=meta.get("subject_matter", ""),
                question=rec["question"],
            )
            text = generate_answer(model, tokenizer, prompt, args.max_new_tokens)
            answers[rec["question_id"]][model_key] = {
                "model": str(model_path),
                "text":  text,
            }
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(questions)}")

        del model, tokenizer
        torch.cuda.empty_cache()

    out = args.output_dir / "eval_answers.jsonl"
    with out.open("w") as f:
        for row in answers.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(answers)} rows → {out}")


# ---------------------------------------------------------------------------
# Phase 2 · judge
# ---------------------------------------------------------------------------

def judge_phase(args) -> None:
    answers_file = args.output_dir / "eval_answers.jsonl"
    if not answers_file.exists():
        raise FileNotFoundError(f"{answers_file} not found — run --phase generate first.")

    client = OpenAI(
        api_key=os.environ.get("CSCS_SERVING_API"),
        base_url=args.judge_api_base,
    )
    rows   = load_jsonl(answers_file)

    # Seed from existing scores so already-judged (qid, model_key, judge) combos are skipped.
    out = args.output_dir / "eval_judge_scores.jsonl"
    existing: dict[str, dict] = {}
    seed_file = args.existing_scores or (out if out.exists() else None)
    if seed_file and Path(seed_file).exists():
        for r in load_jsonl(Path(seed_file)):
            existing[r["question_id"]] = r
        print(f"Loaded existing scores from {seed_file} ({len(existing)} questions) — skipping completed entries.")

    scored: list[dict] = []

    for i, row in enumerate(rows):
        qid  = row["question_id"]
        meta = row.get("meta", {})
        prev = existing.get(qid, {})
        scores_for_q: dict[str, dict] = {
            "baseline": dict(prev.get("scores", {}).get("baseline", {})),
            "dpo":      dict(prev.get("scores", {}).get("dpo", {})),
        }

        for judge_model in args.judge_models:
            max_tokens = JUDGE_MAX_TOKENS.get(judge_model, DEFAULT_JUDGE_MAX_TOKENS)
            for model_key in ("baseline", "dpo"):
                existing_sc = scores_for_q[model_key].get(judge_model, {})
                if all(existing_sc.get(d) is not None for d in DIMS):
                    continue  # already scored
                candidate = row.get(model_key, {}).get("text", "")
                if not candidate:
                    continue
                prompt = JUDGE_PROMPT.format(
                    question=row["question"],
                    topic=meta.get("topic", ""),
                    subject_matter=meta.get("subject_matter", ""),
                    ground_truth=row["ground_truth"],
                    candidate=candidate,
                )
                try:
                    raw = call_judge(client, judge_model, prompt, max_tokens)
                    sc  = parse_scores(raw)
                except Exception as e:
                    print(f"  Judge error ({judge_model}, {model_key}, {qid}): {e}")
                    sc = {k: None for k in DIMS}
                scores_for_q[model_key][judge_model] = sc

        scored.append({
            "question_id": qid,
            "regulator":   row.get("regulator", ""),
            "scores":      scores_for_q,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} judged")

    out = args.output_dir / "eval_judge_scores.jsonl"
    with out.open("w") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(scored)} rows → {out}")


# ---------------------------------------------------------------------------
# Phase 3 · report
# ---------------------------------------------------------------------------

def report_phase(args) -> None:
    answers_file = args.output_dir / "eval_answers.jsonl"
    scores_file  = args.output_dir / "eval_judge_scores.jsonl"
    for p in (answers_file, scores_file):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run earlier phases first.")

    answers = {r["question_id"]: r for r in load_jsonl(answers_file)}
    scored  = load_jsonl(scores_file)

    # --- Judge scores & win rate -----------------------------------------
    dim_sums:   dict[str, dict[str, float]] = {"baseline": defaultdict(float), "dpo": defaultdict(float)}
    dim_counts: dict[str, dict[str, int]]   = {"baseline": defaultdict(int),   "dpo": defaultdict(int)}
    wins = {"baseline": 0, "dpo": 0, "tie": 0}
    n_judged = 0

    for row in scored:
        for model_key in ("baseline", "dpo"):
            for judge_model in args.judge_models:
                sc = row["scores"].get(model_key, {}).get(judge_model, {})
                for d in DIMS:
                    v = sc.get(d)
                    if v is not None:
                        dim_sums[model_key][d]   += float(v)
                        dim_counts[model_key][d] += 1

        def _overall(key: str) -> float | None:
            vals = []
            for jm in args.judge_models:
                sc = row["scores"].get(key, {}).get(jm, {})
                vs = [sc.get(d) for d in DIMS if sc.get(d) is not None]
                if vs:
                    vals.append(sum(vs) / len(vs))
            return sum(vals) / len(vals) if vals else None

        b, d = _overall("baseline"), _overall("dpo")
        if b is not None and d is not None:
            n_judged += 1
            gap = d - b
            if abs(gap) < 0.05:
                wins["tie"] += 1
            elif gap > 0:
                wins["dpo"] += 1
            else:
                wins["baseline"] += 1

    # --- Citation F1 ------------------------------------------------------
    cit_f1: dict[str, list[float]] = {"baseline": [], "dpo": []}
    if CITATION_PARSER_AVAILABLE:
        for row in answers.values():
            ref = row.get("ground_truth", "")
            for key in ("baseline", "dpo"):
                pred = row.get(key, {}).get("text", "")
                if pred:
                    cit_f1[key].append(citation_f1(pred, ref))

    # --- BERTScore --------------------------------------------------------
    bert_f1: dict[str, float | None] = {"baseline": None, "dpo": None}
    if BERT_SCORE_AVAILABLE:
        qids = list(answers)
        refs = [answers[q].get("ground_truth", "") for q in qids]
        for key in ("baseline", "dpo"):
            cands = [answers[q].get(key, {}).get("text", "") for q in qids]
            _, _, F = _bert_score(cands, refs, lang="en", verbose=False)
            bert_f1[key] = float(F.mean())

    # --- Print -----------------------------------------------------------
    W = 56
    print(f"\n{'=' * W}")
    print("  DPO Evaluation Report")
    print(f"{'=' * W}")
    print(f"  {'Metric':<28}{'Baseline':>8}{'DPO':>8}{'Δ':>8}")
    print(f"  {'-' * (W - 2)}")

    def _mean(key: str, d: str) -> float:
        c = dim_counts[key][d]
        return dim_sums[key][d] / c if c else float("nan")

    for d in DIMS:
        b, dv = _mean("baseline", d), _mean("dpo", d)
        print(f"  Judge {d:<22}{b:>8.3f}{dv:>8.3f}{dv - b:>+8.3f}")

    b_tot = sum(dim_sums["baseline"].values())
    d_tot = sum(dim_sums["dpo"].values())
    b_cnt = sum(dim_counts["baseline"].values())
    d_cnt = sum(dim_counts["dpo"].values())
    b_all = b_tot / b_cnt if b_cnt else float("nan")
    d_all = d_tot / d_cnt if d_cnt else float("nan")
    print(f"  {'-' * (W - 2)}")
    print(f"  {'Judge overall (mean)':<28}{b_all:>8.3f}{d_all:>8.3f}{d_all - b_all:>+8.3f}")
    print(f"  {'-' * (W - 2)}")

    if n_judged:
        print(f"  {'Win rate (DPO preferred)':<36}{wins['dpo'] / n_judged:>8.1%}")
        print(f"  {'Win rate (baseline preferred)':<36}{wins['baseline'] / n_judged:>8.1%}")
        print(f"  {'Tie rate':<36}{wins['tie'] / n_judged:>8.1%}")
        print(f"  {'Questions judged':<36}{n_judged:>8d}")
        print(f"  {'-' * (W - 2)}")

    if CITATION_PARSER_AVAILABLE:
        for key in ("baseline", "dpo"):
            if cit_f1[key]:
                avg = sum(cit_f1[key]) / len(cit_f1[key])
                print(f"  {'Citation F1 (' + key + ')':<28}{avg:>8.3f}")
        if cit_f1["baseline"] and cit_f1["dpo"]:
            delta = sum(cit_f1["dpo"]) / len(cit_f1["dpo"]) - sum(cit_f1["baseline"]) / len(cit_f1["baseline"])
            print(f"  {'Citation F1 delta':<44}{delta:>+8.3f}")
        print(f"  {'-' * (W - 2)}")

    if BERT_SCORE_AVAILABLE:
        for key in ("baseline", "dpo"):
            if bert_f1[key] is not None:
                print(f"  {'BERTScore F1 (' + key + ')':<28}{bert_f1[key]:>8.3f}")
        if bert_f1["baseline"] is not None and bert_f1["dpo"] is not None:
            delta = bert_f1["dpo"] - bert_f1["baseline"]
            print(f"  {'BERTScore F1 delta':<44}{delta:>+8.3f}")

    print(f"{'=' * W}\n")

    if not CITATION_PARSER_AVAILABLE:
        print("Note: citation_parser not found — citation F1 skipped.")
    if not BERT_SCORE_AVAILABLE:
        print("Note: bert_score not installed — BERTScore skipped.  pip install bert-score")

    # --- Save JSON --------------------------------------------------------
    report = {
        "judge_scores": {
            k: {d: (_mean(k, d) if dim_counts[k][d] else None) for d in DIMS}
            for k in ("baseline", "dpo")
        },
        "win_rate": {
            k: (wins[k] / n_judged if n_judged else None)
            for k in ("baseline", "dpo", "tie")
        },
        "n_judged": n_judged,
        "citation_f1": {
            k: (sum(cit_f1[k]) / len(cit_f1[k]) if cit_f1[k] else None)
            for k in ("baseline", "dpo")
        },
        "bertscore_f1": bert_f1,
    }
    out = args.output_dir / "eval_report.json"
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", choices=["generate", "judge", "report", "all"], default="all")
    parser.add_argument("--baseline_model", default="Qwen3.5-4B",
                        help="Model directory name under $SCRATCH/models/")
    parser.add_argument("--dpo_model_path", type=Path, default=DEFAULT_DPO_MODEL_PATH,
                        help="Path to the DPO-trained model checkpoint.")
    parser.add_argument("--judge_models", nargs="+", default=ALL_JUDGE_MODELS,
                        help="vLLM-served judge model names (space-separated).")
    parser.add_argument("--judge_api_base", default="https://api.swissai.svc.cscs.ch/v1",
                        help="Base URL of the judge API (default: SwissAI).")
    parser.add_argument("--models", choices=["all", "baseline", "dpo"], default="all",
                        help="Which models to run in generate phase. Use 'baseline' once, "
                             "then 'dpo' for each variant (requires --baseline_answers).")
    parser.add_argument("--baseline_answers", type=Path, default=None,
                        help="Pre-computed baseline eval_answers.jsonl to reuse (--models dpo only).")
    parser.add_argument("--existing_scores", type=Path,
                        default=Path("output/eval_baseline/llama/eval_judge_scores.jsonl"),
                        help="Path to an existing eval_judge_scores.jsonl to seed from (skips already-judged entries).")
    parser.add_argument("--n_questions", type=int, default=100,
                        help="Number of held-out questions to sample (default: 100). Pass -1 for all.")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load answerer models in 4-bit quantization.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", type=Path, default=BASE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.phase in ("generate", "all"):
        generate_phase(args)
    if args.phase in ("judge", "all"):
        judge_phase(args)
    if args.phase in ("report", "all"):
        report_phase(args)


if __name__ == "__main__":
    main()
