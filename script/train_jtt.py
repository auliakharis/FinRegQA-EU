"""
Just Train Twice (JTT) for regulatory answer alignment.

JTT (Liu et al. 2021) first identifies "hard" training pairs where a stage-1
model still assigns higher log-probability to the rejected answer than the
chosen, then retrains DPO with those pairs repeated lambda_up times so the
optimizer focuses more on the failure cases.

Two-phase workflow:

  identify  Load the stage-1 model checkpoint and score every training pair by
            computing log P(chosen | prompt) − log P(rejected | prompt).
            Pairs where the margin ≤ 0 (model prefers rejected) are hard.
            Saves output/jtt_hard_ids.json.

  train     Run DPO on the full training set, inserting each hard pair
            lambda_up extra times. Requires output/jtt_hard_ids.json.

Usage:
    # Full pipeline in one go
    python train_jtt.py --phase all \\
        --stage1_model_path /cluster/scratch/you/dpo_checkpoints \\
        --model Qwen3.5-4B --use_lora --load_in_4bit

    # Split across nodes (identify on GPU, train on GPU)
    python train_jtt.py --phase identify \\
        --stage1_model_path /cluster/scratch/you/dpo_checkpoints \\
        --load_in_4bit
    python train_jtt.py --phase train --model Qwen3.5-4B --use_lora --load_in_4bit
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

try:
    from peft import LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

RECORDS_FILE   = Path("output/judge_results_train_api.jsonl")
HARD_IDS_FILE  = Path("output/jtt_hard_ids.json")

PROMPT_TEMPLATE = """\
You are an expert in EU financial regulation, with deep knowledge of EBA and \
ESMA guidelines, technical standards, and related directives and regulations.

Answer the following regulatory question. Base your answer on the actual \
content of the legal act identified below and support every substantive \
claim with a specific citation in the form [Source, Article/Paragraph]. \
Write 100-400 words of prose, matching the style of official EBA/ESMA Q&A \
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

def load_meta_by_qid(path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                meta[rec["question_id"]] = rec.get("meta", {})
    return meta


def load_pairs(paths: list[str], corruption_variants: list[str] | None) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                variant = row.get("rejected", {}).get("corruption_variant", "original")
                if corruption_variants and variant not in corruption_variants:
                    continue
                rows.append(row)
    if not rows:
        raise ValueError("No rows matched the given filters.")
    return rows


def load_model_for_scoring(model_path: str | Path, load_in_4bit: bool):
    """Load a model for log-prob scoring (inference only)."""
    model_path = str(model_path)
    print(f"Loading stage-1 model from: {model_path}")

    adapter_cfg_path = Path(model_path) / "adapter_config.json"
    if adapter_cfg_path.exists():
        with adapter_cfg_path.open() as f:
            base_path = json.load(f)["base_model_name_or_path"]
        print(f"  LoRA adapter detected — base: {base_path}")
    else:
        base_path = model_path

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

    model = AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True,
        torch_dtype=torch.float16, device_map="auto",
        quantization_config=qconfig,
    )

    if adapter_cfg_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def load_model_for_training(model_name: str, load_in_4bit: bool, load_in_8bit: bool):
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH environment variable is not set.")
    model_path = os.path.join(scratch, "models", model_name)
    print(f"Loading training model from: {model_path}")

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

    if (load_in_4bit or load_in_8bit) and PEFT_AVAILABLE:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Log-prob scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def completion_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> float:
    """Sum of log P(token | context) over completion tokens."""
    full = tokenizer(
        prompt + completion,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    full_ids = full["input_ids"].to(model.device)
    prompt_len = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"].shape[1]

    logits = model(full_ids).logits[0, :-1]          # [seq-1, vocab]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    labels = full_ids[0, 1:]                          # [seq-1]
    token_lps = log_probs[torch.arange(len(labels)), labels]
    # Sum only over completion tokens
    return token_lps[max(prompt_len - 1, 0):].sum().item()


# ---------------------------------------------------------------------------
# Phase 1 · identify hard examples
# ---------------------------------------------------------------------------

def identify_phase(args) -> None:
    rows = load_pairs(args.data, args.corruption_variants)
    meta_by_qid = load_meta_by_qid(RECORDS_FILE)

    model, tokenizer = load_model_for_scoring(args.stage1_model_path, args.load_in_4bit)

    hard_ids: list[str] = []
    margins: list[float] = []

    for i, row in enumerate(rows):
        meta   = meta_by_qid.get(row["question_id"], {})
        prompt = PROMPT_TEMPLATE.format(
            legal_act=meta.get("legal_act", ""),
            topic=meta.get("topic", ""),
            subject_matter=meta.get("subject_matter", ""),
            question=row["question"],
        )
        lp_chosen   = completion_logprob(model, tokenizer, prompt, row["chosen"]["text"],   args.max_length)
        lp_rejected = completion_logprob(model, tokenizer, prompt, row["rejected"]["text"], args.max_length)
        margin = lp_chosen - lp_rejected
        margins.append(margin)

        if margin <= 0:
            # Include the corruption_variant in the key so the same question
            # with different corruptions is counted separately.
            key = f"{row['question_id']}::{row['rejected'].get('corruption_variant', '')}"
            hard_ids.append(key)

        if (i + 1) % 100 == 0:
            n_hard = sum(1 for m in margins if m <= 0)
            print(f"  {i+1}/{len(rows)}  hard so far: {n_hard}  "
                  f"mean margin: {sum(margins)/len(margins):.3f}")

    hard_set = set(hard_ids)
    print(f"\nIdentified {len(hard_set)} hard pairs out of {len(rows)} "
          f"({100*len(hard_set)/len(rows):.1f}%)")

    HARD_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HARD_IDS_FILE.open("w") as f:
        json.dump({"hard_ids": list(hard_set), "n_total": len(rows)}, f, indent=2)
    print(f"Saved → {HARD_IDS_FILE}")


# ---------------------------------------------------------------------------
# Phase 2 · train DPO with upweighted hard examples
# ---------------------------------------------------------------------------

def build_dpo_dataset(
    rows: list[dict],
    meta_by_qid: dict[str, dict],
    hard_set: set[str],
    lambda_up: int,
    max_examples: int | None,
    seed: int,
) -> Dataset:
    if max_examples is not None and len(rows) > max_examples:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_examples]

    prompts, chosen_texts, rejected_texts = [], [], []
    for row in rows:
        key = f"{row['question_id']}::{row['rejected'].get('corruption_variant', '')}"
        meta   = meta_by_qid.get(row["question_id"], {})
        prompt = PROMPT_TEMPLATE.format(
            legal_act=meta.get("legal_act", ""),
            topic=meta.get("topic", ""),
            subject_matter=meta.get("subject_matter", ""),
            question=row["question"],
        )
        repeat = lambda_up if key in hard_set else 1
        for _ in range(repeat):
            prompts.append(prompt)
            chosen_texts.append(row["chosen"]["text"])
            rejected_texts.append(row["rejected"]["text"])

    return Dataset.from_dict({
        "prompt": prompts, "chosen": chosen_texts, "rejected": rejected_texts,
    })


def train_phase(args) -> None:
    if not HARD_IDS_FILE.exists():
        raise FileNotFoundError(
            f"{HARD_IDS_FILE} not found — run --phase identify first."
        )
    with HARD_IDS_FILE.open() as f:
        hard_data = json.load(f)
    hard_set = set(hard_data["hard_ids"])
    print(f"Loaded {len(hard_set)} hard pair IDs (lambda_up={args.lambda_up})")

    rows        = load_pairs(args.data, args.corruption_variants)
    meta_by_qid = load_meta_by_qid(RECORDS_FILE)

    dataset = build_dpo_dataset(rows, meta_by_qid, hard_set, args.lambda_up,
                                args.max_examples, args.seed)
    split   = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    n_base  = len(rows) if args.max_examples is None else min(len(rows), args.max_examples)
    print(f"Train examples: {len(split['train'])}  "
          f"(base {n_base} × upweight → {len(split['train'])+len(split['test'])} total)  "
          f"Eval: {len(split['test'])}")

    model, tokenizer = load_model_for_training(args.model, args.load_in_4bit, args.load_in_8bit)

    ref_model = None
    if not args.use_lora:
        ref_model, _ = load_model_for_training(
            args.ref_model or args.model, args.load_in_4bit, args.load_in_8bit
        )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        beta=args.beta,
        max_length=args.max_length,
        loss_type=[args.loss_type],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved JTT-DPO model → {args.output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", choices=["identify", "train", "all"], default="all")

    # Shared data args
    parser.add_argument("--data", nargs="+", default=["output/dpo_corruption_pairs.jsonl"])
    parser.add_argument("--corruption_variants", nargs="+", default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    # Identify phase
    parser.add_argument("--stage1_model_path", type=Path, default=None,
                        help="Path to a trained checkpoint to use as the stage-1 model "
                             "(e.g. output/dpo_checkpoints). Required for --phase identify.")

    # Train phase
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct",
                        help="Base model for DPO retraining (under $SCRATCH/models/).")
    parser.add_argument("--ref_model", default=None)
    parser.add_argument("--output_dir", default="output/jtt_checkpoints")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--lambda_up", type=int, default=4,
                        help="How many extra times to repeat each hard example (JTT upweight factor).")
    parser.add_argument("--loss_type", default="sigmoid",
                        choices=["sigmoid", "ipo", "robust", "hinge",
                                 "apo_zero", "apo_down", "nca_pair"],
                        help="DPO loss variant for the upweighted retraining step.")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap on base pairs before upweighting.")
    args = parser.parse_args()

    if args.phase in ("identify", "all"):
        if args.stage1_model_path is None:
            parser.error("--stage1_model_path is required for the identify phase.")
        identify_phase(args)

    if args.phase in ("train", "all"):
        if args.use_lora and not PEFT_AVAILABLE:
            raise ImportError("peft is required for --use_lora. pip install peft")
        train_phase(args)


if __name__ == "__main__":
    main()
