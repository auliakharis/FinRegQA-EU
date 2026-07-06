"""
DPO fine-tuning between preferred and less-preferred regulatory answers.

Trains a policy model with Direct Preference Optimization (TRL's DPOTrainer,
trl==0.8.6 API) on chosen/rejected pairs produced by generate_dpo_corruptions.py
(output/dpo_corruption_pairs.jsonl), or any JSONL with the same
{"question_id", "question", "chosen": {"text": ...}, "rejected": {"text": ...}}
schema.

Model loading mirrors judge.py's convention ($SCRATCH/models/<name>, optional
4-bit/8-bit quantization). Use --use_lora to train a LoRA adapter instead of
the full model (requires `pip install peft`) — recommended for anything above
a few billion parameters on a single GPU.

The prompt template below approximates (but does not byte-for-byte reproduce)
judge_api.py's ANSWERER_PROMPT used to generate these answers: it reuses the
LEGAL_ACT/TOPIC/SUBJECT_MATTER context fields (looked up from
output/judge_results_train_api.jsonl by question_id) but omits the BACKGROUND
field, which isn't persisted in that file. If you need an exact-match prompt,
join against the original data/splits/*.jsonl source instead.

Usage:
    python train_dpo.py --data output/dpo_corruption_pairs.jsonl \\
        --model Llama-3.1-8B-Instruct --use_lora --load_in_4bit

    python train_dpo.py --data output/dpo_corruption_pairs.jsonl \\
        --corruption_variants law_swap article_swap combined \\
        --model Qwen3.5-4B --epochs 1
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
from trl import DPOTrainer

try:
    from trl import DPOConfig   # TRL >= 0.9 moved beta/max_length here
    _DPO_CONFIG_API = True
except ImportError:
    _DPO_CONFIG_API = False     # fall back to TRL 0.8.x inline args

try:
    from peft import LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

RECORDS_FILE = Path("output/judge_results_train_api.jsonl")

# Approximates judge_api.py's ANSWERER_PROMPT (see module docstring for caveats).
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
# Model loading (mirrors judge.py's load_model)
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    for_training: bool = True,
):
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

    # Required for LoRA training on a quantized model: enables gradient flow
    # through frozen quantized layers and turns on gradient checkpointing.
    if for_training and (load_in_4bit or load_in_8bit) and PEFT_AVAILABLE:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_meta_by_question_id(path: Path) -> dict[str, dict]:
    meta_by_qid = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                meta_by_qid[rec["question_id"]] = rec.get("meta", {})
    return meta_by_qid


def load_dpo_dataset(
    paths: list[str],
    corruption_variants: list[str] | None,
    max_examples: int | None,
    seed: int,
) -> Dataset:
    meta_by_qid = load_meta_by_question_id(RECORDS_FILE)

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
        raise ValueError("No DPO rows matched the given --data / --corruption_variants filters.")

    if max_examples is not None and len(rows) > max_examples:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_examples]

    prompts, chosen, rejected = [], [], []
    for row in rows:
        meta = meta_by_qid.get(row["question_id"], {})
        prompt = PROMPT_TEMPLATE.format(
            legal_act=meta.get("legal_act", ""),
            topic=meta.get("topic", ""),
            subject_matter=meta.get("subject_matter", ""),
            question=row["question"],
        )
        prompts.append(prompt)
        chosen.append(row["chosen"]["text"])
        rejected.append(row["rejected"]["text"])

    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", nargs="+", default=["output/dpo_corruption_pairs.jsonl"],
        help="One or more JSONL files with {question_id, question, chosen, rejected} rows.",
    )
    parser.add_argument(
        "--corruption_variants", nargs="+", default=None,
        help="Restrict training to specific rejected.corruption_variant values "
             "(e.g. law_swap article_swap combined). Default: use every row in --data.",
    )
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct",
                         help="Model directory name under $SCRATCH/models/")
    parser.add_argument("--ref_model", default=None,
                         help="Reference model name for full fine-tuning (default: same as "
                              "--model, loaded as a frozen copy). Ignored when --use_lora is set.")
    parser.add_argument("--output_dir", default="output/dpo_checkpoints")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true",
                         help="Train a LoRA adapter instead of the full model (requires peft).")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", nargs="+",
                         default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--beta", type=float, default=0.05,
                         help="DPO beta (KL penalty). Lower values (0.01-0.05) are better "
                              "for off-policy data where chosen/rejected come from a different "
                              "model than the one being trained.")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.use_lora and not PEFT_AVAILABLE:
        raise ImportError("peft is required for --use_lora. Install with: pip install peft")

    dataset = load_dpo_dataset(args.data, args.corruption_variants, args.max_examples, args.seed)
    split = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    print(f"Train examples: {len(split['train'])}  Eval examples: {len(split['test'])}")

    model, tokenizer = load_model(args.model, args.load_in_4bit, args.load_in_8bit)

    ref_model = None
    if not args.use_lora:
        # DPOTrainer needs an explicit frozen reference model for full fine-tuning.
        # With peft_config it instead derives the reference policy from the
        # frozen base weights underneath the LoRA adapter, so no separate
        # ref_model load is needed (saves a full copy of GPU memory).
        ref_model, _ = load_model(
            args.ref_model or args.model, args.load_in_4bit, args.load_in_8bit,
            for_training=False,  # frozen reference — skip prepare_model_for_kbit_training
        )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )

    _shared_train_kwargs = dict(
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
    )

    if _DPO_CONFIG_API:
        # TRL >= 0.9: DPO-specific params (beta, max_length, max_prompt_length)
        # live in DPOConfig; tokenizer is passed as processing_class.
        training_args = DPOConfig(
            **_shared_train_kwargs,
            beta=args.beta,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
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
    else:
        # TRL 0.8.x legacy API
        training_args = TrainingArguments(**_shared_train_kwargs)
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            beta=args.beta,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            peft_config=peft_config,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved DPO-trained model -> {args.output_dir}")


if __name__ == "__main__":
    main()
