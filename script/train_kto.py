"""
KTO fine-tuning on regulatory answer preferences.

Kahneman-Tversky Optimization (Ethayarajh et al. 2024) converts chosen/rejected
pairs into individual labeled examples: the chosen response is labeled as
desirable (True) and the rejected as undesirable (False). The KTO loss does not
require paired comparisons at training time, making it more robust to preference
noise and applicable when only one-sided feedback is available.

Each DPO pair → 2 KTO examples:
  chosen   → completion with label=True  (desirable)
  rejected → completion with label=False (undesirable)

Usage:
    python train_kto.py --model Llama-3.1-8B-Instruct --use_lora --load_in_4bit

    python train_kto.py --data output/dpo_corruption_pairs.jsonl \\
        --corruption_variants law_swap article_swap \\
        --model Qwen3.5-4B --epochs 1 --use_lora
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import KTOConfig, KTOTrainer

try:
    from peft import LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

RECORDS_FILE = Path("output/judge_results_train_api.jsonl")

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

    if (load_in_4bit or load_in_8bit) and PEFT_AVAILABLE:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def load_kto_dataset(
    paths: list[str],
    corruption_variants: list[str] | None,
    max_pairs: int | None,
    seed: int,
) -> Dataset:
    meta_by_qid: dict[str, dict] = {}
    with RECORDS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                meta_by_qid[rec["question_id"]] = rec.get("meta", {})

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

    if max_pairs is not None and len(rows) > max_pairs:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_pairs]

    prompts, completions, labels = [], [], []
    for row in rows:
        meta = meta_by_qid.get(row["question_id"], {})
        prompt = PROMPT_TEMPLATE.format(
            legal_act=meta.get("legal_act", ""),
            topic=meta.get("topic", ""),
            subject_matter=meta.get("subject_matter", ""),
            question=row["question"],
        )
        prompts.append(prompt)
        completions.append(row["chosen"]["text"])
        labels.append(True)

        prompts.append(prompt)
        completions.append(row["rejected"]["text"])
        labels.append(False)

    return Dataset.from_dict({"prompt": prompts, "completion": completions, "label": labels})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", nargs="+", default=["output/dpo_corruption_pairs.jsonl"],
    )
    parser.add_argument("--corruption_variants", nargs="+", default=None)
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", default="output/kto_checkpoints")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KTO beta (KL penalty). KTO typically uses a higher beta than DPO.")
    parser.add_argument("--desirable_weight", type=float, default=1.0,
                        help="Loss weight for desirable (chosen) examples.")
    parser.add_argument("--undesirable_weight", type=float, default=1.0,
                        help="Loss weight for undesirable (rejected) examples. "
                             "Increase to penalize rejected outputs more heavily.")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max DPO pairs to use (each pair becomes 2 KTO examples).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.use_lora and not PEFT_AVAILABLE:
        raise ImportError("peft is required for --use_lora. pip install peft")

    dataset = load_kto_dataset(args.data, args.corruption_variants, args.max_pairs, args.seed)
    split = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    n_train_pairs = sum(1 for x in split["train"]["label"] if x)
    print(f"Train: {len(split['train'])} examples ({n_train_pairs} pairs)  "
          f"Eval: {len(split['test'])} examples")

    model, tokenizer = load_model(args.model, args.load_in_4bit, args.load_in_8bit)

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )

    training_args = KTOConfig(
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
        seed=args.seed,
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
        max_length=args.max_length,
    )

    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved KTO model → {args.output_dir}")


if __name__ == "__main__":
    main()
