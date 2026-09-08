"""
SFT fine-tuning on chosen regulatory answers (supervised baseline).

Trains on {prompt → chosen_answer} pairs from dpo_corruption_pairs.jsonl using
standard causal LM loss. Serves as the simplest alignment baseline — no
preference signal, just MLE on preferred responses. Questions with multiple
corruption variants are deduplicated so each question is seen once.

Usage:
    python train_sft.py --model Llama-3.1-8B-Instruct --use_lora --load_in_4bit

    python train_sft.py --data output/dpo_corruption_pairs.jsonl \\
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
from trl import SFTConfig, SFTTrainer

try:
    from peft import LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

RECORDS_FILE = Path("output/judge_results_train_api.jsonl")


class _PaddingCollator:
    """Pads pre-tokenized examples; labels are already masked in the dataset."""

    def __init__(self, pad_token_id: int):
        self._pad_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids_out, attn_out, labels_out = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lbls = list(f["labels"])
            pad_len = max_len - len(ids)
            input_ids_out.append(ids + [self._pad_id] * pad_len)
            attn_out.append([1] * len(ids) + [0] * pad_len)
            labels_out.append(lbls + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids_out),
            "attention_mask": torch.tensor(attn_out),
            "labels": torch.tensor(labels_out),
        }

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

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    qconfig = None
    if load_in_4bit:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        qconfig = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True,
        torch_dtype=compute_dtype, device_map="auto",
        quantization_config=qconfig,
    )

    if (load_in_4bit or load_in_8bit) and PEFT_AVAILABLE:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def load_sft_dataset(
    paths: list[str],
    corruption_variants: list[str] | None,
    max_examples: int | None,
    seed: int,
    tokenizer,
    max_length: int,
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

    # One chosen text per question — deduplicate so the same ground truth isn't
    # seen multiple times just because the question had several corruption types.
    seen: set[str] = set()
    unique_rows = []
    for row in rows:
        qid = row["question_id"]
        if qid not in seen:
            seen.add(qid)
            unique_rows.append(row)
    rows = unique_rows

    if max_examples is not None and len(rows) > max_examples:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_examples]

    # Tokenize prompt and answer separately so the boundary is exact.
    # Searching for a response-template substring in the merged token IDs is
    # unreliable because BPE merges tokens across the boundary differently
    # than when the substring is encoded in isolation.
    all_input_ids, all_labels = [], []
    skipped = 0
    for row in rows:
        meta = meta_by_qid.get(row["question_id"], {})
        prompt = PROMPT_TEMPLATE.format(
            legal_act=meta.get("legal_act", ""),
            topic=meta.get("topic", ""),
            subject_matter=meta.get("subject_matter", ""),
            question=row["question"],
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        answer_ids = tokenizer.encode(row["chosen"]["text"], add_special_tokens=False)
        answer_ids = answer_ids + [tokenizer.eos_token_id]

        if len(prompt_ids) >= max_length:
            # Prompt alone fills the context: no answer tokens survive the truncation,
            # so labels would be all -100, causing NaN loss. Drop the example.
            skipped += 1
            continue

        ids = (prompt_ids + answer_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + list(answer_ids))[:max_length]

        all_input_ids.append(ids)
        all_labels.append(labels)

    if skipped:
        print(f"WARNING: skipped {skipped} examples where prompt alone >= max_length ({max_length}). "
              "Increase --max_length or check your data.")
    if not all_input_ids:
        raise ValueError("All examples were skipped. Increase --max_length.")

    return Dataset.from_dict({"input_ids": all_input_ids, "labels": all_labels})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", nargs="+", default=["output/dpo_corruption_pairs.jsonl"],
    )
    parser.add_argument("--corruption_variants", nargs="+", default=None)
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", default="output/sft_checkpoints")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.use_lora and not PEFT_AVAILABLE:
        raise ImportError("peft is required for --use_lora. pip install peft")

    # Load model first: tokenizer is needed to pre-tokenize the dataset so we
    # know the exact prompt/answer boundary (BPE tokenization is context-dependent,
    # so searching for a template string in merged token IDs is unreliable).
    model, tokenizer = load_model(args.model, args.load_in_4bit, args.load_in_8bit)

    dataset = load_sft_dataset(
        args.data, args.corruption_variants, args.max_examples, args.seed,
        tokenizer, args.max_length,
    )
    split = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    print(f"Train: {len(split['train'])}  Eval: {len(split['test'])}")

    steps_per_epoch = max(1, len(split["train"]) // (args.batch_size * args.grad_accum))
    eval_steps = max(1, steps_per_epoch // 2)

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )

    collator = _PaddingCollator(tokenizer.pad_token_id)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=args.seed,
        packing=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved SFT model → {args.output_dir}")


if __name__ == "__main__":
    main()
