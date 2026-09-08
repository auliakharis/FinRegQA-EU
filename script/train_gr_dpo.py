"""
GroupDRO training for regulatory answer alignment.

Applies Distributionally Robust Optimization (Sagawa et al. 2020) grouped by
corruption variant (law_swap, article_swap, combined). Instead of minimising
the average DPO loss across all corruption types, the training upsamples groups
that the model currently finds hardest, so the final model is robust across
all corruption types rather than just performing well on average.

Algorithm (epoch-level GroupDRO):
  1. Initialise uniform group weights w_g = 1/G.
  2. For each epoch:
       a. Build a training dataset whose group proportions match w_g (by
          repeating examples from high-weight groups).
       b. Train DPO for one epoch on this dataset.
       c. Evaluate per-group DPO margin on the full training set:
              margin_g = mean[ log P(chosen|prompt) - log P(rejected|prompt) ]
          Lower margin = model is struggling with this group.
       d. Update weights: w_g ← w_g · exp(η · (−margin_g)),  then renormalize.
  3. Save the final model.

Note: the DPOTrainer optimizer is re-instantiated each epoch (standard
approximation when using HuggingFace trainers). The reference model is fixed
to the initial pre-trained weights and stays frozen throughout.

Usage:
    python train_doro.py --model Llama-3.1-8B-Instruct --use_lora --load_in_4bit

    python train_doro.py --data output/dpo_corruption_pairs.jsonl \\
        --model Qwen3.5-4B --epochs 3 --eta 0.1 --use_lora --load_in_4bit
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

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


# ---------------------------------------------------------------------------
# Model loading
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

    if for_training and (load_in_4bit or load_in_8bit) and PEFT_AVAILABLE:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading
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


def load_pairs(
    paths: list[str],
    corruption_variants: list[str] | None,
    max_examples: int | None,
    seed: int,
) -> list[dict]:
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
    if max_examples is not None and len(rows) > max_examples:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_examples]
    return rows


def build_weighted_dataset(
    rows: list[dict],
    meta_by_qid: dict[str, dict],
    group_weights: dict[str, float],
    target_size: int,
    seed: int,
) -> Dataset:
    """
    Sample `target_size` examples with replacement, proportional to group weights.
    Each example belongs to the group defined by its corruption_variant.
    """
    # Group rows by corruption_variant
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        g = row["rejected"].get("corruption_variant", "unknown")
        by_group[g].append(row)

    # How many samples to draw from each group
    total_w = sum(group_weights.values())
    samples_per_group = {
        g: max(1, round(group_weights.get(g, 0) / total_w * target_size))
        for g in by_group
    }

    rng = random.Random(seed)
    sampled: list[dict] = []
    for g, n in samples_per_group.items():
        pool = by_group[g]
        # Sample with replacement if n > len(pool), else without
        if n <= len(pool):
            sampled.extend(rng.sample(pool, n))
        else:
            sampled.extend(rng.choices(pool, k=n))

    rng.shuffle(sampled)

    prompts, chosen_texts, rejected_texts = [], [], []
    for row in sampled:
        meta   = meta_by_qid.get(row["question_id"], {})
        prompt = PROMPT_TEMPLATE.format(
            legal_act=meta.get("legal_act", ""),
            topic=meta.get("topic", ""),
            subject_matter=meta.get("subject_matter", ""),
            question=row["question"],
        )
        prompts.append(prompt)
        chosen_texts.append(row["chosen"]["text"])
        rejected_texts.append(row["rejected"]["text"])

    return Dataset.from_dict({
        "prompt": prompts, "chosen": chosen_texts, "rejected": rejected_texts,
    })


# ---------------------------------------------------------------------------
# Per-group margin evaluation (used to update group weights each epoch)
# ---------------------------------------------------------------------------

@torch.no_grad()
def completion_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> float:
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

    logits    = model(full_ids).logits[0, :-1]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    labels    = full_ids[0, 1:]
    token_lps = log_probs[torch.arange(len(labels)), labels]
    return token_lps[max(prompt_len - 1, 0):].sum().item()


def compute_group_margins(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: list[dict],
    meta_by_qid: dict[str, dict],
    max_length: int,
    sample_size: int | None,
    seed: int,
) -> dict[str, float]:
    """
    Compute mean log-margin (log P_chosen - log P_rejected) per corruption group.
    A lower margin means the model struggles more with that group.
    """
    eval_rows = rows
    if sample_size is not None and len(rows) > sample_size:
        rng = random.Random(seed)
        eval_rows = rng.sample(rows, sample_size)

    group_margins: dict[str, list[float]] = defaultdict(list)
    model.eval()

    for i, row in enumerate(eval_rows):
        g      = row["rejected"].get("corruption_variant", "unknown")
        meta   = meta_by_qid.get(row["question_id"], {})
        prompt = PROMPT_TEMPLATE.format(
            legal_act=meta.get("legal_act", ""),
            topic=meta.get("topic", ""),
            subject_matter=meta.get("subject_matter", ""),
            question=row["question"],
        )
        lp_c = completion_logprob(model, tokenizer, prompt, row["chosen"]["text"],   max_length)
        lp_r = completion_logprob(model, tokenizer, prompt, row["rejected"]["text"], max_length)
        group_margins[g].append(lp_c - lp_r)

        if (i + 1) % 200 == 0:
            print(f"    margin eval: {i+1}/{len(eval_rows)}")

    return {g: sum(ms) / len(ms) for g, ms in group_margins.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", nargs="+", default=["output/dpo_corruption_pairs.jsonl"])
    parser.add_argument("--corruption_variants", nargs="+", default=None,
                        help="Restrict to these groups. Default: all variants present in --data.")
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--ref_model", default=None)
    parser.add_argument("--output_dir", default="output/doro_checkpoints")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--eta", type=float, default=0.1,
                        help="GroupDRO step size for updating group weights.")
    parser.add_argument("--min_group_weight", type=float, default=0.05,
                        help="Minimum weight for any group after renormalisation. "
                             "Prevents collapse to a single group. Set 0 to disable.")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--loss_type", default="sigmoid",
                        choices=["sigmoid", "ipo", "robust", "hinge",
                                 "apo_zero", "apo_down", "nca_pair"])
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of outer GroupDRO epochs (each trains DPO for 1 epoch).")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap on total training pairs per outer epoch (before reweighting).")
    parser.add_argument("--margin_eval_sample", type=int, default=None,
                        help="Number of pairs to use for per-group margin evaluation each epoch. "
                             "Default: use all training pairs (slower but more accurate).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.use_lora and not PEFT_AVAILABLE:
        raise ImportError("peft is required for --use_lora. pip install peft")

    meta_by_qid = load_meta_by_qid(RECORDS_FILE)
    all_rows    = load_pairs(args.data, args.corruption_variants, args.max_examples, args.seed)

    # Discover groups from data
    all_groups = sorted({r["rejected"].get("corruption_variant", "unknown") for r in all_rows})
    print(f"Groups found: {all_groups}")

    # Uniform initial weights
    group_weights: dict[str, float] = {g: 1.0 / len(all_groups) for g in all_groups}

    # Load models — ref_model stays frozen throughout all epochs
    model, tokenizer = load_model(args.model, args.load_in_4bit, args.load_in_8bit,
                                  for_training=True)

    ref_model = None
    if not args.use_lora:
        ref_model, _ = load_model(
            args.ref_model or args.model, args.load_in_4bit, args.load_in_8bit,
            for_training=False,
        )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )

    weight_log: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"  Outer epoch {epoch}/{args.epochs}")
        fmt = "  ".join(f"{g}: {w:.4f}" for g, w in sorted(group_weights.items()))
        print(f"  Group weights: {fmt}")
        print(f"{'='*60}")

        dataset = build_weighted_dataset(
            all_rows, meta_by_qid, group_weights,
            target_size=len(all_rows),
            seed=args.seed + epoch,
        )
        split = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
        print(f"  Dataset: {len(split['train'])} train  {len(split['test'])} eval")

        steps_per_epoch = max(1, len(split["train"]) // (args.batch_size * args.grad_accum))
        eval_steps = max(1, steps_per_epoch // 2)

        epoch_output = os.path.join(args.output_dir, f"epoch_{epoch}")
        _dpo_kwargs: dict = dict(beta=args.beta, max_length=args.max_length, loss_type=args.loss_type)
        if "max_prompt_length" in inspect.signature(DPOConfig.__init__).parameters:
            _dpo_kwargs["max_prompt_length"] = args.max_prompt_length
        training_args = DPOConfig(
            output_dir=epoch_output,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            num_train_epochs=1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="no",          # save only at the end
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to=[],
            remove_unused_columns=False,
            seed=args.seed,
            **_dpo_kwargs,
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
        # Carry the trained model (with LoRA weights) forward to the next epoch.
        # Without this, DPOTrainer wraps a fresh base model each epoch and discards
        # the previous epoch's LoRA adapter.
        model = trainer.model
        peft_config = None  # model is now a PeftModel; don't re-wrap in later epochs

        # ---- Compute per-group margins ----
        print(f"\n  Computing per-group margins for epoch {epoch} weight update...")
        group_margins = compute_group_margins(
            model, tokenizer, all_rows, meta_by_qid,
            args.max_length,
            args.margin_eval_sample,
            args.seed + epoch,
        )

        print("  Per-group margins:")
        for g, m in sorted(group_margins.items()):
            print(f"    {g}: {m:+.4f}")

        # ---- Update group weights (exponential reweighting, log-space) ----
        # Log-space update avoids overflow and is numerically equivalent to
        # w_g *= exp(eta * -margin), but stable under cumulative updates.
        log_weights = {
            g: math.log(group_weights[g]) + args.eta * (-group_margins.get(g, 0.0))
            for g in all_groups
        }
        max_lw = max(log_weights.values())
        unnorm = {g: math.exp(lw - max_lw) for g, lw in log_weights.items()}
        total = sum(unnorm.values())
        group_weights = {g: v / total for g, v in unnorm.items()}

        # Floor: prevent any group from collapsing to ~0, which would cause
        # the weighted dataset to stop sampling it entirely.
        if args.min_group_weight > 0:
            group_weights = {g: max(w, args.min_group_weight) for g, w in group_weights.items()}
            total = sum(group_weights.values())
            group_weights = {g: w / total for g, w in group_weights.items()}

        weight_log.append({"epoch": epoch, "weights": dict(group_weights), "margins": dict(group_margins)})

    # ---- Save final model and weight history ----
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log_path = Path(args.output_dir) / "doro_weight_log.json"
    with log_path.open("w") as f:
        json.dump(weight_log, f, indent=2)

    print(f"\nSaved GroupDRO model → {final_dir}")
    print(f"Weight log → {log_path}")

    # Print final group weights for easy inspection
    print("\nFinal group weights:")
    for g, w in sorted(group_weights.items()):
        print(f"  {g}: {w:.4f}")


if __name__ == "__main__":
    main()
