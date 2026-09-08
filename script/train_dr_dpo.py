"""
Distributionally Robust DPO (DR-DPO) fine-tuning on regulatory answer pairs.

Replaces the standard mean aggregation in DPO with the entropic risk measure:

    loss = -beta_1 * log( mean( exp(-per_sample_dpo_loss / beta_1) ) )

This up-weights harder examples (large per-sample loss) relative to the mean,
making the policy more robust to the worst-case examples in each batch.
beta_1 controls the degree of robustness: as beta_1 → ∞ the objective
recovers standard DPO (mean), and as beta_1 → 0 it approaches the max loss.

Trains on chosen/rejected pairs produced by generate_dpo_corruptions.py
(output/dpo_corruption_pairs.jsonl), or any JSONL with the same
{"question_id", "question", "chosen": {"text": ...}, "rejected": {"text": ...}}
schema.

Model loading mirrors judge.py's convention ($SCRATCH/models/<name>, optional
4-bit/8-bit quantization). Use --use_lora to train a LoRA adapter instead of
the full model (requires `pip install peft`) — recommended for anything above
a few billion parameters on a single GPU.

Usage:
    python train_dr_dpo.py --data output/dpo_corruption_pairs.jsonl \\
        --model Llama-3.1-8B-Instruct --use_lora --load_in_4bit --beta_1 1.0

    python train_dr_dpo.py --data output/dpo_corruption_pairs.jsonl \\
        --corruption_variants law_swap article_swap combined \\
        --model Qwen3.5-4B --epochs 1 --beta_1 0.5
"""

from __future__ import annotations

import argparse
import inspect
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

    # Match model storage dtype to training precision to avoid NaN loss from
    # fp16/bf16 mismatch in gradient computation.
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
# DR-DPO trainer
# ---------------------------------------------------------------------------

class DRDPOTrainer(DPOTrainer):
    """DPO trainer with Distributionally Robust loss aggregation.

    Overrides the standard mean aggregation with the entropic risk measure:
        loss = -beta_1 * log( mean( exp(-per_sample_loss / beta_1) ) )

    Implementation: dpo_loss() returns per-sample losses; we capture them
    before the parent's get_batch_loss_metrics() aggregates with .mean(),
    then replace that aggregation with the DR formula on return.
    """

    def __init__(self, *args, beta_1: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_1 = beta_1
        self._per_sample_losses: torch.Tensor | None = None

    def dpo_loss(self, *args, **kwargs):
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(*args, **kwargs)
        # Keep a reference with its grad_fn so we can re-aggregate below.
        self._per_sample_losses = losses
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        # Let the parent run the full forward pass and collect all metrics.
        # Its returned loss (simple mean) is discarded; we recompute with DR.
        _, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        losses = self._per_sample_losses
        beta_1 = self.beta_1

        # Numerically stable log-mean-exp via logsumexp:
        #   -beta_1 * log(mean(exp(-losses/beta_1)))
        #   = -beta_1 * (logsumexp(-losses/beta_1) - log(n))
        x = -losses / beta_1
        n = torch.tensor(x.numel(), dtype=x.dtype, device=x.device)
        loss = -beta_1 * (torch.logsumexp(x, dim=0) - torch.log(n))

        return loss, metrics


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
    parser.add_argument("--output_dir", default="output/dr_dpo_checkpoints")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true",
                         help="Train a LoRA adapter instead of the full model (requires peft).")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", nargs="+",
                         default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--loss_type", default="sigmoid",
                         choices=["sigmoid", "ipo", "robust", "hinge", "apo_zero", "apo_down",
                                  "nca_pair", "bco_pair", "exo_pair", "sppo_hard",
                                  "aot", "aot_unpaired", "discopop"],
                         help="DPO loss variant. 'sigmoid'=standard DPO, 'ipo'=IPO, "
                              "'robust'=noise-robust DPO, 'hinge'=SLiC hinge.")
    parser.add_argument("--beta", type=float, default=0.05,
                         help="DPO beta (KL penalty). Lower values (0.01-0.05) are better "
                              "for off-policy data where chosen/rejected come from a different "
                              "model than the one being trained.")
    parser.add_argument("--beta_1", type=float, default=1.0,
                         help="DR-DPO robustness temperature. Controls the entropic risk "
                              "measure: higher values approach standard DPO (mean), lower "
                              "values up-weight harder examples. Typical range: 0.1–2.0.")
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

    # Compute eval_steps so evaluation fires ~twice per epoch regardless of dataset size.
    # Hardcoding 50 would never trigger on small datasets (e.g. 341 examples / grad_accum 8
    # = ~43 optimizer steps/epoch, so eval_steps=50 would never fire).
    steps_per_epoch = max(1, len(split["train"]) // (args.batch_size * args.grad_accum))
    eval_steps = max(1, steps_per_epoch // 2)

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
        eval_steps=eval_steps,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    if _DPO_CONFIG_API:
        # TRL >= 0.9: DPO-specific params live in DPOConfig; tokenizer is passed
        # as processing_class. max_prompt_length was dropped in newer TRL builds,
        # so we probe the signature rather than hard-coding it.
        _dpo_kwargs: dict = dict(beta=args.beta, max_length=args.max_length, loss_type=args.loss_type)
        if "max_prompt_length" in inspect.signature(DPOConfig.__init__).parameters:
            _dpo_kwargs["max_prompt_length"] = args.max_prompt_length
        training_args = DPOConfig(**_shared_train_kwargs, **_dpo_kwargs)
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
