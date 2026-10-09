"""
finetune_qwen.py
────────────────
Fine-tune a Qwen-family model on ARC-AGI-2 tasks using LoRA/PEFT.

Usage (Kaggle GPU notebook or local):
  python finetune_qwen.py \
      --model_path /path/to/qwen \
      --data_path  arc2_finetune_500.jsonl \
      --output_dir ./arc_finetuned \
      --epochs 3
"""

import os
import json
import argparse
from typing import List, Dict


# ──────────────────────────────────────────────────────────────────
# Data loading and formatting
# ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_sample(sample: Dict, tokenizer) -> str:
    """
    Convert a JSONL sample with 'instruction', 'input', 'output' fields
    into a single chat-formatted string.
    """
    instruction = sample.get("instruction", "")
    inp         = sample.get("input", "")
    output      = sample.get("output", "")

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user",   "content": inp},
        {"role": "assistant", "content": output},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        return f"SYSTEM: {instruction}\nUSER: {inp}\nASSISTANT: {output}"


class ArcDataset:
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = [format_sample(d, tokenizer) for d in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        return {
            "input_ids":      input_ids,
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         input_ids.clone(),
        }


# ──────────────────────────────────────────────────────────────────
# LoRA configuration
# ──────────────────────────────────────────────────────────────────

def get_lora_config():
    from peft import LoraConfig, TaskType
    return LoraConfig(
        task_type    = TaskType.CAUSAL_LM,
        r            = 16,
        lora_alpha   = 32,
        lora_dropout = 0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias         = "none",
    )


# ──────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────

def train(args):
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import get_peft_model

    print(f"[train] Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[train] Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # Apply LoRA
    lora_cfg = get_lora_config()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset
    print(f"[train] Loading data from {args.data_path}")
    data = load_jsonl(args.data_path)
    split = int(len(data) * 0.9)
    train_ds = ArcDataset(data[:split],  tokenizer, args.max_length)
    eval_ds  = ArcDataset(data[split:],  tokenizer, args.max_length)
    print(f"[train] Train={len(train_ds)}  Eval={len(eval_ds)}")

    training_args = TrainingArguments(
        output_dir              = args.output_dir,
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = 1,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate           = args.lr,
        weight_decay            = 0.01,
        warmup_ratio            = 0.05,
        lr_scheduler_type       = "cosine",
        bf16                    = torch.cuda.is_available(),
        logging_steps           = 10,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        report_to               = "none",
        dataloader_num_workers  = 0,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        data_collator   = DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("[train] Starting fine-tuning …")
    trainer.train()

    print(f"[train] Saving to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[train] Done.")


# ──────────────────────────────────────────────────────────────────
# Evaluate on ARC task format
# ──────────────────────────────────────────────────────────────────

def evaluate_on_tasks(model, tokenizer, tasks_path: str, max_tasks: int = 50):
    import torch
    from llm_solver import ArcLLMSolver
    from grid_utils import equal

    with open(tasks_path) as f:
        tasks = json.load(f)

    solver  = ArcLLMSolver(model, tokenizer, max_new_tokens=256, temperature=0.0)
    correct = 0
    total   = 0

    for tid, task in list(tasks.items())[:max_tasks]:
        train_pairs = task["train"]
        for test_item in task["test"]:
            if "output" not in test_item:
                continue
            preds = solver.solve(train_pairs, test_item["input"], n_attempts=1)
            if preds[0] is not None and equal(preds[0], test_item["output"]):
                correct += 1
            total += 1

    acc = correct / total if total else 0
    print(f"[eval] LLM accuracy on {total} test outputs: {correct}/{total} = {acc:.2%}")
    return acc


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  required=True)
    p.add_argument("--data_path",   default="arc2_finetune_500.jsonl")
    p.add_argument("--output_dir",  default="./arc_qwen_finetuned")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max_length",  type=int,   default=2048)
    p.add_argument("--eval_tasks",  default=None,
                   help="Path to tasks JSON for post-train evaluation")
    args = p.parse_args()

    train(args)

    if args.eval_tasks:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, args.output_dir)
        model = model.merge_and_unload()
        tok   = AutoTokenizer.from_pretrained(args.output_dir)
        evaluate_on_tasks(model, tok, args.eval_tasks)
