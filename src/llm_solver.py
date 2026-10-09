"""
llm_solver.py
─────────────
LLM-based solver built on a fine-tuned Qwen base model.
Encodes ARC grids as compact token sequences and prompts the model
to predict the output grid for each test input.
"""

import re
import sys
import json
from typing import List, Dict, Optional

from grid_utils import grid_to_str, str_to_grid, shape, clone, equal


# ──────────────────────────────────────────────────────────────────
# Grid ↔ Text encoding
# ──────────────────────────────────────────────────────────────────

def encode_grid(grid: List[List[int]], style: str = "space") -> str:
    """
    Serialise a grid to a string.
      style='space'   → "1 2 3\n4 5 6"
      style='compact' → "123\n456"
      style='pipe'    → "|1|2|3|\n|4|5|6|"
    """
    if style == "space":
        return "\n".join(" ".join(str(v) for v in row) for row in grid)
    elif style == "compact":
        return "\n".join("".join(str(v) for v in row) for row in grid)
    elif style == "pipe":
        return "\n".join("|" + "|".join(str(v) for v in row) + "|" for row in grid)
    return grid_to_str(grid)


def decode_grid(text: str) -> Optional[List[List[int]]]:
    """
    Parse a model output back to a 2D grid.
    Handles space-separated, compact, and pipe styles.
    Falls back to best-effort parsing.
    """
    text = text.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None
    # Detect style
    if lines[0].startswith("|"):
        # pipe style
        rows = []
        for l in lines:
            nums = re.findall(r"\d", l)
            if nums:
                rows.append([int(n) for n in nums])
        return rows if rows else None
    if " " in lines[0]:
        # space-separated
        try:
            return [[int(v) for v in l.split()] for l in lines]
        except ValueError:
            pass
    # compact (no spaces)
    try:
        rows = [[int(ch) for ch in l if ch.isdigit()] for l in lines]
        return rows if all(rows) else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert at abstract pattern recognition. "
    "Study the input→output examples carefully. "
    "Identify the transformation rule and apply it to the test input. "
    "Output ONLY the predicted grid, nothing else."
)

def build_prompt(train_pairs: List[Dict], test_input: List[List[int]],
                 grid_style: str = "space") -> str:
    lines = []
    for i, p in enumerate(train_pairs, 1):
        lines.append(f"## Example {i}")
        lines.append("Input:")
        lines.append(encode_grid(p["input"], grid_style))
        lines.append("Output:")
        lines.append(encode_grid(p["output"], grid_style))
    lines.append("## Test")
    lines.append("Input:")
    lines.append(encode_grid(test_input, grid_style))
    lines.append("Output:")
    return "\n".join(lines)


def build_chat_messages(train_pairs, test_input, grid_style="space"):
    prompt = build_prompt(train_pairs, test_input, grid_style)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


# ──────────────────────────────────────────────────────────────────
# ARC LLM Solver  (wraps a loaded HuggingFace model)
# ──────────────────────────────────────────────────────────────────

class ArcLLMSolver:
    """
    Usage:
        solver = ArcLLMSolver(model, tokenizer)
        pred = solver.solve(train_pairs, test_input)
    """

    def __init__(self, model, tokenizer,
                 max_new_tokens: int = 512,
                 temperature: float = 0.0,
                 grid_style: str = "space"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.grid_style = grid_style

    def _generate(self, prompt_text: str) -> str:
        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        with __import__("torch").no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        # Decode only newly generated tokens
        new_ids = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _chat_generate(self, messages) -> str:
        """Use apply_chat_template if available, else fall back to plain prompt."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self._generate(prompt)

    def solve(self, train_pairs, test_input,
              n_attempts: int = 2) -> List[Optional[List[List[int]]]]:
        """
        Returns a list of up to `n_attempts` grid predictions.
        Each prediction is a 2D list or None.
        """
        preds = []
        messages = build_chat_messages(train_pairs, test_input, self.grid_style)

        for attempt in range(n_attempts):
            # Slightly vary temperature for second attempt
            if attempt > 0 and self.temperature == 0:
                self.temperature = 0.3
            raw = self._chat_generate(messages)
            pred = decode_grid(raw)
            preds.append(pred)
            if attempt > 0:
                self.temperature = 0.0  # reset

        return preds


# ──────────────────────────────────────────────────────────────────
# Batch solve helper
# ──────────────────────────────────────────────────────────────────

def batch_solve(llm_solver: ArcLLMSolver,
                tasks: Dict,
                skip_ids: set = None,
                verbose: bool = False) -> Dict:
    """
    tasks: {task_id: {"train": [...], "test": [...]}}
    skip_ids: task IDs already solved by other methods
    Returns: {task_id: [{"attempt_1": grid, "attempt_2": grid}]}
    """
    skip_ids = skip_ids or set()
    results = {}
    total = len(tasks)
    for i, (tid, task) in enumerate(tasks.items()):
        if tid in skip_ids:
            continue
        if verbose and i % 10 == 0:
            print(f"  LLM solving [{i}/{total}] {tid}")
        train_pairs = task["train"]
        test_inputs = task["test"]
        task_preds = []
        for test_item in test_inputs:
            preds = llm_solver.solve(train_pairs, test_item["input"], n_attempts=2)
            a1 = preds[0] if preds[0] is not None else [[0, 0], [0, 0]]
            a2 = preds[1] if preds[1] is not None else a1
            task_preds.append({"attempt_1": a1, "attempt_2": a2})
        results[tid] = task_preds
    return results
