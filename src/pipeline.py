"""
pipeline.py
───────────
Master solving pipeline for ARC-AGI-2.

Stage 1: Deterministic rule solvers  (fast, exact)
Stage 2: Pattern-induction engine    (medium speed)
Stage 3: LLM inference               (slow, GPU required)

Output: submission.json in competition format.
"""

import json
import sys
import os
from typing import Dict, List, Optional

from grid_utils import equal, clone, shape

# Import DSA solver (stage 1)
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dsa_solver import solve_with_dsa  # See dsa_solver.py
    DSA_AVAILABLE = True
except ImportError:
    DSA_AVAILABLE = False

# Import pattern engine (stage 2)
from pattern_engine import solve_task as pattern_solve

# Fallback grid
FALLBACK = [[0, 0], [0, 0]]


# ──────────────────────────────────────────────────────────────────
# Submission format helpers
# ──────────────────────────────────────────────────────────────────

def format_attempt(grid):
    """Ensure grid is a list-of-lists of ints."""
    if grid is None:
        return FALLBACK
    return [[int(v) for v in row] for row in grid]


def make_submission_entry(preds_per_test: List[Dict]) -> List[Dict]:
    """
    Each test input needs {"attempt_1": grid, "attempt_2": grid}.
    """
    return preds_per_test


def build_submission(predictions: Dict) -> Dict:
    """
    predictions: {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}
    Returns the exact submission.json structure.
    """
    return predictions


def save_submission(predictions: Dict, path: str = "submission.json"):
    with open(path, "w") as f:
        json.dump(build_submission(predictions), f)
    print(f"[pipeline] Saved submission → {path}  ({len(predictions)} tasks)")


# ──────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────

def run_pipeline(
    tasks: Dict,
    llm_solver=None,
    verbose: bool = True,
    submission_path: str = "submission.json",
) -> Dict:
    """
    tasks: {task_id: {"train": [...], "test": [...]}}
    llm_solver: optional ArcLLMSolver instance (stage 3)
    Returns full predictions dict.
    """
    predictions = {}
    stage_counts = {"dsa": 0, "pattern": 0, "llm": 0, "fallback": 0}
    total = len(tasks)

    for idx, (tid, task) in enumerate(tasks.items()):
        if verbose and (idx % 20 == 0):
            print(f"  [{idx:3d}/{total}] dsa={stage_counts['dsa']} "
                  f"pat={stage_counts['pattern']} llm={stage_counts['llm']}")

        train_pairs = task["train"]
        test_items  = task["test"]
        task_preds  = []

        for test_item in test_items:
            test_inp = test_item["input"]
            a1, a2 = None, None
            stage = "fallback"

            # ── Stage 1: DSA deterministic rules ────────────────
            if DSA_AVAILABLE:
                try:
                    dsa_pred = solve_with_dsa(train_pairs, test_inp)
                    if dsa_pred is not None:
                        a1 = dsa_pred
                        stage = "dsa"
                except Exception:
                    pass

            # ── Stage 2: Pattern induction engine ───────────────
            if a1 is None:
                try:
                    pred, solver_name, conf = pattern_solve(train_pairs, test_inp)
                    if pred is not None and conf > 0.0:
                        a1 = pred
                        stage = "pattern"
                except Exception:
                    pass

            # ── Stage 3: LLM inference ───────────────────────────
            if a1 is None and llm_solver is not None:
                try:
                    llm_preds = llm_solver.solve(train_pairs, test_inp, n_attempts=2)
                    if llm_preds[0] is not None:
                        a1 = llm_preds[0]
                        a2 = llm_preds[1] if len(llm_preds) > 1 else a1
                        stage = "llm"
                except Exception:
                    pass

            # ── Fallback ─────────────────────────────────────────
            if a1 is None:
                a1 = FALLBACK
                stage = "fallback"
            if a2 is None:
                a2 = a1   # Use same prediction for both attempts

            task_preds.append({
                "attempt_1": format_attempt(a1),
                "attempt_2": format_attempt(a2),
            })
            stage_counts[stage] += 1

        predictions[tid] = task_preds

    if verbose:
        print("\n[pipeline] Stage breakdown:")
        for stage, cnt in stage_counts.items():
            print(f"  {stage:10s}: {cnt}")

    save_submission(predictions, submission_path)
    return predictions


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",  default="arc2_combined_train.json")
    parser.add_argument("--output", default="submission.json")
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)

    llm_solver = None
    if not args.no_llm:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            MODEL_PATH = (
                "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
                if os.path.exists("/kaggle")
                else os.environ.get("ARC_MODEL_PATH", "")
            )
            if MODEL_PATH and os.path.exists(MODEL_PATH):
                print(f"[pipeline] Loading model from {MODEL_PATH}")
                tok = AutoTokenizer.from_pretrained(MODEL_PATH)
                mdl = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                from llm_solver import ArcLLMSolver
                llm_solver = ArcLLMSolver(mdl, tok)
            else:
                print("[pipeline] Model path not found, skipping LLM stage.")
        except Exception as e:
            print(f"[pipeline] LLM unavailable: {e}")

    run_pipeline(tasks, llm_solver=llm_solver,
                 submission_path=args.output, verbose=True)
