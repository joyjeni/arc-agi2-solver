"""
benchmark.py
────────────
Runs Stage 1 (deterministic rule bank) and Stage 2 (pattern induction engine)
against every task in a challenge JSON file, then reports per-category results.

Usage
─────
  python scripts/benchmark.py \
      --tasks  data/arc2_training.json \
      --output benchmark_results.json

The challenge file is expected to be a dict of {task_id: {"train": [...], "test": [...]}}.
Each "train" entry is {"input": grid, "output": grid};
each "test"  entry is {"input": grid, "output": grid}  (output required for scoring).

Exit code 0 always — failures are counted, not raised.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── add src/ to path so we can import without installing ──────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR   = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dsa_solver    import solve_with_dsa
from pattern_engine import solve_task  as solve_task_pattern


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def grids_equal(a, b):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
    return True


def run_benchmark(tasks: dict) -> dict:
    """
    For each task attempt Stage 1 then Stage 2.
    Returns a results dict.
    """
    total        = len(tasks)
    s1_solved    = 0   # solved by Stage 1 (DSA)
    s2_solved    = 0   # solved by Stage 2 (pattern engine), not already by S1
    failed       = 0
    errors       = 0

    per_task = {}

    for idx, (task_id, task_data) in enumerate(tasks.items(), 1):
        train_pairs = task_data.get("train", [])
        test_pairs  = task_data.get("test",  [])

        if not train_pairs or not test_pairs:
            errors += 1
            per_task[task_id] = {"stage": "skip", "reason": "missing train or test"}
            continue

        test_input  = test_pairs[0]["input"]
        test_output = test_pairs[0].get("output")

        t0 = time.perf_counter()

        # ── Stage 1: deterministic rule bank ─────────────────────────────────
        prediction = None
        stage      = "failed"
        try:
            prediction = solve_with_dsa(train_pairs, test_input)
        except Exception as exc:
            per_task[task_id] = {"stage": "error_s1", "error": str(exc)[:120]}
            errors += 1
            continue

        if prediction is not None and test_output is not None:
            if grids_equal(prediction, test_output):
                s1_solved += 1
                stage = "s1"
                per_task[task_id] = {
                    "stage": stage,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
                continue   # no need to run Stage 2

        # ── Stage 2: pattern induction engine ────────────────────────────────
        prediction = None
        try:
            prediction = solve_task_pattern(train_pairs, test_input)
        except Exception as exc:
            per_task[task_id] = {"stage": "error_s2", "error": str(exc)[:120]}
            errors += 1
            continue

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        if prediction is not None and test_output is not None:
            if grids_equal(prediction, test_output):
                s2_solved += 1
                stage = "s2"
            else:
                failed += 1
                stage = "failed"
        else:
            failed += 1
            stage = "no_pred" if prediction is None else "no_gt"

        per_task[task_id] = {"stage": stage, "elapsed_ms": elapsed_ms}

        # Progress heartbeat every 20 tasks
        if idx % 20 == 0 or idx == total:
            pct_done  = idx / total * 100
            solved_so = s1_solved + s2_solved
            print(f"  [{idx:3d}/{total}] {pct_done:5.1f}%  solved={solved_so}  "
                  f"s1={s1_solved}  s2={s2_solved}  failed={failed}  err={errors}")

    return {
        "total":     total,
        "s1_solved": s1_solved,
        "s2_solved": s2_solved,
        "combined":  s1_solved + s2_solved,
        "failed":    failed,
        "errors":    errors,
        "accuracy":  round((s1_solved + s2_solved) / max(total, 1) * 100, 2),
        "per_task":  per_task,
    }


def print_summary(results: dict):
    total  = results["total"]
    solved = results["combined"]
    print()
    print("=" * 52)
    print("  BENCHMARK SUMMARY")
    print("=" * 52)
    print(f"  Total tasks       : {total}")
    print(f"  Stage-1 solved    : {results['s1_solved']}")
    print(f"  Stage-2 solved    : {results['s2_solved']}")
    print(f"  Combined solved   : {solved}  ({results['accuracy']:.1f}%)")
    print(f"  Failed            : {results['failed']}")
    print(f"  Errors            : {results['errors']}")
    print("=" * 52)

    # Stage breakdown
    stage_counts = {}
    for info in results["per_task"].values():
        s = info.get("stage", "unknown")
        stage_counts[s] = stage_counts.get(s, 0) + 1
    print("\n  Stage breakdown:")
    for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"    {stage:<20s}: {count}")
    print()


def write_github_step_summary(results: dict, path: str):
    """Write a Markdown summary to $GITHUB_STEP_SUMMARY if running in CI."""
    lines = [
        "## ARC-AGI-2 Benchmark Results\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total tasks | {results['total']} |",
        f"| Stage-1 solved (DSA) | {results['s1_solved']} |",
        f"| Stage-2 solved (pattern engine) | {results['s2_solved']} |",
        f"| **Combined solved** | **{results['combined']} / {results['total']}** |",
        f"| Accuracy | {results['accuracy']:.1f}% |",
        f"| Errors | {results['errors']} |",
        "",
        "### Stage breakdown",
        "| Stage | Count |",
        "|-------|-------|",
    ]
    stage_counts = {}
    for info in results["per_task"].values():
        s = info.get("stage", "unknown")
        stage_counts[s] = stage_counts.get(s, 0) + 1
    for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{stage}` | {count} |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Stage 1+2 benchmark")
    parser.add_argument("--tasks",  required=True,
                        help="Path to challenge JSON file (dict of task_id → task_data)")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Path to write JSON results (default: benchmark_results.json)")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Limit to first N tasks (0 = all)")
    args = parser.parse_args()

    print(f"Loading tasks from: {args.tasks}")
    with open(args.tasks) as f:
        tasks = json.load(f)

    if args.limit > 0:
        tasks = dict(list(tasks.items())[:args.limit])
        print(f"Limiting to {args.limit} tasks")

    print(f"Running benchmark on {len(tasks)} tasks ...\n")
    t_start  = time.perf_counter()
    results  = run_benchmark(tasks)
    elapsed  = time.perf_counter() - t_start
    results["wall_seconds"] = round(elapsed, 1)

    print_summary(results)
    print(f"  Wall time: {elapsed:.1f}s")

    # Write JSON results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Write GitHub step summary if in CI
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        write_github_step_summary(results, summary_path)
        print(f"GitHub step summary written to: {summary_path}")


if __name__ == "__main__":
    main()
