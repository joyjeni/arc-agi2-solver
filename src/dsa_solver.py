"""
dsa_solver.py
─────────────
Thin wrapper around the deterministic rule solver bank.
Exposes solve_with_dsa(train_pairs, test_input) → grid or None.

Training-consistency check
──────────────────────────
Before accepting a solver's test prediction, we verify it correctly
reproduces at least `MIN_TRAIN_PASS_RATE` (default 50%) of the training
outputs.  This filters out solvers that happen to return a valid-shaped
grid for the test input but don't actually model the task's pattern —
a common source of false positives when running 100+ heuristic solvers
sequentially and returning the first valid result.
"""
import sys, os

# Allow importing dsa_prepass_src from the same directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Minimum fraction of training pairs a solver must reproduce correctly
# before its test prediction is trusted.
_MIN_TRAIN_PASS_RATE = 0.50

try:
    import dsa_prepass_src as _dsa
    _SOLVERS = _dsa._T1_SOLVERS
    _valid   = _dsa._valid_grid

    def _grids_equal(a, b):
        """Return True iff two grids (list-of-lists) are element-wise equal."""
        if len(a) != len(b):
            return False
        for ra, rb in zip(a, b):
            if list(ra) != list(rb):
                return False
        return True

    def _training_pass_rate(solver, pairs):
        """
        Run solver on every training input and return the fraction of
        training outputs it reproduces exactly.  Exceptions count as misses.
        """
        if not pairs:
            return 0.0
        passed = 0
        for pair in pairs:
            try:
                pred = solver(pairs, pair["input"])
                if _valid(pred) and _grids_equal(pred, pair["output"]):
                    passed += 1
            except Exception:
                pass
        return passed / len(pairs)

    def solve_with_dsa(train_pairs, test_input,
                       min_train_rate=_MIN_TRAIN_PASS_RATE):
        """
        Run all deterministic T1 solvers on a single test input.

        A solver is accepted only if:
          1. It correctly reproduces >= min_train_rate of the training outputs.
          2. Its test prediction is a valid grid.

        Returns the first accepted prediction, or None.
        """
        pairs = [p for p in train_pairs
                 if p.get("input") is not None and p.get("output") is not None]
        if not pairs:
            return None

        for solver in _SOLVERS:
            try:
                # Gate: solver must model the training pattern reliably
                if _training_pass_rate(solver, pairs) < min_train_rate:
                    continue
                pred = solver(pairs, test_input)
                if _valid(pred):
                    return pred
            except Exception:
                pass

        return None

    def solve_task_dsa(task_data):
        """Run DSA on a full task dict {train, test}. Returns submission entries."""
        results = _dsa._run_solvers_on_task(task_data)
        if results is None:
            return None
        return results   # list of [attempt_1, attempt_2] per test

    DSA_AVAILABLE = True

except ImportError as e:
    print(f"[dsa_solver] dsa_prepass_src not found: {e}")
    DSA_AVAILABLE = False

    def solve_with_dsa(train_pairs, test_input, min_train_rate=_MIN_TRAIN_PASS_RATE):
        return None

    def solve_task_dsa(task_data):
        return None
