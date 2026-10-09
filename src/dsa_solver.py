"""
dsa_solver.py
─────────────
Thin wrapper around the deterministic rule solver bank.
Exposes solve_with_dsa(train_pairs, test_input) → grid or None.
"""
import sys, os

# Allow importing dsa_prepass_src from the same directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    import dsa_prepass_src as _dsa
    _SOLVERS = _dsa._T1_SOLVERS
    _valid   = _dsa._valid_grid

    def solve_with_dsa(train_pairs, test_input):
        """
        Run all deterministic T1 solvers on a single test input.
        Returns the first valid prediction, or None.
        """
        pairs = [p for p in train_pairs
                 if p.get("input") is not None and p.get("output") is not None]
        if not pairs:
            return None
        for solver in _SOLVERS:
            try:
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

    def solve_with_dsa(train_pairs, test_input):
        return None

    def solve_task_dsa(task_data):
        return None
