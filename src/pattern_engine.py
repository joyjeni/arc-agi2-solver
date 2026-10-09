"""
pattern_engine.py
─────────────────
Rule-induction and pattern-matching solvers for ARC-AGI-2.

Each solver exposes:
  .learn(train_pairs)  → infer transformation from examples
  .apply(test_input)   → apply rule to unseen grid
  .confidence()        → float [0,1] – reliability estimate

SolverEnsemble tries every solver, self-validates on training
pairs, and returns the highest-confidence correct prediction.
"""

import copy, math
from collections import Counter, defaultdict, deque
from itertools import product as iproduct

from grid_utils import (
    shape, bg, fg_colors, clone, equal, make_grid,
    rotate_90cw, rotate_180, rotate_90ccw, flip_h, flip_v, transpose,
    bfs_fill, connected_components, bounding_box,
    extract_subgrid, paste_subgrid,
    find_objects, get_neighbourhood, get_neighbourhood_cross,
    grid_to_str, str_to_grid,
    remap_colors, color_histogram,
    is_horizontally_symmetric, is_vertically_symmetric,
    is_diagonally_symmetric, is_rotationally_symmetric_180,
    grid_diff, changed_cells
)


# ══════════════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════════════

class BaseSolver:
    name = "BaseSolver"

    def __init__(self):
        self._conf = 0.0

    def learn(self, train_pairs):
        raise NotImplementedError

    def apply(self, test_input):
        raise NotImplementedError

    def confidence(self):
        return self._conf

    def _validate(self, train_pairs):
        """Self-test: return fraction of train pairs reproduced exactly."""
        ok = 0
        for p in train_pairs:
            try:
                pred = self.apply(p['input'])
                if pred is not None and equal(pred, p['output']):
                    ok += 1
            except Exception:
                pass
        return ok / len(train_pairs) if train_pairs else 0.0


# ══════════════════════════════════════════════════════════════════
# 1.  Neighbourhood Rule Learner
#     Learns: context_window → output_colour lookup table.
#     Handles same-size I/O grids via cellular-automaton-style rules.
# ══════════════════════════════════════════════════════════════════

class NeighborhoodRuleLearner(BaseSolver):
    name = "NeighborhoodRuleLearner"

    def __init__(self):
        super().__init__()
        self._rules_r1 = {}   # radius-1 (3×3)
        self._rules_r2 = {}   # radius-2 (5×5)
        self._use_r2 = False

    def learn(self, train_pairs):
        # Try radius-1 first
        r1_rules = defaultdict(Counter)
        r2_rules = defaultdict(Counter)

        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) != shape(out):
                continue
            R, C = shape(inp)
            for r in range(R):
                for c in range(C):
                    ctx1 = get_neighbourhood(inp, r, c, radius=1)
                    ctx2 = get_neighbourhood(inp, r, c, radius=2)
                    target = out[r][c]
                    r1_rules[ctx1][target] += 1
                    r2_rules[ctx2][target] += 1

        # Resolve ambiguous rules by majority
        self._rules_r1 = {k: max(v, key=v.get) for k, v in r1_rules.items()}
        self._rules_r2 = {k: max(v, key=v.get) for k, v in r2_rules.items()}

        # Check coverage on training set
        r1_hits = self._coverage(train_pairs, self._rules_r1, radius=1)
        r2_hits = self._coverage(train_pairs, self._rules_r2, radius=2)
        self._use_r2 = r2_hits > r1_hits
        self._conf = max(r1_hits, r2_hits)

    def _coverage(self, train_pairs, rules, radius):
        ok = total = 0
        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) != shape(out):
                continue
            R, C = shape(inp)
            for r in range(R):
                for c in range(C):
                    ctx = get_neighbourhood(inp, r, c, radius=radius)
                    total += 1
                    if rules.get(ctx) == out[r][c]:
                        ok += 1
        return ok / total if total else 0.0

    def apply(self, test_input):
        R, C = shape(test_input)
        rules = self._rules_r2 if self._use_r2 else self._rules_r1
        radius = 2 if self._use_r2 else 1
        out = clone(test_input)
        for r in range(R):
            for c in range(C):
                ctx = get_neighbourhood(test_input, r, c, radius=radius)
                if ctx in rules:
                    out[r][c] = rules[ctx]
        return out


# ══════════════════════════════════════════════════════════════════
# 2.  Bilateral Symmetry Solver
#     Detects partially-filled grids; completes via mirror reflection.
# ══════════════════════════════════════════════════════════════════

class BilateralSymmetrySolver(BaseSolver):
    name = "BilateralSymmetrySolver"

    def __init__(self):
        super().__init__()
        self._axis = None   # 'h', 'v', 'd', 'anti_d'

    def learn(self, train_pairs):
        axes = Counter()
        for p in train_pairs:
            inp, out = p['input'], p['output']
            if is_horizontally_symmetric(out): axes['h'] += 1
            if is_vertically_symmetric(out):   axes['v'] += 1
            if is_diagonally_symmetric(out):   axes['d'] += 1
            if shape(out)[0] == shape(out)[1] and is_diagonally_symmetric(flip_h(out)):
                axes['anti_d'] += 1
        if axes:
            self._axis = axes.most_common(1)[0][0]
            self._conf = self._validate(train_pairs)
        else:
            self._conf = 0.0

    def apply(self, test_input):
        if self._axis is None:
            return None
        g = clone(test_input)
        R, C = shape(g)
        bg_val = bg(g)

        if self._axis == 'h':
            # Mirror right side to left (or fill missing half)
            for r in range(R):
                for c in range(C):
                    mirror_c = C - 1 - c
                    if g[r][c] == bg_val and g[r][mirror_c] != bg_val:
                        g[r][c] = g[r][mirror_c]
                    elif g[r][mirror_c] == bg_val and g[r][c] != bg_val:
                        g[r][mirror_c] = g[r][c]

        elif self._axis == 'v':
            for r in range(R):
                mirror_r = R - 1 - r
                for c in range(C):
                    if g[r][c] == bg_val and g[mirror_r][c] != bg_val:
                        g[r][c] = g[mirror_r][c]
                    elif g[mirror_r][c] == bg_val and g[r][c] != bg_val:
                        g[mirror_r][c] = g[r][c]

        elif self._axis == 'd' and R == C:
            for r in range(R):
                for c in range(C):
                    if g[r][c] == bg_val and g[c][r] != bg_val:
                        g[r][c] = g[c][r]
                    elif g[c][r] == bg_val and g[r][c] != bg_val:
                        g[c][r] = g[r][c]

        elif self._axis == 'anti_d' and R == C:
            for r in range(R):
                for c in range(C):
                    ar, ac = C - 1 - c, R - 1 - r
                    if g[r][c] == bg_val and g[ar][ac] != bg_val:
                        g[r][c] = g[ar][ac]
                    elif g[ar][ac] == bg_val and g[r][c] != bg_val:
                        g[ar][ac] = g[r][c]
        return g


# ══════════════════════════════════════════════════════════════════
# 3.  Quadrant Symmetry Solver  (4-fold / rotational invariance)
# ══════════════════════════════════════════════════════════════════

class QuadrantSymmetrySolver(BaseSolver):
    name = "QuadrantSymmetrySolver"

    def learn(self, train_pairs):
        ok = 0
        for p in train_pairs:
            out = p['output']
            if is_rotationally_symmetric_180(out):
                ok += 1
        ratio = ok / len(train_pairs) if train_pairs else 0
        if ratio >= 0.5:
            self._conf = self._validate(train_pairs)
        else:
            self._conf = 0.0

    def apply(self, test_input):
        g = clone(test_input)
        R, C = shape(g)
        bg_val = bg(g)
        # Fill by 180° rotational partner
        changed = True
        while changed:
            changed = False
            for r in range(R):
                for c in range(C):
                    mr, mc = R - 1 - r, C - 1 - c
                    if g[r][c] == bg_val and g[mr][mc] != bg_val:
                        g[r][c] = g[mr][mc]
                        changed = True
                    elif g[mr][mc] == bg_val and g[r][c] != bg_val:
                        g[mr][mc] = g[r][c]
                        changed = True
        return g


# ══════════════════════════════════════════════════════════════════
# 4.  Transposition Symmetry Solver  (diagonal symmetry)
# ══════════════════════════════════════════════════════════════════

class TranspositionSymmetrySolver(BaseSolver):
    name = "TranspositionSymmetrySolver"

    def learn(self, train_pairs):
        ok = sum(1 for p in train_pairs if is_diagonally_symmetric(p['output']))
        ratio = ok / len(train_pairs) if train_pairs else 0
        self._conf = self._validate(train_pairs) if ratio >= 0.5 else 0.0

    def apply(self, test_input):
        R, C = shape(test_input)
        if R != C:
            return None
        g = clone(test_input)
        bg_val = bg(g)
        for r in range(R):
            for c in range(C):
                if g[r][c] == bg_val and g[c][r] != bg_val:
                    g[r][c] = g[c][r]
                elif g[c][r] == bg_val and g[r][c] != bg_val:
                    g[c][r] = g[r][c]
        return g


# ══════════════════════════════════════════════════════════════════
# 5.  Region Extraction Solver
#     Finds the relevant sub-object/crop from a larger grid.
# ══════════════════════════════════════════════════════════════════

class RegionExtractionSolver(BaseSolver):
    name = "RegionExtractionSolver"

    def __init__(self):
        super().__init__()
        self._strategy = None  # 'largest', 'smallest', 'unique_color', 'bbox'
        self._target_color = None

    def learn(self, train_pairs):
        strategies = Counter()
        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) == shape(out):
                continue  # Skip same-size: not an extraction task
            objs = find_objects(inp)
            if not objs:
                continue
            # Try matching output to a sub-object
            for obj in objs:
                r1, c1, r2, c2 = obj['bbox']
                sub = extract_subgrid(inp, r1, c1, r2, c2)
                if equal(sub, out):
                    strategies['bbox'] += 1
                    break
            # Try unique-colour sub-region
            hist = color_histogram(inp)
            unique_cols = [c for c, n in hist.items() if n == min(hist.values())]
            for uc in unique_cols:
                comps = connected_components(inp, color=uc)
                if comps:
                    r1, c1, r2, c2 = bounding_box(comps[0])
                    sub = extract_subgrid(inp, r1, c1, r2, c2)
                    if equal(sub, out):
                        strategies['unique_color'] += 1
                        self._target_color = uc
                        break

        if strategies:
            self._strategy = strategies.most_common(1)[0][0]
            self._conf = self._validate(train_pairs)
        else:
            self._conf = 0.0

    def apply(self, test_input):
        objs = find_objects(test_input)
        if not objs:
            return None

        if self._strategy == 'unique_color' and self._target_color is not None:
            comps = connected_components(test_input, color=self._target_color)
            if comps:
                r1, c1, r2, c2 = bounding_box(comps[0])
                return extract_subgrid(test_input, r1, c1, r2, c2)

        if self._strategy == 'bbox':
            # Return the bounding box of the largest object
            largest = max(objs, key=lambda o: len(o['cells']))
            r1, c1, r2, c2 = largest['bbox']
            return extract_subgrid(test_input, r1, c1, r2, c2)

        return None


# ══════════════════════════════════════════════════════════════════
# 6.  Flood-Fill Enclosure Solver
#     Detects closed perimeters; fills interior with learned colour.
# ══════════════════════════════════════════════════════════════════

class FloodFillEnclosureSolver(BaseSolver):
    name = "FloodFillEnclosureSolver"

    def __init__(self):
        super().__init__()
        self._fill_color = None
        self._border_color = None

    def _find_interiors(self, grid):
        """Return list of interior cell-sets (unreachable from border)."""
        R, C = shape(grid)
        bg_val = bg(grid)
        # BFS from all border background cells
        reachable = [[False] * C for _ in range(R)]
        q = deque()
        for r in range(R):
            for c in range(C):
                if (r == 0 or r == R-1 or c == 0 or c == C-1) and grid[r][c] == bg_val:
                    reachable[r][c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r+dr, c+dc
                if 0<=nr<R and 0<=nc<C and not reachable[nr][nc] and grid[nr][nc] == bg_val:
                    reachable[nr][nc] = True
                    q.append((nr, nc))
        interiors = []
        for r in range(R):
            for c in range(C):
                if not reachable[r][c] and grid[r][c] == bg_val:
                    interiors.append((r, c))
        return interiors

    def learn(self, train_pairs):
        fill_colors = Counter()
        border_colors = Counter()
        has_fill = False

        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) != shape(out):
                continue
            interiors = self._find_interiors(inp)
            if not interiors:
                continue
            # What colour did these cells get in output?
            out_colors = Counter(out[r][c] for r, c in interiors)
            if out_colors:
                fc = out_colors.most_common(1)[0][0]
                fill_colors[fc] += 1
                has_fill = True
            # Detect border colour
            bg_val = bg(inp)
            for r, c in interiors:
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    R, C = shape(inp)
                    if 0<=nr<R and 0<=nc<C and inp[nr][nc] != bg_val:
                        border_colors[inp[nr][nc]] += 1

        if has_fill:
            self._fill_color = fill_colors.most_common(1)[0][0]
            if border_colors:
                self._border_color = border_colors.most_common(1)[0][0]
            self._conf = self._validate(train_pairs)
        else:
            self._conf = 0.0

    def apply(self, test_input):
        if self._fill_color is None:
            return None
        interiors = self._find_interiors(test_input)
        if not interiors:
            return None
        g = clone(test_input)
        for r, c in interiors:
            g[r][c] = self._fill_color
        return g


# ══════════════════════════════════════════════════════════════════
# 7.  Object Projection Solver
#     Learns object motion vectors; projects ("ghosts") objects to
#     their continuation positions.
# ══════════════════════════════════════════════════════════════════

class ObjectProjectionSolver(BaseSolver):
    name = "ObjectProjectionSolver"

    def __init__(self):
        super().__init__()
        self._delta = None  # (dr, dc) average motion vector
        self._ghost_color = None

    def _objects_by_color(self, grid, color):
        comps = connected_components(grid, color=color)
        return [bounding_box(comp) for comp in comps]

    def learn(self, train_pairs):
        deltas = []
        ghost_colors = Counter()

        for p in train_pairs:
            inp, out = p['input'], p['output']
            changes = changed_cells(inp, out)
            if not changes:
                continue
            # New cells that appeared in output (not bg)
            bg_val = bg(inp)
            new_cells = [(r, c, v) for r, c, v in changes if inp[r][c] == bg_val and v != bg_val]
            if not new_cells:
                continue

            # Find existing objects in input
            objs = find_objects(inp)
            if not objs:
                continue

            for obj in objs:
                # Compute centroid of object
                cells = obj['cells']
                cr_in = sum(r for r, c in cells) / len(cells)
                cc_in = sum(c for r, c in cells) / len(cells)

                # Check if same-shaped object appeared at new position
                for r, c, v in new_cells:
                    if v == obj['color']:
                        # Approximate delta
                        dr = r - cr_in
                        dc = c - cc_in
                        deltas.append((round(dr), round(dc)))
                        ghost_colors[v] += 1
                        break

        if deltas and ghost_colors:
            # Most common delta
            dc = Counter(deltas)
            self._delta = dc.most_common(1)[0][0]
            self._ghost_color = ghost_colors.most_common(1)[0][0]
            self._conf = self._validate(train_pairs)
        else:
            self._conf = 0.0

    def apply(self, test_input):
        if self._delta is None:
            return None
        dr, dc = self._delta
        R, C = shape(test_input)
        objs = [o for o in find_objects(test_input) if o['color'] == self._ghost_color]
        if not objs:
            objs = find_objects(test_input)
        g = clone(test_input)
        for obj in objs:
            for r, c in obj['cells']:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    g[nr][nc] = obj['color']
        return g


# ══════════════════════════════════════════════════════════════════
# 8.  Local Context Expansion Solver
#     Learns: which markers to place AROUND existing objects/cells.
# ══════════════════════════════════════════════════════════════════

class LocalContextExpansionSolver(BaseSolver):
    name = "LocalContextExpansionSolver"

    def __init__(self):
        super().__init__()
        self._expand_offsets = []   # list of (dr, dc, new_color)
        self._trigger_color = None

    def learn(self, train_pairs):
        offset_counter = Counter()
        trigger_colors = Counter()

        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) != shape(out):
                continue
            bg_val = bg(inp)
            R, C = shape(inp)
            for r in range(R):
                for c in range(C):
                    if inp[r][c] != bg_val and out[r][c] == inp[r][c]:
                        trigger_colors[inp[r][c]] += 1
                        # Check what new cells appeared nearby
                        for dr in range(-3, 4):
                            for dc in range(-3, 4):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r+dr, c+dc
                                if 0<=nr<R and 0<=nc<C:
                                    if inp[nr][nc] == bg_val and out[nr][nc] != bg_val:
                                        offset_counter[(dr, dc, out[nr][nc])] += 1

        if offset_counter and trigger_colors:
            # Keep offsets that appear consistently
            max_cnt = max(offset_counter.values())
            threshold = max_cnt * 0.5
            self._expand_offsets = [k for k, v in offset_counter.items() if v >= threshold]
            self._trigger_color = trigger_colors.most_common(1)[0][0]
            self._conf = min(0.8, len(self._expand_offsets) / 20) if self._expand_offsets else 0.0
            self._conf = self._validate(train_pairs)
        else:
            self._conf = 0.0

    def apply(self, test_input):
        if not self._expand_offsets:
            return None
        R, C = shape(test_input)
        bg_val = bg(test_input)
        g = clone(test_input)
        for r in range(R):
            for c in range(C):
                cv = test_input[r][c]
                if (self._trigger_color is None or cv == self._trigger_color) and cv != bg_val:
                    for dr, dc, new_color in self._expand_offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == bg_val:
                            g[nr][nc] = new_color
        return g


# ══════════════════════════════════════════════════════════════════
# 9.  Palette Remap Solver
#     Learns bijective colour→colour substitution table.
# ══════════════════════════════════════════════════════════════════

class PaletteRemapSolver(BaseSolver):
    name = "PaletteRemapSolver"

    def __init__(self):
        super().__init__()
        self._mapping = {}

    def learn(self, train_pairs):
        vote = defaultdict(Counter)
        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) != shape(out):
                continue
            R, C = shape(inp)
            for r in range(R):
                for c in range(C):
                    vote[inp[r][c]][out[r][c]] += 1
        self._mapping = {k: max(v, key=v.get) for k, v in vote.items()}
        self._conf = self._validate(train_pairs)

    def apply(self, test_input):
        if not self._mapping:
            return None
        return remap_colors(test_input, self._mapping)


# ══════════════════════════════════════════════════════════════════
# 10.  Spatial Scaling Solver  (zoom in / zoom out)
# ══════════════════════════════════════════════════════════════════

class SpatialScalingSolver(BaseSolver):
    name = "SpatialScalingSolver"

    def __init__(self):
        super().__init__()
        self._scale = None   # integer scale factor

    def _detect_scale(self, inp, out):
        Ri, Ci = shape(inp)
        Ro, Co = shape(out)
        if Ro % Ri == 0 and Co % Ci == 0 and Ro // Ri == Co // Ci:
            return Ro // Ri
        if Ri % Ro == 0 and Ci % Co == 0 and Ri // Ro == Ci // Co:
            return -(Ri // Ro)   # negative = zoom out
        return None

    def learn(self, train_pairs):
        scales = Counter()
        for p in train_pairs:
            s = self._detect_scale(p['input'], p['output'])
            if s is not None:
                scales[s] += 1
        if scales:
            self._scale = scales.most_common(1)[0][0]
            self._conf = self._validate(train_pairs)

    def _scale_up(self, grid, factor):
        return [[grid[r][c] for _ in range(factor) for c in range(len(grid[0]))]
                for r in range(len(grid)) for _ in range(factor)]

    def _scale_down(self, grid, factor):
        R, C = shape(grid)
        return [[grid[r * factor][c * factor]
                 for c in range(C // factor)]
                for r in range(R // factor)]

    def apply(self, test_input):
        if self._scale is None:
            return None
        if self._scale > 1:
            return self._scale_up(test_input, self._scale)
        elif self._scale < 0:
            return self._scale_down(test_input, -self._scale)
        return None


# ══════════════════════════════════════════════════════════════════
# 11.  Gravitational Fall Solver
#      Simulates objects falling toward a gravity axis.
# ══════════════════════════════════════════════════════════════════

class GravitationalFallSolver(BaseSolver):
    name = "GravitationalFallSolver"

    def __init__(self):
        super().__init__()
        self._direction = 'down'

    def _apply_gravity(self, grid, direction):
        R, C = shape(grid)
        bg_val = bg(grid)
        g = clone(grid)

        if direction == 'down':
            for c in range(C):
                col = [g[r][c] for r in range(R)]
                non_bg = [v for v in col if v != bg_val]
                new_col = [bg_val] * (R - len(non_bg)) + non_bg
                for r in range(R):
                    g[r][c] = new_col[r]
        elif direction == 'up':
            for c in range(C):
                col = [g[r][c] for r in range(R)]
                non_bg = [v for v in col if v != bg_val]
                new_col = non_bg + [bg_val] * (R - len(non_bg))
                for r in range(R):
                    g[r][c] = new_col[r]
        elif direction == 'left':
            for r in range(R):
                row = g[r]
                non_bg = [v for v in row if v != bg_val]
                g[r] = non_bg + [bg_val] * (C - len(non_bg))
        elif direction == 'right':
            for r in range(R):
                row = g[r]
                non_bg = [v for v in row if v != bg_val]
                g[r] = [bg_val] * (C - len(non_bg)) + non_bg
        return g

    def learn(self, train_pairs):
        best_dir, best_val = 'down', 0
        for d in ('down', 'up', 'left', 'right'):
            ok = sum(1 for p in train_pairs
                     if equal(self._apply_gravity(p['input'], d), p['output']))
            if ok > best_val:
                best_val, best_dir = ok, d
        self._direction = best_dir
        self._conf = best_val / len(train_pairs) if train_pairs else 0.0

    def apply(self, test_input):
        return self._apply_gravity(test_input, self._direction)


# ══════════════════════════════════════════════════════════════════
# 12.  Connected Path Solver
#      BFS-shortest-path filler between two anchor cells.
# ══════════════════════════════════════════════════════════════════

class ConnectedPathSolver(BaseSolver):
    name = "ConnectedPathSolver"

    def __init__(self):
        super().__init__()
        self._path_color = None
        self._anchor_colors = []

    def learn(self, train_pairs):
        path_colors = Counter()
        anchor_colors = Counter()
        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) != shape(out):
                continue
            changes = changed_cells(inp, out)
            if not changes:
                continue
            for r, c, v in changes:
                path_colors[v] += 1
            fgs = fg_colors(inp)
            for fc in fgs:
                anchor_colors[fc] += 1
        if path_colors:
            self._path_color = path_colors.most_common(1)[0][0]
            self._anchor_colors = [c for c, _ in anchor_colors.most_common(2)]
            self._conf = self._validate(train_pairs)

    def _bfs_path(self, grid, start, end, path_color):
        R, C = shape(grid)
        bg_val = bg(grid)
        parent = {start: None}
        q = deque([start])
        while q:
            r, c = q.popleft()
            if (r, c) == end:
                break
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r+dr, c+dc
                if (0<=nr<R and 0<=nc<C and (nr,nc) not in parent
                        and (grid[nr][nc] == bg_val or (nr,nc) == end)):
                    parent[(nr,nc)] = (r,c)
                    q.append((nr,nc))
        if end not in parent:
            return []
        path = []
        node = end
        while node is not None:
            path.append(node)
            node = parent[node]
        return path

    def apply(self, test_input):
        if self._path_color is None:
            return None
        R, C = shape(test_input)
        anchors = []
        for col in self._anchor_colors:
            for r in range(R):
                for c in range(C):
                    if test_input[r][c] == col:
                        anchors.append((r, c))
        if len(anchors) < 2:
            return None
        g = clone(test_input)
        for i in range(0, len(anchors) - 1):
            path = self._bfs_path(g, anchors[i], anchors[i+1], self._path_color)
            for r, c in path:
                g[r][c] = self._path_color
        return g


# ══════════════════════════════════════════════════════════════════
# 13.  Object Count Solver  (frequency analysis + output encoding)
# ══════════════════════════════════════════════════════════════════

class ObjectCountSolver(BaseSolver):
    name = "ObjectCountSolver"

    def __init__(self):
        super().__init__()
        self._count_to_output = {}

    def _count_objects(self, grid):
        return len(find_objects(grid))

    def learn(self, train_pairs):
        mapping = {}
        ok = 0
        for p in train_pairs:
            n = self._count_objects(p['input'])
            mapping[n] = p['output']
        self._count_to_output = mapping
        self._conf = self._validate(train_pairs)

    def apply(self, test_input):
        n = self._count_objects(test_input)
        return self._count_to_output.get(n)


# ══════════════════════════════════════════════════════════════════
# 14.  Template Inference Solver
#      Matches input patterns to known templates; pastes solution.
# ══════════════════════════════════════════════════════════════════

class TemplateInferenceSolver(BaseSolver):
    name = "TemplateInferenceSolver"

    def __init__(self):
        super().__init__()
        self._templates = []  # list of (inp_normalized, out)

    def _normalize(self, grid):
        """Remap colours so min=0, second=1, etc."""
        hist = color_histogram(grid)
        rank = {c: i for i, (c, _) in enumerate(hist.most_common())}
        return remap_colors(grid, rank), rank

    def learn(self, train_pairs):
        for p in train_pairs:
            norm_inp, rank = self._normalize(p['input'])
            self._templates.append((norm_inp, p['output'], rank))
        self._conf = min(0.7, len(self._templates) / 3)

    def apply(self, test_input):
        norm_test, _ = self._normalize(test_input)
        for norm_inp, out, _ in self._templates:
            if equal(norm_inp, norm_test):
                return out
        # Try rotation/flip variants
        for rot_fn in (lambda g: g, flip_h, flip_v, rotate_90cw, rotate_180, rotate_90ccw):
            transformed = rot_fn(test_input)
            norm_t, _ = self._normalize(transformed)
            for norm_inp, out, _ in self._templates:
                if equal(norm_t, norm_inp):
                    return out
        return None


# ══════════════════════════════════════════════════════════════════
# 15.  Spatial Dilation Solver
#      Expands each foreground cell to an NxN block.
# ══════════════════════════════════════════════════════════════════

class SpatialDilationSolver(BaseSolver):
    name = "SpatialDilationSolver"

    def __init__(self):
        super().__init__()
        self._factor = None

    def learn(self, train_pairs):
        factors = Counter()
        for p in train_pairs:
            Ri, Ci = shape(p['input'])
            Ro, Co = shape(p['output'])
            if Ri > 0 and Ci > 0 and Ro % Ri == 0 and Co % Ci == 0:
                factors[Ro // Ri] += 1
        if factors:
            self._factor = factors.most_common(1)[0][0]
            self._conf = self._validate(train_pairs)

    def apply(self, test_input):
        if self._factor is None or self._factor < 2:
            return None
        f = self._factor
        R, C = shape(test_input)
        out = make_grid(R * f, C * f, 0)
        for r in range(R):
            for c in range(C):
                v = test_input[r][c]
                for dr in range(f):
                    for dc in range(f):
                        out[r * f + dr][c * f + dc] = v
        return out


# ══════════════════════════════════════════════════════════════════
# 16.  Majority Vote Filter  (outlier removal / denoising)
#      Replaces each cell with neighbourhood majority vote colour.
# ══════════════════════════════════════════════════════════════════

class MajorityVoteFilterSolver(BaseSolver):
    name = "MajorityVoteFilterSolver"

    def __init__(self):
        super().__init__()
        self._radius = 1
        self._iterations = 1

    def _apply_filter(self, grid, radius):
        R, C = shape(grid)
        out = clone(grid)
        for r in range(R):
            for c in range(C):
                window = get_neighbourhood(grid, r, c, radius=radius)
                majority = Counter(window).most_common(1)[0][0]
                out[r][c] = majority
        return out

    def learn(self, train_pairs):
        best_score = 0
        for rad in (1, 2):
            for itr in (1, 2, 3):
                ok = 0
                for p in train_pairs:
                    if shape(p['input']) != shape(p['output']):
                        continue
                    g = p['input']
                    for _ in range(itr):
                        g = self._apply_filter(g, rad)
                    if equal(g, p['output']):
                        ok += 1
                score = ok / len(train_pairs) if train_pairs else 0
                if score > best_score:
                    best_score = score
                    self._radius, self._iterations = rad, itr
        self._conf = best_score

    def apply(self, test_input):
        g = clone(test_input)
        for _ in range(self._iterations):
            g = self._apply_filter(g, self._radius)
        return g


# ══════════════════════════════════════════════════════════════════
# 17.  Conditional Replacement Solver
#      Replaces a colour A with colour B/C based on neighbour context.
# ══════════════════════════════════════════════════════════════════

class ConditionalReplacementSolver(BaseSolver):
    name = "ConditionalReplacementSolver"

    def __init__(self):
        super().__init__()
        self._rules = {}   # (center_color, neighbor_signature) → new_color

    def learn(self, train_pairs):
        vote = defaultdict(Counter)
        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) != shape(out):
                continue
            R, C = shape(inp)
            for r in range(R):
                for c in range(C):
                    if inp[r][c] != out[r][c]:
                        ctx = get_neighbourhood_cross(inp, r, c)
                        sig = (inp[r][c], ctx)
                        vote[sig][out[r][c]] += 1
        self._rules = {k: max(v, key=v.get) for k, v in vote.items()}
        self._conf = self._validate(train_pairs)

    def apply(self, test_input):
        if not self._rules:
            return None
        R, C = shape(test_input)
        out = clone(test_input)
        for r in range(R):
            for c in range(C):
                ctx = get_neighbourhood_cross(test_input, r, c)
                sig = (test_input[r][c], ctx)
                if sig in self._rules:
                    out[r][c] = self._rules[sig]
        return out


# ══════════════════════════════════════════════════════════════════
# 18.  Composite Pipeline Solver
#      Sequences two solvers: first does coarse pass, second refines.
# ══════════════════════════════════════════════════════════════════

class CompositePipelineSolver(BaseSolver):
    name = "CompositePipelineSolver"

    def __init__(self, s1_cls, s2_cls):
        super().__init__()
        self._s1 = s1_cls()
        self._s2 = s2_cls()

    def learn(self, train_pairs):
        self._s1.learn(train_pairs)
        # Feed s1 predictions as "inputs" to s2
        synthetic = []
        for p in train_pairs:
            pred1 = None
            try:
                pred1 = self._s1.apply(p['input'])
            except Exception:
                pass
            if pred1 is not None and shape(pred1) == shape(p['output']):
                synthetic.append({'input': pred1, 'output': p['output']})
        if synthetic:
            self._s2.learn(synthetic)
        self._conf = self._validate(train_pairs)

    def apply(self, test_input):
        try:
            stage1 = self._s1.apply(test_input)
            if stage1 is None:
                return None
            stage2 = self._s2.apply(stage1)
            return stage2 if stage2 is not None else stage1
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════
# 19.  Perimeter Outline Solver
#      Draws border outline of objects with a learned colour.
# ══════════════════════════════════════════════════════════════════

class PerimeterOutlineSolver(BaseSolver):
    name = "PerimeterOutlineSolver"

    def __init__(self):
        super().__init__()
        self._outline_color = None

    def learn(self, train_pairs):
        colors = Counter()
        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) != shape(out):
                continue
            changes = changed_cells(inp, out)
            for _, _, v in changes:
                colors[v] += 1
        if colors:
            self._outline_color = colors.most_common(1)[0][0]
            self._conf = self._validate(train_pairs)

    def _perimeter_cells(self, grid):
        R, C = shape(grid)
        bg_val = bg(grid)
        perim = []
        for r in range(R):
            for c in range(C):
                if grid[r][c] != bg_val:
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = r+dr, c+dc
                        if 0<=nr<R and 0<=nc<C and grid[nr][nc] == bg_val:
                            perim.append((r, c))
                            break
        return perim

    def apply(self, test_input):
        if self._outline_color is None:
            return None
        g = clone(test_input)
        for r, c in self._perimeter_cells(test_input):
            g[r][c] = self._outline_color
        return g


# ══════════════════════════════════════════════════════════════════
# 20.  Recursive Structure Solver  (self-similar / tiling patterns)
# ══════════════════════════════════════════════════════════════════

class RecursiveStructureSolver(BaseSolver):
    name = "RecursiveStructureSolver"

    def __init__(self):
        super().__init__()
        self._tile = None
        self._reps_r = 1
        self._reps_c = 1

    def learn(self, train_pairs):
        for p in train_pairs:
            inp, out = p['input'], p['output']
            Ri, Ci = shape(inp)
            Ro, Co = shape(out)
            if Ro < Ri or Co < Ci:
                continue
            if Ro % Ri == 0 and Co % Ci == 0:
                rr, rc = Ro // Ri, Co // Ci
                # Verify tiling
                tiled = self._tile_grid(inp, rr, rc)
                if equal(tiled, out):
                    self._tile = inp
                    self._reps_r, self._reps_c = rr, rc
                    self._conf = self._validate(train_pairs)
                    return
        self._conf = 0.0

    def _tile_grid(self, tile, rr, rc):
        R, C = shape(tile)
        out = make_grid(R * rr, C * rc, 0)
        for tr in range(rr):
            for tc in range(rc):
                for r in range(R):
                    for c in range(C):
                        out[tr*R+r][tc*C+c] = tile[r][c]
        return out

    def apply(self, test_input):
        if self._tile is None:
            return None
        return self._tile_grid(test_input, self._reps_r, self._reps_c)


# ══════════════════════════════════════════════════════════════════
# 21.  Spatial Relation Solver  (relative-position object placement)
# ══════════════════════════════════════════════════════════════════

class SpatialRelationSolver(BaseSolver):
    name = "SpatialRelationSolver"

    def __init__(self):
        super().__init__()
        self._rel_rules = []   # list of (ref_color, tgt_color, dr, dc)

    def learn(self, train_pairs):
        rule_votes = defaultdict(Counter)
        for p in train_pairs:
            inp, out = p['input'], p['output']
            if shape(inp) != shape(out):
                continue
            inp_objs = find_objects(inp)
            out_objs = find_objects(out)
            for ref in inp_objs:
                rc_r = ref['bbox'][0], ref['bbox'][1]
                for tgt in out_objs:
                    if tgt['color'] != ref['color']:
                        rc_t = tgt['bbox'][0], tgt['bbox'][1]
                        dr = rc_t[0] - rc_r[0]
                        dc = rc_t[1] - rc_r[1]
                        rule_votes[(ref['color'], tgt['color'])][(dr, dc)] += 1

        self._rel_rules = []
        for (ref_c, tgt_c), delta_ctr in rule_votes.items():
            dr, dc = max(delta_ctr, key=delta_ctr.get)
            self._rel_rules.append((ref_c, tgt_c, dr, dc))
        self._conf = self._validate(train_pairs)

    def apply(self, test_input):
        if not self._rel_rules:
            return None
        R, C = shape(test_input)
        g = clone(test_input)
        for ref_c, tgt_c, dr, dc in self._rel_rules:
            comps = connected_components(test_input, color=ref_c)
            for comp in comps:
                r1, c1, r2, c2 = bounding_box(comp)
                nr, nc = r1 + dr, c1 + dc
                if 0 <= nr < R and 0 <= nc < C:
                    g[nr][nc] = tgt_c
        return g




# ──────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────
# TARGETED PATTERN SOLVERS
# ──────────────────────────────────────────────────────────────────


class RigidBodyAttractionSolver(BaseSolver):
    """
    Identifies a mobile object (mover) and a stationary target object (attractor).
    Slides the mover as a rigid body along a cardinal direction until it is
    immediately adjacent to the attractor.
    The direction is computed DYNAMICALLY per apply() call from mover→attractor
    centroid vector, so it handles tasks where direction varies across instances.
    learn() only validates that such mover/attractor roles are consistent across
    all training pairs.
    """
    name = "RigidBodyAttractionSolver"

    def __init__(self):
        super().__init__()
        self._mover_color   = None
        self._attract_color = None

    @staticmethod
    def _cells_of_color(grid, color):
        R, C = shape(grid)
        return [(r, c) for r in range(R) for c in range(C) if grid[r][c] == color]

    @staticmethod
    def _centroid(cells):
        return (sum(r for r, _ in cells) / len(cells),
                sum(c for _, c in cells) / len(cells))

    @staticmethod
    def _direction_from_delta(dr_raw, dc_raw):
        """Convert centroid delta to cardinal direction (dr, dc) ∈ {(0,1),(0,-1),(1,0),(-1,0)}."""
        if abs(dr_raw) >= abs(dc_raw):
            return (1 if dr_raw > 0 else -1, 0)
        else:
            return (0, 1 if dc_raw > 0 else -1)

    @staticmethod
    def _slide_until_adjacent(mover_cells, attract_cells, dr, dc, R, C):
        """Slide mover by (dr,dc) steps until min Manhattan distance to attractor == 1."""
        cells = list(mover_cells)
        att_set = set(attract_cells)
        for _ in range(max(R, C) * 2):
            nxt = [(r + dr, c + dc) for r, c in cells]
            if any(r < 0 or r >= R or c < 0 or c >= C for r, c in nxt):
                break
            if set(nxt) & att_set:
                break
            min_d = min(abs(r1 - r2) + abs(c1 - c2)
                        for r1, c1 in nxt for r2, c2 in att_set)
            if min_d < 1:
                break
            cells = nxt
            if min_d == 1:
                break
        return cells

    def learn(self, train_pairs):
        # Identify which color is the mover (changes position) and which is the
        # attractor (stays put), and validate consistency across all pairs.
        candidates = []
        for p in train_pairs:
            inp, out = p['input'], p['output']
            bg_val = bg(inp)
            hist_in = color_histogram(inp)
            hist_in.pop(bg_val, None)
            if len(hist_in) < 2:
                return
            colors = list(hist_in.keys())
            R, C = shape(inp)
            found = None
            for mc in colors:
                in_cells  = set(self._cells_of_color(inp, mc))
                out_cells = set(self._cells_of_color(out, mc))
                if in_cells == out_cells or not in_cells or not out_cells:
                    continue
                for ac in colors:
                    if ac == mc:
                        continue
                    ac_in  = set(self._cells_of_color(inp, ac))
                    ac_out = set(self._cells_of_color(out, ac))
                    if ac_in != ac_out or not ac_in:
                        continue
                    # Compute dynamic direction for THIS pair
                    cr_i, cc_i = self._centroid(list(in_cells))
                    cr_a, cc_a = self._centroid(list(ac_in))
                    dr_s, dc_s = self._direction_from_delta(cr_a - cr_i, cc_a - cc_i)
                    # Exact-reconstruction validation
                    final_pos = self._slide_until_adjacent(
                        list(in_cells), list(ac_in), dr_s, dc_s, R, C)
                    pred = clone(inp)
                    for r, c in in_cells:
                        pred[r][c] = bg_val
                    for r, c in final_pos:
                        pred[r][c] = mc
                    if equal(pred, out):
                        found = (mc, ac)
                        break
                if found:
                    break
            if found:
                candidates.append(found)

        if not candidates:
            return
        unique = set(candidates)
        if len(unique) == 1 and len(candidates) == len(train_pairs):
            self._mover_color, self._attract_color = candidates[0]
            self._conf = 0.85
        elif len(unique) == 1 and len(candidates) >= len(train_pairs) - 1:
            self._mover_color, self._attract_color = candidates[0]
            self._conf = 0.70

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if self._mover_color is None:
            return None
        R, C = shape(test_input)
        mover   = self._cells_of_color(test_input, self._mover_color)
        attract = self._cells_of_color(test_input, self._attract_color)
        if not mover or not attract:
            return None
        # Compute direction dynamically from mover→attractor centroid
        cr_m, cc_m = self._centroid(mover)
        cr_a, cc_a = self._centroid(attract)
        dr, dc = self._direction_from_delta(cr_a - cr_m, cc_a - cc_m)
        final_pos = self._slide_until_adjacent(mover, attract, dr, dc, R, C)
        g = clone(test_input)
        bg_val = bg(test_input)
        for r, c in mover:
            g[r][c] = bg_val
        for r, c in final_pos:
            g[r][c] = self._mover_color
        return g

# ──────────────────────────────────────────────────────────────────


class FrameSymmetryExpandSolver(BaseSolver):
    """
    Detects a hollow rectangular frame (perimeter pixels of one color) and
    an interior object (another color).  Completes the interior object to be
    symmetric (horizontal, vertical, or both) within the bounding frame.
    Accepts both solid frames and notched-corner frames where the four
    corner cells may be absent.
    """
    name = "FrameSymmetryExpandSolver"

    def __init__(self):
        super().__init__()
        self._frame_color = None
        self._inner_color = None
        self._sym_h       = False
        self._sym_v       = False

    @staticmethod
    def _detect_frame(grid):
        """
        Return (frame_color, r1, c1, r2, c2) for a rectangular frame, or None.
        Handles solid, notched-corner, wing-extended, and notched-top/bottom frames.
        Strategy: use (leftmost, rightmost) fc-cell per row to identify candidate
        column spans; find topmost/bottommost rows sharing the same span; validate
        interior vertical sides with ±1 tolerance for wing-extended rows.
        """
        from collections import defaultdict
        R, C = shape(grid)
        hist = color_histogram(grid)
        hist.pop(bg(grid), None)
        for fc, _ in sorted(hist.items(), key=lambda x: -x[1]):
            # Per-row: (c_left, c_right) using leftmost/rightmost fc cell
            row_span = {}
            for r in range(R):
                fc_cols_in_row = [c for c in range(C) if grid[r][c] == fc]
                if len(fc_cols_in_row) >= 2:
                    row_span[r] = (fc_cols_in_row[0], fc_cols_in_row[-1])

            if not row_span:
                continue

            # Group rows by their (c_left, c_right) span
            span_to_rows = defaultdict(list)
            for r, span in row_span.items():
                if span[1] - span[0] >= 2:
                    span_to_rows[span].append(r)

            best_result = None
            best_score  = 0
            for (c1, c2), rows in span_to_rows.items():
                if len(rows) < 2:
                    continue
                r1, r2 = min(rows), max(rows)
                if r2 - r1 < 2:
                    continue
                interior = list(range(r1 + 1, r2))
                if interior:
                    # Allow ±1 wing-replacement: side may be one column beyond c1/c2
                    def has_left(r, c1=c1):
                        return (grid[r][c1] == fc or
                                (c1 > 0 and grid[r][c1-1] == fc))
                    def has_right(r, c2=c2):
                        return (grid[r][c2] == fc or
                                (c2 < C-1 and grid[r][c2+1] == fc))
                    if not all(has_left(r) and has_right(r) for r in interior):
                        continue
                score = (c2 - c1) * (r2 - r1)
                if score > best_score:
                    best_score = score
                    best_result = (fc, r1, c1, r2, c2)

            if best_result:
                return best_result
        return None
    def learn(self, train_pairs):
        for p in train_pairs:
            inp, out = p['input'], p['output']
            res = self._detect_frame(inp)
            if res is None:
                return
            fc, r1, c1, r2, c2 = res
            # dominant inner color
            inner_hist = {}
            for r in range(r1 + 1, r2):
                for c in range(c1 + 1, c2):
                    v = inp[r][c]
                    if v != 0 and v != fc:
                        inner_hist[v] = inner_hist.get(v, 0) + 1
            if not inner_hist:
                return
            ic = max(inner_hist, key=inner_hist.get)
            H = r2 - r1 - 1;  W = c2 - c1 - 1
            if H == 0 or W == 0:
                return
            inner_out = [[out[r][c] for c in range(c1 + 1, c2)]
                         for r in range(r1 + 1, r2)]
            h_sym = all(inner_out[i][j] == inner_out[i][W - 1 - j]
                        for i in range(H) for j in range(W))
            v_sym = all(inner_out[i][j] == inner_out[H - 1 - i][j]
                        for i in range(H) for j in range(W))
            if not h_sym and not v_sym:
                return
            if self._frame_color is None:
                self._frame_color = fc
                self._inner_color = ic
                self._sym_h = h_sym
                self._sym_v = v_sym
                self._conf  = 0.78
            elif self._frame_color != fc or self._inner_color != ic:
                return

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if self._frame_color is None:
            return None
        fc = self._frame_color
        ic = self._inner_color
        R, C = shape(test_input)
        g = clone(test_input)

        # Find all connected components of frame-color cells
        visited = [[False]*C for _ in range(R)]
        def bfs(sr, sc):
            comp = []
            q = [(sr, sc)]
            visited[sr][sc] = True
            while q:
                r, c = q.pop()
                comp.append((r, c))
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and test_input[nr][nc] == fc:
                        visited[nr][nc] = True
                        q.append((nr, nc))
            return comp

        ccs = []
        for r in range(R):
            for c in range(C):
                if test_input[r][c] == fc and not visited[r][c]:
                    ccs.append(bfs(r, c))

        any_change = False
        for cc in ccs:
            cc_set = set(cc)
            # Build masked grid: fc only for this CC, else 0
            masked = [[0]*C for _ in range(R)]
            for r2, c2 in cc:
                masked[r2][c2] = fc
            res = self._detect_frame(masked)
            if res is None:
                continue
            _, r1, c1, r2c, c2c = res
            H = r2c - r1 - 1;  W = c2c - c1 - 1
            if H <= 0 or W <= 0:
                continue
            if self._sym_h:
                for i in range(H):
                    for j in range(W):
                        r  = r1 + 1 + i;  c  = c1 + 1 + j
                        rj = c1 + 1 + (W - 1 - j)
                        if g[r][c] == ic:
                            g[r][rj] = ic;  any_change = True
                        elif g[r][rj] == ic:
                            g[r][c] = ic;   any_change = True
            if self._sym_v:
                for i in range(H):
                    for j in range(W):
                        r  = r1 + 1 + i;  ri = r1 + 1 + (H - 1 - i)
                        c  = c1 + 1 + j
                        if g[r][c] == ic:
                            g[ri][c] = ic;  any_change = True
                        elif g[ri][c] == ic:
                            g[r][c] = ic;   any_change = True
        return g if any_change else None


# ──────────────────────────────────────────────────────────────────


class SeparatorZoneFillSolver(BaseSolver):
    """
    Detects solid-color separator lines (full rows or columns of one color)
    that partition the grid into rectangular zones.  Learns a per-zone fill
    color from training examples and replicates it in the test output.
    """
    name = "SeparatorZoneFillSolver"

    def __init__(self):
        super().__init__()
        self._sep_color  = None
        self._axis       = None     # 'row' | 'col'
        self._zone_fills = []       # list of fill colors per zone index

    @staticmethod
    def _find_sep_lines(grid, axis):
        R, C = shape(grid)
        lines = []
        if axis == 'row':
            for r in range(R):
                vals = set(grid[r])
                if len(vals) == 1:
                    lines.append((r, vals.pop()))
        else:
            for c in range(C):
                vals = set(grid[r][c] for r in range(R))
                if len(vals) == 1:
                    lines.append((c, vals.pop()))
        return lines

    @staticmethod
    def _zone_ranges(sep_indices, total):
        boundaries = [-1] + sep_indices + [total]
        zones = []
        for i in range(len(boundaries) - 1):
            s = boundaries[i] + 1
            e = boundaries[i + 1]
            if s < e:
                zones.append((s, e))
        return zones

    def _zone_dominant_color(self, grid, zone, axis, bg_val):
        R, C = shape(grid)
        cells = []
        if axis == 'row':
            s, e = zone
            for r in range(s, e):
                for c in range(C):
                    if grid[r][c] != bg_val:
                        cells.append(grid[r][c])
        else:
            s, e = zone
            for r in range(R):
                for c in range(s, e):
                    if grid[r][c] != bg_val:
                        cells.append(grid[r][c])
        if not cells:
            return None
        from collections import Counter
        return Counter(cells).most_common(1)[0][0]

    def learn(self, train_pairs):
        for axis in ('row', 'col'):
            ok = True
            sep_color_cand = None
            zone_fills_cand = None
            for p in train_pairs:
                inp, out = p['input'], p['output']
                bg_val = bg(inp)
                lines = self._find_sep_lines(inp, axis)
                if not lines:
                    ok = False; break
                sep_idx  = [idx for idx, _ in lines]
                sep_cols = set(col for _, col in lines)
                if len(sep_cols) != 1:
                    ok = False; break
                sc = sep_cols.pop()
                total = shape(inp)[0] if axis == 'row' else shape(inp)[1]
                zones = self._zone_ranges(sep_idx, total)
                if not zones:
                    ok = False; break
                fills = [self._zone_dominant_color(out, z, axis, bg_val)
                         for z in zones]
                if sep_color_cand is None:
                    sep_color_cand = sc
                    zone_fills_cand = fills
                else:
                    if sep_color_cand != sc or zone_fills_cand != fills:
                        ok = False; break
            if ok and sep_color_cand is not None and zone_fills_cand:
                self._sep_color  = sep_color_cand
                self._axis       = axis
                self._zone_fills = zone_fills_cand
                self._conf       = 0.72
                return

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if self._sep_color is None:
            return None
        bg_val = bg(test_input)
        lines  = self._find_sep_lines(test_input, self._axis)
        if not lines:
            return None
        sep_idx = [idx for idx, _ in lines]
        total   = shape(test_input)[0] if self._axis == 'row' else shape(test_input)[1]
        zones   = self._zone_ranges(sep_idx, total)
        if len(zones) != len(self._zone_fills):
            return None
        R, C = shape(test_input)
        g = clone(test_input)
        for (s, e), fc in zip(zones, self._zone_fills):
            if fc is None:
                continue
            if self._axis == 'row':
                for r in range(s, e):
                    for c in range(C):
                        if g[r][c] == bg_val:
                            g[r][c] = fc
            else:
                for r in range(R):
                    for c in range(s, e):
                        if g[r][c] == bg_val:
                            g[r][c] = fc
        return g


# ──────────────────────────────────────────────────────────────────


class EmbeddedPatternExtractSolver(BaseSolver):
    """
    Handles tasks where the output is a crop of a connected sub-object from
    a larger padded input grid.  Learns whether to use a bounding-box crop or
    to extract a specific connected component (by size rank, smallest first).
    """
    name = "EmbeddedPatternExtractSolver"

    def __init__(self):
        super().__init__()
        self._rule    = None   # 'bbox_crop' | 'inner_object'
        self._cc_rank = None   # 0 = smallest matching CC

    def _bbox_of_fg(self, grid):
        bg_val = bg(grid)
        R, C   = shape(grid)
        rows = [r for r in range(R)
                if any(grid[r][c] != bg_val for c in range(C))]
        cols = [c for c in range(C)
                if any(grid[r][c] != bg_val for r in range(R))]
        if not rows or not cols:
            return None
        return rows[0], cols[0], rows[-1], cols[-1]

    def learn(self, train_pairs):
        for p in train_pairs:
            inp, out = p['input'], p['output']
            iR, iC = shape(inp)
            oR, oC = shape(out)
            if oR >= iR and oC >= iC:
                return
            # 1. Simple bounding-box crop
            bbox = self._bbox_of_fg(inp)
            if bbox is not None:
                r1, c1, r2, c2 = bbox
                extracted = extract_subgrid(inp, r1, c1, r2, c2)
                if equal(extracted, out):
                    self._rule = 'bbox_crop'
                    self._conf = 0.82
                    return
            # 2. Try every CC by size rank (smallest first)
            comps = connected_components(inp, color=None, include_bg=False)
            if comps:
                for rank, comp in enumerate(sorted(comps, key=len)):
                    r1b, c1b, r2b, c2b = bounding_box(comp)
                    sub = extract_subgrid(inp, r1b, c1b, r2b, c2b)
                    if shape(sub) == (oR, oC) and equal(sub, out):
                        self._rule    = 'inner_object'
                        self._cc_rank = rank
                        self._conf    = 0.80
                        return

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if self._rule is None:
            return None
        if self._rule == 'bbox_crop':
            bbox = self._bbox_of_fg(test_input)
            if bbox is None:
                return None
            r1, c1, r2, c2 = bbox
            return extract_subgrid(test_input, r1, c1, r2, c2)
        if self._rule == 'inner_object':
            comps = connected_components(test_input, color=None, include_bg=False)
            if not comps:
                return None
            sorted_comps = sorted(comps, key=len)
            rank = min(self._cc_rank, len(sorted_comps) - 1)
            r1, c1, r2, c2 = bounding_box(sorted_comps[rank])
            return extract_subgrid(test_input, r1, c1, r2, c2)
        return None


# ──────────────────────────────────────────────────────────────────


class AngularConcaveFillSolver(BaseSolver):
    """
    Identifies an L-shaped or angular wall structure (a foreground color
    forming axis-aligned corner shapes).  Background cells at the inner
    concave corner — i.e., cells adjacent to wall pixels on both a horizontal
    and a vertical neighbour simultaneously — are filled with a learned color.
    """
    name = "AngularConcaveFillSolver"

    def __init__(self):
        super().__init__()
        self._wall_color = None
        self._fill_color = None

    @staticmethod
    def _concave_cells(grid, wall_c, bg_val):
        """
        Return the set of background cells that have at least one wall neighbour
        in the horizontal direction AND at least one in the vertical direction.
        """
        R, C = shape(grid)
        cells = set()
        for r in range(R):
            for c in range(C):
                if grid[r][c] != bg_val:
                    continue
                has_h = ((c > 0     and grid[r][c - 1] == wall_c) or
                         (c < C - 1 and grid[r][c + 1] == wall_c))
                has_v = ((r > 0     and grid[r - 1][c] == wall_c) or
                         (r < R - 1 and grid[r + 1][c] == wall_c))
                if has_h and has_v:
                    cells.add((r, c))
        return cells

    def learn(self, train_pairs):
        candidate = None
        for p in train_pairs:
            inp, out = p['input'], p['output']
            bg_val = bg(inp)
            # cells that were background and changed in the output
            changed = [(r, c) for r, c, _ in changed_cells(inp, out)
                       if inp[r][c] == bg_val and out[r][c] != bg_val]
            if not changed:
                return
            # fill color = most common new value at changed cells
            fill_ctr = {}
            for r, c in changed:
                fill_ctr[out[r][c]] = fill_ctr.get(out[r][c], 0) + 1
            fc = max(fill_ctr, key=fill_ctr.get)
            # wall color = most common foreground color other than fill color
            hist = color_histogram(inp)
            hist.pop(bg_val, None)
            hist.pop(fc, None)
            if not hist:
                return
            wc = max(hist, key=hist.get)
            # validate: concave corner cells ≈ changed cells
            corners   = self._concave_cells(inp, wc, bg_val)
            chg_set   = set(changed)
            if not corners:
                return
            overlap = len(corners & chg_set)
            if overlap < 0.7 * max(len(corners), len(chg_set)):
                return
            if candidate is None:
                candidate = (wc, fc)
            elif candidate != (wc, fc):
                return
        if candidate:
            self._wall_color = candidate[0]
            self._fill_color = candidate[1]
            self._conf = 0.80

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if self._wall_color is None:
            return None
        bg_val = bg(test_input)
        g      = clone(test_input)
        cells  = self._concave_cells(test_input, self._wall_color, bg_val)
        for r, c in cells:
            g[r][c] = self._fill_color
        return g




# ──────────────────────────────────────────────────────────────────
# BoundingBoxEnclosureReplaceSolver
# Finds the bounding box of all cells with a learned "marker" color,
# then replaces all cells of a learned "target" color that fall inside
# that bounding box with a learned "replacement" color.
# ──────────────────────────────────────────────────────────────────
class BoundingBoxEnclosureReplaceSolver(BaseSolver):
    """
    Learns three colors from training pairs:
      marker_color  — acts as a boundary / frame
      target_color  — cells to be relabelled
      replace_color — the new label for target cells inside the bbox
    At inference: compute axis-aligned bounding box of all marker_color
    cells, then remap every target_color cell inside that bbox to
    replace_color.
    """
    name = "BoundingBoxEnclosureReplaceSolver"

    def __init__(self):
        super().__init__()
        self._marker  = None
        self._target  = None
        self._replace = None

    @staticmethod
    def _bbox(grid, color):
        rows = [r for r, row in enumerate(grid) for v in row if v == color]
        cols = [c for row in grid for c, v in enumerate(row) if v == color]
        if not rows:
            return None
        return min(rows), max(rows), min(cols), max(cols)

    def learn(self, train_pairs):
        candidate = None
        for p in train_pairs:
            inp, out = p['input'], p['output']
            # Find cells that changed value
            changes = [(r, c, inp[r][c], out[r][c])
                       for r, c, nv in changed_cells(inp, out)
                       for _ in [None]
                       if True]
            # rebuild with proper format
            changes = []
            for r, c, nv in changed_cells(inp, out):
                changes.append((r, c, inp[r][c], nv))
            if not changes:
                return

            # target_color  = old value at changed cells (should be unique)
            old_vals = {ov for _, _, ov, _ in changes}
            new_vals = {nv for _, _, _, nv in changes}
            if len(old_vals) != 1 or len(new_vals) != 1:
                return
            tgt = old_vals.pop()
            rep = new_vals.pop()
            if tgt == rep:
                return

            # marker_color: a color present in inp that is NOT tgt, NOT rep,
            # whose bbox contains ALL changed cells AND whose interior
            # target-color cells exactly match the changed cells.
            bg_v = bg(inp)
            inp_colors = sorted(set(
                v for row in inp for v in row
                if v != tgt and v != rep and v != bg_v
            ))
            found_marker = None
            actual = sorted([(r, c) for r, c, _, _ in changes])
            for mc in inp_colors:
                bb = self._bbox(inp, mc)
                if bb is None:
                    continue
                r0, r1, c0, c1 = bb
                # All changed cells must lie inside bbox
                if not all(r0 <= r <= r1 and c0 <= c <= c1
                           for r, c, _, _ in changes):
                    continue
                # The target-color cells inside bbox must equal changed cells
                expected = sorted([(r, c) for r in range(r0, r1+1)
                                   for c in range(c0, c1+1)
                                   if inp[r][c] == tgt])
                if expected == actual:
                    found_marker = mc
                    break
            if found_marker is None:
                return

            this = (found_marker, tgt, rep)
            if candidate is None:
                candidate = this
            elif candidate != this:
                return

        if candidate:
            self._marker, self._target, self._replace = candidate
            self._conf = 0.88

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if self._marker is None:
            return None
        bb = self._bbox(test_input, self._marker)
        if bb is None:
            return None
        r0, r1, c0, c1 = bb
        g = clone(test_input)
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                if g[r][c] == self._target:
                    g[r][c] = self._replace
        return g


# ──────────────────────────────────────────────────────────────────
# GridSectionThresholdFillSolver
# For grids separated by a single "separator" color forming a full
# row+column cross-hatch, divides the grid into rectangular sections.
# Sections where a single non-bg, non-sep color appears ≥ threshold
# times get filled solid with that color; other sections are cleared
# to background.
# ──────────────────────────────────────────────────────────────────
class GridSectionThresholdFillSolver(BaseSolver):
    """
    Detects axis-aligned grid separators (rows and columns entirely of
    one color) that divide the input into rectangular sub-sections.
    Learns a threshold (default 2): sections with ≥ threshold cells of
    the same foreground color are filled solid with that color; sections
    below the threshold are cleared to background.
    """
    name = "GridSectionThresholdFillSolver"

    def __init__(self):
        super().__init__()
        self._sep_color = None
        self._threshold = 2
        self._fill_rule = "dominant"   # "dominant" = fill solid with dominant color

    @staticmethod
    def _find_separators(grid, sep_color):
        R, C = shape(grid)
        sep_rows = [r for r in range(R) if all(grid[r][c] == sep_color for c in range(C))]
        sep_cols = [c for c in range(C) if all(grid[r][c] == sep_color for r in range(R))]
        return sep_rows, sep_cols

    @staticmethod
    def _get_sections(grid, sep_rows, sep_cols):
        R, C = shape(grid)
        row_bounds = []
        prev = 0
        for sr in sorted(sep_rows):
            if sr > prev:
                row_bounds.append((prev, sr))
            prev = sr + 1
        if prev < R:
            row_bounds.append((prev, R))
        col_bounds = []
        prev = 0
        for sc in sorted(sep_cols):
            if sc > prev:
                col_bounds.append((prev, sc))
            prev = sc + 1
        if prev < C:
            col_bounds.append((prev, C))
        sections = []
        for (r0, r1) in row_bounds:
            for (c0, c1) in col_bounds:
                cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
                sections.append(cells)
        return sections

    def learn(self, train_pairs):
        # Step 1: detect separator color from first pair
        first_inp = train_pairs[0]['input']
        bg_val = bg(first_inp)

        # Find which color forms complete rows AND complete columns
        hist = color_histogram(first_inp)
        hist.pop(bg_val, None)
        sep_cand = None
        for col, cnt in hist.items():
            sr, sc = self._find_separators(first_inp, col)
            if sr and sc:
                sep_cand = col
                break
        if sep_cand is None:
            return

        # Step 2: validate rule — fill section(s) with MAX fg-cell count,
        # clear all other sections.  Rule must hold on every training pair.
        for p in train_pairs:
            inp, out = p['input'], p['output']
            bg_v = bg(inp)
            sr, sc = self._find_separators(inp, sep_cand)
            if not sr or not sc:
                return
            sections = self._get_sections(inp, sr, sc)

            # Count fg cells per section
            section_counts = []
            section_dominant = []
            for cells in sections:
                color_cnt = {}
                for r, c in cells:
                    v = inp[r][c]
                    if v != bg_v and v != sep_cand:
                        color_cnt[v] = color_cnt.get(v, 0) + 1
                dominant  = max(color_cnt, key=color_cnt.get) if color_cnt else None
                dom_count = color_cnt.get(dominant, 0) if dominant else 0
                section_counts.append(dom_count)
                section_dominant.append(dominant)

            max_count = max(section_counts)
            if max_count == 0:
                return  # no foreground cells

            # Validate output against max-count rule
            for idx, (cells, cnt, dom) in enumerate(
                    zip(sections, section_counts, section_dominant)):
                out_vals = {out[r][c] for r, c in cells}
                if cnt == max_count and dom is not None:
                    if out_vals != {dom}:
                        return
                else:
                    if out_vals != {bg_v}:
                        return

        self._sep_color = sep_cand
        self._conf = 0.85

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if self._sep_color is None:
            return None
        bg_v = bg(test_input)
        sr, sc = self._find_separators(test_input, self._sep_color)
        if not sr or not sc:
            return None
        sections = self._get_sections(test_input, sr, sc)
        # Count fg cells per section
        section_counts = []
        section_dominant = []
        for cells in sections:
            color_cnt = {}
            for r, c in cells:
                v = test_input[r][c]
                if v != bg_v and v != self._sep_color:
                    color_cnt[v] = color_cnt.get(v, 0) + 1
            dominant  = max(color_cnt, key=color_cnt.get) if color_cnt else None
            dom_count = color_cnt.get(dominant, 0) if dominant else 0
            section_counts.append(dom_count)
            section_dominant.append(dominant)
        max_count = max(section_counts) if section_counts else 0
        g = clone(test_input)
        for cells, cnt, dom in zip(sections, section_counts, section_dominant):
            fill = dom if (cnt == max_count and max_count > 0 and dom is not None) else bg_v
            for r, c in cells:
                g[r][c] = fill
        return g



# ══════════════════════════════════════════════════════════════════
# ENSEMBLE  –  run all solvers, self-validate, pick best
# ══════════════════════════════════════════════════════════════════


# ── New edge-marker solvers (injected) ────────────────────
# ══════════════════════════════════════════════════════════════════
# LPathConnectorSolver
#   Each foreground color has: one top-edge marker (defines its column)
#   and optionally one side-edge marker (defines its row).
#   Output draws a vertical segment from row 0 down to the side-marker row
#   (or fills the entire column if no side marker), then a horizontal
#   segment from the corner toward the nearest edge.
# ══════════════════════════════════════════════════════════════════

class LPathConnectorSolver(BaseSolver):
    name = "LPathConnectorSolver"

    def __init__(self):
        super().__init__()
        self._ok = False

    def _parse(self, inp):
        """Return (top_markers, side_markers) or None if the input doesn't fit."""
        R, C = len(inp), len(inp[0]) if inp else 0
        bg_c = bg(inp)
        top_markers  = {}   # color -> col  (row-0 non-corner cells)
        side_markers = {}   # color -> (row, 'left'|'right')
        for r in range(R):
            for c in range(C):
                v = inp[r][c]
                if v == bg_c:
                    continue
                on_top   = (r == 0)
                on_bot   = (r == R - 1)
                on_left  = (c == 0)
                on_right = (c == C - 1)
                on_edge  = on_top or on_bot or on_left or on_right
                if not on_edge:
                    return None                       # interior cell ⇒ reject
                if on_top and not on_left and not on_right:
                    if v in top_markers:
                        return None                   # duplicate top marker
                    top_markers[v] = c
                elif on_left and not on_top and not on_bot:
                    if v in side_markers:
                        return None
                    side_markers[v] = (r, 'left')
                elif on_right and not on_top and not on_bot:
                    if v in side_markers:
                        return None
                    side_markers[v] = (r, 'right')
                # corners and bottom-edge cells: ignore silently
        return top_markers, side_markers

    def _build(self, inp):
        parsed = self._parse(inp)
        if parsed is None:
            return None
        top_markers, side_markers = parsed
        R, C = len(inp), len(inp[0]) if inp else 0
        all_colors = set(top_markers) | set(side_markers)
        if not all_colors:
            return None
        # Every color must have a top-row marker
        if not all(col in top_markers for col in all_colors):
            return None
        out = [row[:] for row in inp]
        for color in all_colors:
            col_pos = top_markers[color]
            if color in side_markers:
                row_pos, side = side_markers[color]
                # Vertical segment: rows 0 .. row_pos at col_pos
                for r in range(row_pos + 1):
                    out[r][col_pos] = color
                # Horizontal segment toward the side edge
                if side == 'left':
                    for c in range(col_pos + 1):
                        out[row_pos][c] = color
                else:  # 'right'
                    for c in range(col_pos, C):
                        out[row_pos][c] = color
            else:
                # No side marker → fill entire column
                for r in range(R):
                    out[r][col_pos] = color
        return out

    def learn(self, pairs):
        self._ok = False
        if not pairs:
            return
        ok = sum(1 for p in pairs
                 if self._build(p['input']) is not None
                 and self._build(p['input']) == p['output'])
        if ok == len(pairs):
            self._ok = True
            self._conf = 0.92

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if not self._ok:
            return None
        return self._build(test_input)


# ══════════════════════════════════════════════════════════════════
# PeriodicStripeRepeatSolver
#   Exactly two foreground pixel-markers on the grid's border define
#   positions for alternating "stripes" (whole rows or whole columns).
#   The gap between the two markers determines the period; the pattern
#   is extended to the far end of the grid.
# ══════════════════════════════════════════════════════════════════

class PeriodicStripeRepeatSolver(BaseSolver):
    name = "PeriodicStripeRepeatSolver"

    def __init__(self):
        super().__init__()
        self._ok = False

    def _build(self, inp):
        R, C = len(inp), len(inp[0]) if inp else 0
        bg_c = bg(inp)

        # Collect all non-bg cells
        cells = [(r, c, inp[r][c]) for r in range(R) for c in range(C)
                 if inp[r][c] != bg_c]
        if len(cells) != 2:
            return None

        (r1, c1, v1), (r2, c2, v2) = cells

        def is_edge(r, c):
            return r == 0 or r == R - 1 or c == 0 or c == C - 1

        if not is_edge(r1, c1) or not is_edge(r2, c2):
            return None

        # Classify each cell: "row marker" (on left/right, not top/bottom),
        # "col marker" (on top/bottom, not left/right).
        def row_marker(r, c):
            return (c == 0 or c == C - 1) and r not in (0, R - 1)

        def col_marker(r, c):
            return (r == 0 or r == R - 1) and c not in (0, C - 1)

        out = [row[:] for row in inp]

        if row_marker(r1, c1) and row_marker(r2, c2):
            # Sort by row
            if r1 > r2:
                r1, c1, v1, r2, c2, v2 = r2, c2, v2, r1, c1, v1
            step = r2 - r1
            if step == 0:
                return None
            pos, ci = r1, 0
            colors = [v1, v2]
            while pos < R:
                col = colors[ci % 2]
                for c in range(C):
                    out[pos][c] = col
                pos += step
                ci += 1

        elif col_marker(r1, c1) and col_marker(r2, c2):
            # Sort by col
            if c1 > c2:
                r1, c1, v1, r2, c2, v2 = r2, c2, v2, r1, c1, v1
            step = c2 - c1
            if step == 0:
                return None
            pos, ci = c1, 0
            colors = [v1, v2]
            while pos < C:
                col = colors[ci % 2]
                for r in range(R):
                    out[r][pos] = col
                pos += step
                ci += 1

        else:
            return None

        return out

    def learn(self, pairs):
        self._ok = False
        if not pairs:
            return
        ok = sum(1 for p in pairs
                 if self._build(p['input']) is not None
                 and self._build(p['input']) == p['output'])
        if ok == len(pairs):
            self._ok = True
            self._conf = 0.91

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if not self._ok:
            return None
        return self._build(test_input)


# ══════════════════════════════════════════════════════════════════
# CrossIntersectionFillSolver
#   A single foreground color appears on the top row (defining active
#   columns) and on the right edge (defining active rows).  The output
#   places a learned fill color at every (active_row, active_col) cell.
# ══════════════════════════════════════════════════════════════════

class CrossIntersectionFillSolver(BaseSolver):
    name = "CrossIntersectionFillSolver"

    def __init__(self):
        super().__init__()
        self._ok = False
        self._fill_color = None

    def _parse(self, inp):
        """Return (marker_color, top_cols, right_rows) or None."""
        R, C = len(inp), len(inp[0]) if inp else 0
        bg_c = bg(inp)
        top_cols   = []
        right_rows = []
        marker_colors = set()

        for r in range(R):
            for c in range(C):
                v = inp[r][c]
                if v == bg_c:
                    continue
                on_top   = (r == 0)
                on_right = (c == C - 1)
                on_left  = (c == 0)
                on_bot   = (r == R - 1)
                # Must be on top row or right column only (no corners shared between both)
                if on_top and not on_right:
                    marker_colors.add(v)
                    top_cols.append(c)
                elif on_right and not on_top:
                    marker_colors.add(v)
                    right_rows.append(r)
                else:
                    return None   # unexpected position

        if len(marker_colors) != 1:
            return None
        if not top_cols or not right_rows:
            return None
        return marker_colors.pop(), top_cols, right_rows

    def learn(self, pairs):
        self._ok = False
        self._fill_color = None
        if not pairs:
            return

        ok_count = 0
        fill_colors_seen = set()

        for p in pairs:
            inp, out = p['input'], p['output']
            R, C = len(inp), len(inp[0]) if inp else 0
            if len(out) != R or (R and len(out[0]) != C):
                continue
            bg_c = bg(inp)

            parsed = self._parse(inp)
            if parsed is None:
                continue
            marker_c, top_cols, right_rows = parsed

            # Identify the fill color: present in output but absent in input
            inp_colors = {inp[r][c] for r in range(R) for c in range(C)}
            out_colors = {out[r][c] for r in range(R) for c in range(C)}
            new_colors = out_colors - inp_colors - {bg_c}
            if len(new_colors) != 1:
                continue
            fill_color = next(iter(new_colors))
            fill_colors_seen.add(fill_color)

            # Build expected
            expected = [row[:] for row in inp]
            for rr in right_rows:
                for cc in top_cols:
                    expected[rr][cc] = fill_color

            if expected == out:
                ok_count += 1

        if ok_count == len(pairs) and len(fill_colors_seen) == 1:
            self._ok = True
            self._fill_color = fill_colors_seen.pop()
            self._conf = 0.91

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if not self._ok:
            return None
        inp = test_input
        R, C = len(inp), len(inp[0]) if inp else 0
        bg_c = bg(inp)

        parsed = self._parse(inp)
        if parsed is None:
            return None
        _, top_cols, right_rows = parsed

        out = [row[:] for row in inp]
        for rr in right_rows:
            for cc in top_cols:
                out[rr][cc] = self._fill_color
        return out


# ══════════════════════════════════════════════════════════════════
# MeetInMiddleSolver
#   Two distinct colors appear on opposite (left/right) edges of the
#   same interior row.  The output fills the row: left color fills from
#   col 0 to mid-1, right color fills from mid+1 to C-1, and color 5
#   is placed at the exact midpoint (requires odd width).
# ══════════════════════════════════════════════════════════════════

class MeetInMiddleSolver(BaseSolver):
    name = "MeetInMiddleSolver"

    def __init__(self):
        super().__init__()
        self._ok = False

    def _build(self, inp):
        R, C = len(inp), len(inp[0]) if inp else 0
        if C % 2 == 0:
            return None   # need odd width for a single midpoint
        bg_c = bg(inp)
        mid_col = C // 2

        left_cells  = {}   # row -> color
        right_cells = {}   # row -> color

        for r in range(R):
            for c in range(C):
                v = inp[r][c]
                if v == bg_c:
                    continue
                # Left-edge or right-edge cells (any row)
                if c == 0:
                    if r in left_cells:
                        return None
                    left_cells[r] = v
                elif c == C - 1:
                    if r in right_cells:
                        return None
                    right_cells[r] = v
                else:
                    return None   # unexpected non-bg cell

        # Need at least one row with both markers
        paired = set(left_cells) & set(right_cells)
        if not paired:
            return None
        # All left and right cells must be paired
        if paired != set(left_cells) or paired != set(right_cells):
            return None

        out = [row[:] for row in inp]
        for r in paired:
            lc = left_cells[r]
            rc = right_cells[r]
            for c in range(mid_col):
                out[r][c] = lc
            out[r][mid_col] = 5          # midpoint junction color
            for c in range(mid_col + 1, C):
                out[r][c] = rc
        return out

    def learn(self, pairs):
        self._ok = False
        if not pairs:
            return
        ok = sum(1 for p in pairs
                 if self._build(p['input']) is not None
                 and self._build(p['input']) == p['output'])
        if ok == len(pairs):
            self._ok = True
            self._conf = 0.91

    def confidence(self):
        return self._conf

    def apply(self, test_input):
        if not self._ok:
            return None
        return self._build(test_input)

# ── NEW SOLVERS v4 ──────────────────────────────────────────────────────────
# ── PixelRowColFillSolver ─────────────────────────────────────────────────
class PixelRowColFillSolver(BaseSolver):
    """
    Each non-bg pixel marks its row or column for fill (axis learned from training).
    Multiple pixels of the same color each apply to their own row/column.
    Row fills take priority over column fills at intersections.
    """
    name = "PixelRowColFillSolver"

    def __init__(self):
        super().__init__()
        self._color_axis = {}

    def _infer_axis(self, inp, out):
        R, C = len(inp), len(inp[0])
        bg_col = bg(inp)
        pixels = {}
        for r in range(R):
            for c in range(C):
                v = inp[r][c]
                if v != bg_col:
                    pixels.setdefault(v, []).append((r, c))
        if not pixels:
            return None
        color_axis = {}
        for color, pts in pixels.items():
            best_axis = None
            for axis in ('row', 'col'):
                ok = True
                for (r, c) in pts:
                    if axis == 'row':
                        seq = out[r][:]
                    else:
                        seq = [out[rr][c] for rr in range(R)]
                    match = sum(1 for v in seq if v == color)
                    if match / len(seq) < 0.5:
                        ok = False
                        break
                if ok:
                    best_axis = axis
                    break
            if best_axis is None:
                return None
            if color in color_axis and color_axis[color] != best_axis:
                return None
            color_axis[color] = best_axis
        if len(color_axis) != len(pixels):
            return None
        return color_axis

    def learn(self, train_pairs):
        combined = {}
        for p in train_pairs:
            ca = self._infer_axis(p['input'], p['output'])
            if ca is None:
                self._conf = 0.0
                return
            for color, axis in ca.items():
                if color in combined and combined[color] != axis:
                    self._conf = 0.0
                    return
                combined[color] = axis
        if not combined:
            self._conf = 0.0
            return
        self._color_axis = combined
        self._conf = self._validate(train_pairs)

    def apply(self, input_grid):
        if not self._color_axis:
            return None
        R, C = len(input_grid), len(input_grid[0])
        bg_col = bg(input_grid)
        pixels = {}
        for r in range(R):
            for c in range(C):
                v = input_grid[r][c]
                if v != bg_col and v in self._color_axis:
                    pixels.setdefault(v, []).append((r, c))
        result = [row[:] for row in input_grid]
        for color, pts in pixels.items():
            if self._color_axis[color] == 'col':
                for (r, c) in pts:
                    for rr in range(R):
                        result[rr][c] = color
        for color, pts in pixels.items():
            if self._color_axis[color] == 'row':
                for (r, c) in pts:
                    for cc in range(C):
                        result[r][cc] = color
        return result


# ── CrossFillIntersectSolver ───────────────────────────────────────────────
class CrossFillIntersectSolver(BaseSolver):
    """
    Each non-bg pixel paints its entire row and column.
    Intersections of two different colors receive an intersection marker color
    learned from training.
    """
    name = "CrossFillIntersectSolver"

    def __init__(self):
        super().__init__()
        self._inter_color = None

    def learn(self, train_pairs):
        inter_color = None
        for p in train_pairs:
            inp, out = p['input'], p['output']
            R, C = len(inp), len(inp[0])
            bg_col = bg(inp)
            pixels = [(r, c, inp[r][c]) for r in range(R) for c in range(C) if inp[r][c] != bg_col]
            if len(pixels) < 2:
                self._conf = 0.0
                return
            found = set()
            for i, (ri, ci, vi) in enumerate(pixels):
                for j, (rj, cj, vj) in enumerate(pixels):
                    if i == j or vi == vj:
                        continue
                    found.add(out[ri][cj])
            if len(found) != 1:
                self._conf = 0.0
                return
            ic = found.pop()
            if inter_color is None:
                inter_color = ic
            elif inter_color != ic:
                self._conf = 0.0
                return
        if inter_color is None:
            self._conf = 0.0
            return
        self._inter_color = inter_color
        self._conf = self._validate(train_pairs)

    def apply(self, input_grid):
        if self._inter_color is None:
            return None
        R, C = len(input_grid), len(input_grid[0])
        bg_col = bg(input_grid)
        pixels = [(r, c, input_grid[r][c]) for r in range(R) for c in range(C) if input_grid[r][c] != bg_col]
        result = [row[:] for row in input_grid]
        for (ri, ci, vi) in pixels:
            for cc in range(C):
                if result[ri][cc] == bg_col or result[ri][cc] == vi:
                    result[ri][cc] = vi
            for rr in range(R):
                if result[rr][ci] == bg_col or result[rr][ci] == vi:
                    result[rr][ci] = vi
        for i, (ri, ci, vi) in enumerate(pixels):
            for j, (rj, cj, vj) in enumerate(pixels):
                if i == j or vi == vj:
                    continue
                result[ri][cj] = self._inter_color
                result[rj][ci] = self._inter_color
        return result


# ── GridSectionColorFillSolver ─────────────────────────────────────────────
class GridSectionColorFillSolver(BaseSolver):
    """
    Divider color forms complete rows/columns splitting the grid into rectangular
    sections.  Each section is assigned a fill color learned from training.
    """
    name = "GridSectionColorFillSolver"

    def __init__(self):
        super().__init__()
        self._divider_color = None
        self._section_fills = []
        self._n_sections = 0

    def _get_divider(self, grid):
        R, C = len(grid), len(grid[0])
        for color in range(1, 10):
            rows = [r for r in range(R) if all(grid[r][c] == color for c in range(C))]
            cols = [c for c in range(C) if all(grid[r][c] == color for r in range(R))]
            if rows or cols:
                return color, rows, cols
        return None, [], []

    def _sections(self, R, C, div_rows, div_cols):
        rb = sorted(set([-1] + div_rows + [R]))
        cb = sorted(set([-1] + div_cols + [C]))
        secs = []
        for i in range(len(rb) - 1):
            r0, r1 = rb[i] + 1, rb[i + 1]
            if r0 >= r1: continue
            for j in range(len(cb) - 1):
                c0, c1 = cb[j] + 1, cb[j + 1]
                if c0 >= c1: continue
                secs.append((r0, r1, c0, c1))
        return secs

    def learn(self, train_pairs):
        self._divider_color = None
        all_fills = []
        for p in train_pairs:
            inp, out = p['input'], p['output']
            R, C = len(inp), len(inp[0])
            dc, div_rows, div_cols = self._get_divider(inp)
            if dc is None:
                self._conf = 0.0
                return
            if self._divider_color is None:
                self._divider_color = dc
            elif self._divider_color != dc:
                self._conf = 0.0
                return
            bg_col = bg(inp)
            secs = self._sections(R, C, div_rows, div_cols)
            if len(secs) < 2:
                self._conf = 0.0
                return
            fills = []
            for (r0, r1, c0, c1) in secs:
                cells = [out[r][c] for r in range(r0, r1) for c in range(c0, c1)]
                if len(set(cells)) > 1:
                    self._conf = 0.0
                    return
                fills.append(cells[0])
            all_fills.append(fills)
        if not all_fills:
            self._conf = 0.0
            return
        self._section_fills = all_fills[-1]
        self._n_sections = len(self._section_fills)
        self._conf = self._validate(train_pairs)

    def apply(self, input_grid):
        if self._divider_color is None:
            return None
        R, C = len(input_grid), len(input_grid[0])
        dc, div_rows, div_cols = self._get_divider(input_grid)
        if dc is None or dc != self._divider_color:
            return None
        bg_col = bg(input_grid)
        secs = self._sections(R, C, div_rows, div_cols)
        if len(secs) != self._n_sections:
            return None
        result = [row[:] for row in input_grid]
        for idx, (r0, r1, c0, c1) in enumerate(secs):
            fill = self._section_fills[idx] if idx < len(self._section_fills) else bg_col
            if fill != bg_col:
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        result[r][c] = fill
        return result



ALL_SOLVER_CLASSES = [
    NeighborhoodRuleLearner,
    BilateralSymmetrySolver,
    QuadrantSymmetrySolver,
    TranspositionSymmetrySolver,
    RegionExtractionSolver,
    FloodFillEnclosureSolver,
    ObjectProjectionSolver,
    LocalContextExpansionSolver,
    PaletteRemapSolver,
    SpatialScalingSolver,
    GravitationalFallSolver,
    ConnectedPathSolver,
    ObjectCountSolver,
    TemplateInferenceSolver,
    SpatialDilationSolver,
    MajorityVoteFilterSolver,
    ConditionalReplacementSolver,
    PerimeterOutlineSolver,
    RecursiveStructureSolver,
    SpatialRelationSolver,
    RigidBodyAttractionSolver,
    FrameSymmetryExpandSolver,
    SeparatorZoneFillSolver,
    EmbeddedPatternExtractSolver,
    AngularConcaveFillSolver,
    BoundingBoxEnclosureReplaceSolver,
    GridSectionThresholdFillSolver,
    LPathConnectorSolver,
    PeriodicStripeRepeatSolver,
    CrossIntersectionFillSolver,
    MeetInMiddleSolver,
    PixelRowColFillSolver,
    CrossFillIntersectSolver,
    GridSectionColorFillSolver,
]

# Composite combinations
COMPOSITE_PAIRS = [
    (MajorityVoteFilterSolver, NeighborhoodRuleLearner),
    (PaletteRemapSolver, BilateralSymmetrySolver),
]


def solve_task(train_pairs, test_input, hint_category=None):
    """
    Run full solver ensemble on a single task.
    Returns (prediction, solver_name, confidence).
    """
    # Normalise: accept both (input, output) tuples and {'input':…,'output':…} dicts
    train_pairs = [
        p if isinstance(p, dict) else {'input': p[0], 'output': p[1]}
        for p in train_pairs
    ]

    best_pred   = None
    best_name   = None
    best_conf   = -1.0

    # Pre-compute average training change count for sanity gating
    def _avg_train_changes(pairs):
        total = 0; n = 0
        for p in pairs:
            inp, out = p['input'], p['output']
            R, C = len(inp), len(inp[0]) if inp else 0
            if len(out) != R or (R and len(out[0]) != C): continue
            total += sum(inp[i][j] != out[i][j] for i in range(R) for j in range(C))
            n += 1
        return total / n if n else 0.0

    def _pred_changes(test_inp, pred):
        R, C = len(test_inp), len(test_inp[0]) if test_inp else 0
        if len(pred) != R or (R and len(pred[0]) != C): return -1
        return sum(test_inp[i][j] != pred[i][j] for i in range(R) for j in range(C))

    avg_ch = _avg_train_changes(train_pairs)

    # Single solvers
    for cls in ALL_SOLVER_CLASSES:
        try:
            s = cls()
            s.learn(train_pairs)
            c = s.confidence()
            if c <= best_conf:
                continue
            pred = s.apply(test_input)
            if pred is None:
                continue
            # Self-validate: exact match on ALL training pairs ≥ 1 required
            train_ok = s._validate(train_pairs)
            final_conf = c * train_ok
            # Sanity: penalize predictions that change far fewer cells than training
            if avg_ch > 3:
                pch = _pred_changes(test_input, pred)
                if pch >= 0 and pch < avg_ch * 0.25:
                    final_conf *= 0.5
            if final_conf > best_conf:
                best_conf = final_conf
                best_pred = pred
                best_name = cls.name
        except Exception:
            pass

    # Composite two-stage pipelines
    for s1_cls, s2_cls in COMPOSITE_PAIRS:
        try:
            s = CompositePipelineSolver(s1_cls, s2_cls)
            s.learn(train_pairs)
            c = s.confidence()
            if c > best_conf:
                pred = s.apply(test_input)
                if pred is not None:
                    best_conf = c
                    best_pred = pred
                    best_name = f"{s1_cls.name}+{s2_cls.name}"
        except Exception:
            pass

    return best_pred, best_name, best_conf
