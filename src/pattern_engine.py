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


# ══════════════════════════════════════════════════════════════════
# ENSEMBLE  –  run all solvers, self-validate, pick best
# ══════════════════════════════════════════════════════════════════

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
    best_pred   = None
    best_name   = None
    best_conf   = -1.0

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
