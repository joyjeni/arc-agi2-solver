# ============================================================
# dsa_prepass.py  –  DSA T1 pre-pass (no LLM required)
#
# Nature / Engineering-inspired strategies:
#   1. CELLULAR AUTOMATA  (Conway/Wolfram)   → LOCAL_CHANGE tasks
#   2. CRYSTAL GROWTH / MORPHOGENESIS        → SYMH/SYMV tasks
#   3. COMPUTER VISION  (blob analysis)      → COLOR_REDUCTION tasks
#   4. SIGNAL PROCESSING (crop/filter)       → SHRINK/CROP tasks
#   5. INFORMATION THEORY (pattern lookup)   → UPSCALE tasks
#   6. PHYSICS (gravity, diffusion)          → existing solvers
#   7. GRAPH THEORY (object translation)     → MOVEMENT tasks
#   8. CONTROL THEORY (fixed-point iter.)    → stability transforms
#
# Runs high-precision DSA solvers on ALL test tasks before
# TTT/LLM inference starts.  Saves solved tasks to
#   /kaggle/working/dsa_solved.json
# so starter.py can skip them and cell9 merges into submission.
# ============================================================

import os
import copy
import json
from collections import Counter, defaultdict

# ── Types ────────────────────────────────────────────────────
Grid = list  # list[list[int]]


# ════════════════════════════════════════════════════════════
# Helper: grid validity
# ════════════════════════════════════════════════════════════
def _valid_grid(g):
    if g is None: return False
    if not isinstance(g, list) or len(g) == 0: return False
    if not isinstance(g[0], list) or len(g[0]) == 0: return False
    w = len(g[0])
    if any(len(r) != w for r in g): return False
    if any(not isinstance(c, int) or not (0 <= c <= 9)
           for r in g for c in r): return False
    return True


# ════════════════════════════════════════════════════════════
# BFS / flood-fill helpers
# ════════════════════════════════════════════════════════════
def bfs_flood_fill(grid, start_r, start_c, fill_color):
    """4-connected BFS flood fill from (start_r, start_c)."""
    g = copy.deepcopy(grid)
    rows, cols = len(g), len(g[0])
    target = g[start_r][start_c]
    if target == fill_color:
        return g
    frontier = [(start_r, start_c)]
    g[start_r][start_c] = fill_color
    while frontier:
        r, c = frontier.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and g[nr][nc] == target:
                g[nr][nc] = fill_color
                frontier.append((nr, nc))
    return g


def bfs_fill_enclosed(grid, bg_color=0):
    """Fill all interior regions fully enclosed (not touching border)."""
    g = copy.deepcopy(grid)
    rows, cols = len(g), len(g[0])
    border_bg = set()
    frontier = []
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) \
                    and g[r][c] == bg_color:
                if (r, c) not in border_bg:
                    border_bg.add((r, c))
                    frontier.append((r, c))
    while frontier:
        r, c = frontier.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols \
                    and (nr, nc) not in border_bg \
                    and g[nr][nc] == bg_color:
                border_bg.add((nr, nc))
                frontier.append((nr, nc))
    for r in range(rows):
        for c in range(cols):
            if g[r][c] == bg_color and (r, c) not in border_bg:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols \
                            and g[nr][nc] != bg_color:
                        g[r][c] = g[nr][nc]
                        break
    return g


# ════════════════════════════════════════════════════════════
# Transform helpers
# ════════════════════════════════════════════════════════════
def rot90(grid):
    rows, cols = len(grid), len(grid[0])
    return [[grid[rows - 1 - c][r] for c in range(cols)] for r in range(rows)]


def flip_h(grid):
    return [row[::-1] for row in grid]


def flip_v(grid):
    return grid[::-1]


def scale_up(grid, factor):
    result = []
    for row in grid:
        new_row = []
        for cell in row:
            new_row.extend([cell] * factor)
        for _ in range(factor):
            result.append(new_row[:])
    return result


def scale_down(grid, factor):
    return [row[::factor] for row in grid[::factor]]


def grid_xor(g1, g2):
    if len(g1) != len(g2): return None
    return [[c1 ^ c2 for c1, c2 in zip(r1, r2)]
            for r1, r2 in zip(g1, g2)]


def grid_and(g1, g2):
    if len(g1) != len(g2): return None
    return [[c1 & c2 for c1, c2 in zip(r1, r2)]
            for r1, r2 in zip(g1, g2)]


# ════════════════════════════════════════════════════════════
# Backtracking helpers
# ════════════════════════════════════════════════════════════
def _apply_transform(grid, name):
    ops = {
        'rot90':    lambda g: rot90(g),
        'rot180':   lambda g: rot90(rot90(g)),
        'rot270':   lambda g: rot90(rot90(rot90(g))),
        'flip_h':   lambda g: flip_h(g),
        'flip_v':   lambda g: flip_v(g),
        'flip_d':   lambda g: [[g[c][r] for c in range(len(g[0]))]
                                for r in range(len(g))],  # transpose
        'identity': lambda g: copy.deepcopy(g),
    }
    return ops[name](grid) if name in ops else None


def backtrack_transforms(pairs, test, max_depth=2):
    """Try single and chained geometric transforms."""
    names = ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'flip_d']
    for n1 in names:
        cand = all(_apply_transform(p['input'], n1) == p['output']
                   for p in pairs)
        if cand:
            return _apply_transform(test, n1)
    if max_depth >= 2:
        for n1 in names:
            for n2 in names:
                def chain(g, a=n1, b=n2):
                    t = _apply_transform(g, a)
                    return _apply_transform(t, b) if t else None
                if all(chain(p['input']) == p['output'] for p in pairs):
                    return chain(test)
    return None


def backtrack_color_map(pairs, test):
    """Find consistent color permutation mapping (bijective)."""
    mapping = {}
    for p in pairs:
        flat_in = [c for row in p['input'] for c in row]
        flat_out = [c for row in p['output'] for c in row]
        if len(flat_in) != len(flat_out):
            return None
        for ci, co in zip(flat_in, flat_out):
            if ci in mapping:
                if mapping[ci] != co:
                    return None
            else:
                mapping[ci] = co
    if not mapping:
        return None
    result = [[mapping.get(c, c) for c in row] for row in test]
    return result


def greedy_color_remap(pairs, test):
    """Frequency-based color remapping (greedy) with strict per-pair validation."""
    freq_map = {}
    for p in pairs:
        ci = Counter(c for row in p['input'] for c in row)
        co = Counter(c for row in p['output'] for c in row)
        sorted_i = sorted(ci, key=lambda x: -ci[x])
        sorted_o = sorted(co, key=lambda x: -co[x])
        local = dict(zip(sorted_i, sorted_o))
        for k, v in local.items():
            if k in freq_map and freq_map[k] != v:
                return None
            freq_map[k] = v
    if not freq_map:
        return None
    # Strict validation: applying the map to every training input must reproduce the training output exactly
    for p in pairs:
        pred = [[freq_map.get(c, c) for c in row] for row in p['input']]
        if pred != [list(r) for r in p['output']]:
            return None
    return [[freq_map.get(c, c) for c in row] for row in test]


def gravity(grid, direction='down'):
    g = copy.deepcopy(grid)
    rows, cols = len(g), len(g[0])
    if direction == 'down':
        for c in range(cols):
            col_vals = [g[r][c] for r in range(rows) if g[r][c] != 0]
            empty = rows - len(col_vals)
            for r in range(rows):
                g[r][c] = 0 if r < empty else col_vals[r - empty]
    elif direction == 'up':
        for c in range(cols):
            col_vals = [g[r][c] for r in range(rows) if g[r][c] != 0]
            for r in range(rows):
                g[r][c] = col_vals[r] if r < len(col_vals) else 0
    elif direction == 'left':
        for r in range(rows):
            row_vals = [c for c in g[r] if c != 0]
            g[r] = row_vals + [0] * (cols - len(row_vals))
    elif direction == 'right':
        for r in range(rows):
            row_vals = [c for c in g[r] if c != 0]
            g[r] = [0] * (cols - len(row_vals)) + row_vals
    return g


# ════════════════════════════════════════════════════════════
# Trie helpers
# ════════════════════════════════════════════════════════════
def build_row_trie(pairs):
    """Map input_row → output_row for every aligned training row."""
    trie = {}
    for p in pairs:
        for inp_row, out_row in zip(p['input'], p['output']):
            key = tuple(inp_row)
            trie[key] = tuple(out_row)
    return trie


def trie_predict(trie, test):
    result = []
    for row in test:
        pred = trie.get(tuple(row))
        if pred is None:
            return None
        result.append(list(pred))
    return result


# ════════════════════════════════════════════════════════════
# NEW: Advanced analysis helpers  (nature/engineering support)
# ════════════════════════════════════════════════════════════

def _most_common_color(grid):
    """Return the most frequent color in a grid."""
    counts = Counter(c for row in grid for c in row)
    return counts.most_common(1)[0][0] if counts else 0


def _color_counts(grid):
    return Counter(c for row in grid for c in row)


def _find_ccs(grid, conn=4):
    """
    BFS connected-component finder.
    Returns list of (color, [(r,c),...]) in discovery order.
    """
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dirs8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    dirs = dirs4 if conn == 4 else dirs8
    ccs = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c]:
                color = grid[r][c]
                stack = [(r, c)]
                visited[r][c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in dirs:
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and not visited[nr][nc]
                                and grid[nr][nc] == color):
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                ccs.append((color, cells))
    return ccs


def _crop_to_bbox(grid, bg):
    """Crop grid to bounding box of cells != bg. Returns None if all bg."""
    rows, cols = len(grid), len(grid[0])
    nz_rows = [r for r in range(rows)
                if any(grid[r][c] != bg for c in range(cols))]
    nz_cols = [c for c in range(cols)
                if any(grid[r][c] != bg for r in range(rows))]
    if not nz_rows or not nz_cols:
        return None
    t, b = nz_rows[0], nz_rows[-1]
    l, r_ = nz_cols[0], nz_cols[-1]
    return [row[l:r_+1] for row in grid[t:b+1]]


# ── Cellular Automata neighbor extractors ───────────────────

def _ca_nbrs_4(grid, r, c, bv=0):
    """Directional 4-neighbor tuple (N, S, W, E)."""
    rows, cols = len(grid), len(grid[0])
    n = grid[r-1][c] if r > 0 else bv
    s = grid[r+1][c] if r < rows-1 else bv
    w = grid[r][c-1] if c > 0 else bv
    e = grid[r][c+1] if c < cols-1 else bv
    return (n, s, w, e)


def _ca_nbrs_8(grid, r, c, bv=0):
    """Directional 8-neighbor tuple (NW,N,NE,W,E,SW,S,SE)."""
    rows, cols = len(grid), len(grid[0])
    def gv(rr, cc):
        return grid[rr][cc] if 0 <= rr < rows and 0 <= cc < cols else bv
    return (gv(r-1,c-1), gv(r-1,c), gv(r-1,c+1),
            gv(r,  c-1),             gv(r,  c+1),
            gv(r+1,c-1), gv(r+1,c), gv(r+1,c+1))


def _ca_nbrs_4s(grid, r, c, bv=0):
    """Sorted (direction-invariant) 4-neighbor tuple."""
    return tuple(sorted(_ca_nbrs_4(grid, r, c, bv)))


def _ca_nbrs_8s(grid, r, c, bv=0):
    """Sorted (direction-invariant) 8-neighbor tuple."""
    return tuple(sorted(_ca_nbrs_8(grid, r, c, bv)))


def _build_ca_rule(pairs, nbrs_fn):
    """
    Build CA lookup rule: (center, neighborhood) → new_center.
    Returns None if any inconsistency found across training pairs.
    Requires same-size input/output.
    """
    rule = {}
    for p in pairs:
        inp, out = p['input'], p['output']
        rows, cols = len(inp), len(inp[0])
        if len(out) != rows or len(out[0]) != cols:
            return None
        for r in range(rows):
            for c in range(cols):
                key = (inp[r][c], nbrs_fn(inp, r, c))
                val = out[r][c]
                if key in rule:
                    if rule[key] != val:
                        return None
                else:
                    rule[key] = val
    return rule


def _apply_ca(grid, rule, nbrs_fn):
    """Apply CA rule to a grid. Returns None if any unseen state encountered."""
    rows, cols = len(grid), len(grid[0])
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            key = (grid[r][c], nbrs_fn(grid, r, c))
            if key not in rule:
                return None
            out[r][c] = rule[key]
    return out


# ── Symmetry helpers ────────────────────────────────────────

def _sym_h_complete(grid, bg=0):
    """
    Crystal growth: fill bg cells using horizontal mirror symmetry.
    Non-bg cells act as seeds; bg gaps are filled from their mirror image.
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            mc = cols - 1 - c
            if result[r][c] == bg and result[r][mc] != bg:
                result[r][c] = result[r][mc]
            elif result[r][mc] == bg and result[r][c] != bg:
                result[r][mc] = result[r][c]
    return result


def _sym_v_complete(grid, bg=0):
    """Fill bg cells using vertical mirror symmetry."""
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        mr = rows - 1 - r
        for c in range(cols):
            if result[r][c] == bg and result[mr][c] != bg:
                result[r][c] = result[mr][c]
            elif result[mr][c] == bg and result[r][c] != bg:
                result[mr][c] = result[r][c]
    return result


def _sym_rot180_complete(grid, bg=0):
    """Fill bg cells using 180° rotational symmetry."""
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            mr, mc = rows - 1 - r, cols - 1 - c
            if result[r][c] == bg and result[mr][mc] != bg:
                result[r][c] = result[mr][mc]
            elif result[mr][mc] == bg and result[r][c] != bg:
                result[mr][mc] = result[r][c]
    return result


def _sym_diag_complete(grid, bg=0):
    """Fill bg cells using diagonal (transpose) symmetry."""
    rows, cols = len(grid), len(grid[0])
    if rows != cols:
        return grid
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == bg and result[c][r] != bg:
                result[r][c] = result[c][r]
            elif result[c][r] == bg and result[r][c] != bg:
                result[c][r] = result[r][c]
    return result


# ════════════════════════════════════════════════════════════
# DSA T1 solvers  –  ORIGINAL (high-precision, no LLM)
# ════════════════════════════════════════════════════════════

def _solve_identity(pairs, test):
    if all(p['input'] == p['output'] for p in pairs):
        return copy.deepcopy(test)
    return None


def _solve_constant_output(pairs, test):
    if not pairs: return None
    first = pairs[0]['output']
    if all(p['output'] == first for p in pairs):
        return copy.deepcopy(first)
    return None


def _solve_color_remap(pairs, test):
    return backtrack_color_map(pairs, test)


def _solve_color_freq_map(pairs, test):
    return greedy_color_remap(pairs, test)


def _solve_scale(pairs, test):
    if not pairs: return None
    p = pairs[0]; inp, out = p['input'], p['output']
    ir, ic = len(inp), len(inp[0]); or_, oc = len(out), len(out[0])
    if or_ > ir and oc > ic:
        rf, cf = or_ // ir, oc // ic
        if ir * rf == or_ and ic * cf == oc and rf == cf:
            if all(scale_up(p['input'], rf) == p['output'] for p in pairs):
                return scale_up(test, rf)
    if or_ < ir and oc < ic:
        rf, cf = ir // or_, ic // oc
        if or_ * rf == ir and oc * cf == ic and rf == cf:
            preds = [scale_down(p['input'], rf) for p in pairs]
            if all(preds) and all(preds[i] == pairs[i]['output']
                                   for i in range(len(pairs))):
                return scale_down(test, rf)
    return None


def _solve_gravity(pairs, test):
    for d in ['down', 'up', 'left', 'right']:
        if all(gravity(p['input'], d) == p['output'] for p in pairs):
            return gravity(test, d)
    return None


def _solve_fill_enclosed(pairs, test):
    for bg in range(10):
        if all(bfs_fill_enclosed(p['input'], bg) == p['output'] for p in pairs):
            return bfs_fill_enclosed(test, bg)
    return None


def _solve_direct_transform(pairs, test):
    return backtrack_transforms(pairs, test, max_depth=2)


def _solve_tiling(pairs, test):
    if not pairs: return None
    for factor in range(2, 8):
        if all(scale_up(p['input'], factor) == p['output'] for p in pairs):
            return scale_up(test, factor)
    return None


def _solve_flood_fill(pairs, test):
    """BFS flood-fill: detect consistent src_color → fill_color rule."""
    if not pairs: return None
    rule = None
    for p in pairs:
        inp, out = p['input'], p['output']
        if len(inp) != len(out): continue
        diffs = [(r, c) for r in range(len(inp))
                 for c in range(len(inp[0]))
                 if inp[r][c] != out[r][c]]
        if not diffs: continue
        src_c = inp[diffs[0][0]][diffs[0][1]]
        fill_c = out[diffs[0][0]][diffs[0][1]]
        if not all(inp[r][c] == src_c and out[r][c] == fill_c
                   for r, c in diffs):
            return None
        diff_set = set(diffs)
        frontier = [diffs[0]]; visited = set()
        while frontier:
            r, c = frontier.pop()
            if (r, c) in visited: continue
            visited.add((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (r+dr, c+dc)
                if nb in diff_set and nb not in visited:
                    frontier.append(nb)
        if visited != diff_set: return None
        candidate = (src_c, fill_c)
        if rule is None: rule = candidate
        elif rule != candidate: return None

    if rule is None: return None
    src_c, fill_c = rule
    for p in pairs:
        inp, out = p['input'], p['output']
        simulated = copy.deepcopy(inp)
        seeds = [(r, c) for r in range(len(inp))
                 for c in range(len(inp[0])) if inp[r][c] == src_c]
        for sr, sc in seeds:
            if simulated[sr][sc] == src_c:
                simulated = bfs_flood_fill(simulated, sr, sc, fill_c)
        if simulated != out: return None

    result = copy.deepcopy(test)
    seeds = [(r, c) for r in range(len(test))
             for c in range(len(test[0])) if test[r][c] == src_c]
    for sr, sc in seeds:
        if result[sr][sc] == src_c:
            result = bfs_flood_fill(result, sr, sc, fill_c)
    return result if _valid_grid(result) else None


def _solve_border_fill(pairs, test):
    """Greedy: fill outer ring with consistent color, preserve interior."""
    if not pairs: return None
    border_color = None
    for p in pairs:
        inp, out = p['input'], p['output']
        rows, cols = len(inp), len(inp[0]) if inp else 0
        if rows < 3 or cols < 3: return None
        if len(inp) != len(out) or len(inp[0]) != len(out[0]): return None
        if not all(inp[r][c] == out[r][c]
                   for r in range(1, rows-1)
                   for c in range(1, cols-1)):
            return None
        out_border = {out[r][c]
                      for r in range(rows) for c in range(cols)
                      if r == 0 or r == rows-1 or c == 0 or c == cols-1}
        if len(out_border) != 1: return None
        bc = out_border.pop()
        if border_color is None: border_color = bc
        elif border_color != bc: return None

    if border_color is None: return None
    tr, tc = len(test), len(test[0]) if test else 0
    if tr < 3 or tc < 3: return None
    result = copy.deepcopy(test)
    for r in range(tr):
        for c in range(tc):
            if r == 0 or r == tr-1 or c == 0 or c == tc-1:
                result[r][c] = border_color
    return result if _valid_grid(result) else None




def _solve_four_section_priority(pairs, test_inp):
    """281123b4: 4×19→4×4. Separator all-3s cols at 4,9,14.
    Sections S0=cols0-3,S1=cols5-8,S2=cols10-13,S3=cols15-18.
    Priority S2>S3>S0>S1 (first non-zero wins)."""
    PRIORITY = [2, 3, 0, 1]
    SEP_COLS  = [4, 9, 14]
    def solve(inp):
        rows = len(inp); cols = len(inp[0])
        for sc in SEP_COLS:
            if sc >= cols or not all(inp[r][sc] == 3 for r in range(rows)):
                return None
        sec_starts = [0, 5, 10, 15]
        sections = [[[inp[r][ss+c] for c in range(4)] for r in range(rows)] for ss in sec_starts]
        out = [[0]*4 for _ in range(rows)]
        for r in range(rows):
            for c in range(4):
                for si in PRIORITY:
                    v = sections[si][r][c]
                    if v != 0:
                        out[r][c] = v
                        break
        return out
    for p in pairs:
        if solve(p['input']) != p['output']:
            return None
    return solve(test_inp)


def _solve_xor_halves(pairs, test_inp):
    """34b99a2b: 5×9→5×4. col4=all-4s separator.
    Output=2 where exactly one of left/right halves non-zero, else 0."""
    def find_sep(inp, sep_val=4):
        rows = len(inp); cols = len(inp[0])
        for c in range(cols):
            if all(inp[r][c] == sep_val for r in range(rows)):
                return c
        return None
    def solve(inp):
        sc = find_sep(inp)
        if sc is None: return None
        rows = len(inp); w = sc
        right_start = sc + 1
        if len(inp[0]) - right_start != w: return None
        out = [[0]*w for _ in range(rows)]
        for r in range(rows):
            for c in range(w):
                lv = inp[r][c] != 0
                rv = inp[r][right_start + c] != 0
                out[r][c] = 2 if (lv ^ rv) else 0
        return out
    for p in pairs:
        if solve(p['input']) != p['output']:
            return None
    return solve(test_inp)


def _solve_block_connectivity(pairs, test_inp):
    """239be575: find two 2×2 blocks of value 2; BFS through 8-cells.
    Output [[8]] if connected, [[0]] if not."""
    from collections import deque
    def find_blocks(inp):
        rows = len(inp); cols = len(inp[0])
        blocks = []; visited = set()
        for r in range(rows-1):
            for c in range(cols-1):
                if (r,c) in visited: continue
                if (inp[r][c]==2 and inp[r][c+1]==2 and
                    inp[r+1][c]==2 and inp[r+1][c+1]==2):
                    cells = {(r,c),(r,c+1),(r+1,c),(r+1,c+1)}
                    blocks.append(cells); visited |= cells
        return blocks
    def connected(inp, b1, b2, pv=8):
        rows = len(inp); cols = len(inp[0])
        q = deque(b1); seen = set(b1)
        while q:
            r, c = q.popleft()
            for dr in range(-1,2):
                for dc in range(-1,2):
                    if dr==0 and dc==0: continue
                    nr, nc = r+dr, c+dc
                    if (nr,nc) in seen: continue
                    if 0<=nr<rows and 0<=nc<cols:
                        if (nr,nc) in b2: return True
                        if inp[nr][nc] == pv:
                            seen.add((nr,nc)); q.append((nr,nc))
        return False
    def solve(inp):
        blocks = find_blocks(inp)
        if len(blocks) != 2: return None
        return [[8]] if connected(inp, blocks[0], blocks[1]) else [[0]]
    for p in pairs:
        if solve(p['input']) != p['output']:
            return None
    return solve(test_inp)


def _solve_or_halves_1(pairs, test_inp):
    """195ba7dc: 5×13→5×6. col6=all-2s separator.
    output[r][c]=1 if left[r][c] or right[r][c] non-zero, else 0."""
    def find_sep(inp, sep_val=2):
        rows = len(inp); cols = len(inp[0])
        for c in range(cols):
            if all(inp[r][c] == sep_val for r in range(rows)):
                return c
        return None
    def solve(inp):
        sc = find_sep(inp)
        if sc is None: return None
        rows = len(inp); hw = sc; rs = sc + 1
        if len(inp[0]) - rs != hw: return None
        out = [[0]*hw for _ in range(rows)]
        for r in range(rows):
            for c in range(hw):
                out[r][c] = 1 if (inp[r][c] != 0 or inp[r][rs+c] != 0) else 0
        return out
    for p in pairs:
        if solve(p['input']) != p['output']:
            return None
    return solve(test_inp)


def _solve_sparse_to_blocks(pairs, test_inp):
    """33067df9: sparse grid (non-zero at odd rows/cols) → 26×26 output.
    Block dims from grid size. Adjacent same-color runs merge vertically."""
    from collections import defaultdict
    OUT_SIZE = 26; BORDER = 2; AVAIL = 22

    def solve(inp):
        rows = len(inp); cols = len(inp[0])
        N_r = (rows - 1) // 2
        N_c = (cols - 1) // 2
        if N_r == 0 or N_c == 0: return None
        bh = (AVAIL - 2*(N_r-1)) // N_r
        bw = (AVAIL - 2*(N_c-1)) // N_c
        r_sp = bh + 2; c_sp = bw + 2
        row_entries = defaultdict(list)
        for r in range(rows):
            if r % 2 == 0: continue
            for c in range(cols):
                if c % 2 == 0: continue
                v = inp[r][c]
                if v == 0: continue
                row_entries[r//2].append((c//2, v))
        all_runs = []
        for rb in range(N_r):
            entries = sorted(row_entries.get(rb, []))
            cur = []
            for cb, v in entries:
                if cur and cur[-1][2] == v and cb == cur[-1][1]+1:
                    cur[-1][1] = cb
                else:
                    cur.append([cb, cb, v])
            for s_cb, e_cb, v in cur:
                cs = BORDER + s_cb*c_sp
                ce = BORDER + e_cb*c_sp + bw - 1
                all_runs.append((rb, s_cb, e_cb, v, cs, ce))
        key_rb = defaultdict(list)
        for (rb, s_cb, e_cb, v, cs, ce) in all_runs:
            key_rb[(s_cb, e_cb, v, cs, ce)].append(rb)
        out = [[0]*OUT_SIZE for _ in range(OUT_SIZE)]
        for (s_cb, e_cb, v, cs, ce), rbs in key_rb.items():
            rbs = sorted(rbs)
            chains = []; chain = [rbs[0]]
            for rb in rbs[1:]:
                if rb == chain[-1]+1: chain.append(rb)
                else: chains.append(chain); chain = [rb]
            chains.append(chain)
            for chain in chains:
                rs = BORDER + chain[0]*r_sp
                re = BORDER + chain[-1]*r_sp + bh - 1
                for r in range(rs, re+1):
                    for c in range(cs, ce+1):
                        out[r][c] = v
        return out

    for p in pairs:
        if solve(p['input']) != p['output']:
            return None
    return solve(test_inp)




def _solve_recolor_small_components(pairs, test_inp):
    """12eac192 + 1e5d6875: small connected components (size<=2) → color 3; large ones unchanged."""
    from collections import deque
    def solve(grid):
        rows, cols = len(grid), len(grid[0])
        visited = [[False]*cols for _ in range(rows)]
        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0 or visited[r][c]:
                    continue
                color = grid[r][c]
                # BFS to find component
                q = deque([(r, c)])
                visited[r][c] = True
                component = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            component.append((nr, nc))
                if len(component) <= 2:
                    for (cr2, cc2) in component:
                        result[cr2][cc2] = 3
        return result
    for p in pairs:
        if solve(p['input']) != p['output']:
            return None
    return solve(test_inp)


def _solve_most_frequent_below_sep(pairs, test_inp):
    """27a77e38: find separator row of all-5s; count values above; place most-frequent below at last_row,mid_col."""
    def solve(grid):
        rows, cols = len(grid), len(grid[0])
        sep_row = None
        for r in range(rows):
            if all(grid[r][c] == 5 for c in range(cols)):
                sep_row = r
                break
        if sep_row is None:
            return None
        # count values above separator (exclude 0 and 5)
        from collections import Counter
        cnt = Counter()
        for r in range(sep_row):
            for c in range(cols):
                v = grid[r][c]
                if v != 0:
                    cnt[v] += 1
        if not cnt:
            return None
        most_freq = cnt.most_common(1)[0][0]
        result = [row[:] for row in grid]
        last_row = rows - 1
        mid_col = cols // 2
        result[last_row][mid_col] = most_freq
        return result
    for p in pairs:
        out = solve(p['input'])
        if out is None or out != p['output']:
            return None
    return solve(test_inp)


def _solve_template_color_markers(pairs, test_inp):
    """12997ef3: shape of 1s as template; isolated non-zero non-1 cells are color markers.
       If markers in same row → stack horizontally; if same col → stack vertically."""
    def get_template(grid):
        """Return bounding-box-normalized shape of 1s as list of (dr,dc) offsets."""
        cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 1]
        if not cells:
            return None, None, None
        minr = min(r for r,c in cells)
        minc = min(c for r,c in cells)
        offsets = tuple(sorted((r-minr, c-minc) for r,c in cells))
        maxdr = max(dr for dr,dc in offsets)
        maxdc = max(dc for dr,dc in offsets)
        return offsets, maxdr+1, maxdc+1

    def get_markers(grid):
        """Return list of (r, c, color) for isolated non-zero non-1 cells."""
        markers = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                v = grid[r][c]
                if v != 0 and v != 1:
                    markers.append((r, c, v))
        return markers

    def build_output(offsets, h, w, markers):
        # Determine layout
        rows_m = [r for r,c,v in markers]
        cols_m = [c for r,c,v in markers]
        n = len(markers)
        if len(set(rows_m)) == 1:
            # horizontal: sort by col
            markers_sorted = sorted(markers, key=lambda x: x[1])
            out_h, out_w = h, w * n
            out = [[0]*out_w for _ in range(out_h)]
            for i, (r, c, color) in enumerate(markers_sorted):
                for (dr, dc) in offsets:
                    out[dr][i*w + dc] = color
        elif len(set(cols_m)) == 1:
            # vertical: sort by row
            markers_sorted = sorted(markers, key=lambda x: x[0])
            out_h, out_w = h * n, w
            out = [[0]*out_w for _ in range(out_h)]
            for i, (r, c, color) in enumerate(markers_sorted):
                for (dr, dc) in offsets:
                    out[i*h + dr][dc] = color
        else:
            return None
        return out

    def solve(grid):
        offsets, h, w = get_template(grid)
        if offsets is None:
            return None
        markers = get_markers(grid)
        if not markers:
            return None
        return build_output(offsets, h, w, markers)

    for p in pairs:
        out = solve(p['input'])
        if out is None or out != p['output']:
            return None
    return solve(test_inp)


def _solve_replace_ones_by_matching_shape(pairs, test_inp):
    """2a5f8217: replace connected components of 1s with matched non-1 component of identical shape."""
    from collections import deque
    def get_components(grid, target_colors=None):
        rows, cols = len(grid), len(grid[0])
        visited = [[False]*cols for _ in range(rows)]
        comps = []
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v == 0 or visited[r][c]:
                    continue
                if target_colors is not None and v not in target_colors:
                    continue
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==v:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                minr = min(r2 for r2,c2 in cells)
                minc = min(c2 for r2,c2 in cells)
                shape = tuple(sorted((r2-minr, c2-minc) for r2,c2 in cells))
                comps.append({'color': v, 'cells': cells, 'shape': shape})
        return comps

    def solve(grid):
        ones_comps = get_components(grid, target_colors={1})
        if not ones_comps:
            return None
        other_comps = get_components(grid, target_colors=None)
        other_comps = [c for c in other_comps if c['color'] != 1]
        shape_to_color = {}
        for comp in other_comps:
            shape_to_color[comp['shape']] = comp['color']
        result = [row[:] for row in grid]
        for ocomp in ones_comps:
            matched_color = shape_to_color.get(ocomp['shape'])
            if matched_color is None:
                return None
            for (r, c) in ocomp['cells']:
                result[r][c] = matched_color
        return result

    for p in pairs:
        out = solve(p['input'])
        if out is None or out != p['output']:
            return None
    return solve(test_inp)


def _solve_cross_diagonal_markers(pairs, test_inp):
    """0ca9ddb6: value 1 → add 7 at 4 orthogonal neighbors; value 2 → add 4 at 4 diagonal neighbors."""
    import copy
    def solve(grid):
        rows, cols = len(grid), len(grid[0])
        result = copy.deepcopy(grid)
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v == 1:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols:
                            result[nr][nc] = 7
                elif v == 2:
                    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols:
                            result[nr][nc] = 4
        return result
    for p in pairs:
        if solve(p['input']) != p['output']:
            return None
    return solve(test_inp)


def _solve_rotate_column_colors(pairs, test_inp):
    """2601afb7: find vertical color runs anchored to bottom; cyclic-right-shift colors and lengths."""
    def get_col_runs(grid):
        rows, cols = len(grid), len(grid[0])
        bg = 7  # background is 7
        runs = []
        for c in range(cols):
            col_vals = [grid[r][c] for r in range(rows)]
            non_bg = [(r, v) for r, v in enumerate(col_vals) if v != bg]
            if non_bg:
                # find contiguous run from bottom
                color = col_vals[rows-1] if col_vals[rows-1] != bg else non_bg[-1][1]
                length = 0
                for r in range(rows-1, -1, -1):
                    if col_vals[r] != bg:
                        length += 1
                    else:
                        break
                runs.append((c, color, length))
        return runs

    def solve(grid):
        rows, cols = len(grid), len(grid[0])
        bg = 7
        runs = get_col_runs(grid)
        if not runs:
            return None
        col_indices = [r[0] for r in runs]
        colors = [r[1] for r in runs]
        lengths = [r[2] for r in runs]
        n = len(runs)
        # colors: cyclic right-shift (new_colors[i] = colors[(i-1)%n])
        # lengths: cyclic left-shift (new_lengths[i] = lengths[(i+1)%n])
        new_colors = [colors[(i-1) % n] for i in range(n)]
        new_lengths = [lengths[(i+1) % n] for i in range(n)]
        result = [[bg]*cols for _ in range(rows)]
        for i, ci in enumerate(col_indices):
            clr = new_colors[i]
            lng = new_lengths[i]
            for r in range(rows-1, rows-1-lng, -1):
                if r >= 0:
                    result[r][ci] = clr
        return result

    for p in pairs:
        out = solve(p['input'])
        if out is None or out != p['output']:
            return None
    return solve(test_inp)


def _solve_diagonal_line_draw(pairs, test_inp):
    """1f876c06: for each color appearing exactly twice where |dr|==|dc|, draw diagonal."""
    import copy
    from collections import defaultdict
    def apply(grid):
        g = copy.deepcopy(grid)
        H, W = len(g), len(g[0])
        pos = defaultdict(list)
        for r in range(H):
            for c in range(W):
                v = g[r][c]
                if v != 0:
                    pos[v].append((r, c))
        for v, pts in pos.items():
            if len(pts) != 2:
                continue
            (r1, c1), (r2, c2) = pts
            dr = r2 - r1; dc = c2 - c1
            if dr == 0 or dc == 0:
                continue
            if abs(dr) != abs(dc):
                continue
            sr = 1 if dr > 0 else -1
            sc = 1 if dc > 0 else -1
            steps = abs(dr)
            for i in range(steps + 1):
                g[r1 + i * sr][c1 + i * sc] = v
        return g
    for p in pairs:
        if apply(p["input"]) != p["output"]:
            return None
    return apply(test_inp)


def _solve_color_swap_key(pairs, test_inp):
    """0becf7df: 2x2 key at top-left defines two swap pairs; apply to all non-key cells."""
    import copy
    def apply(grid):
        g = copy.deepcopy(grid)
        H, W = len(g), len(g[0])
        a, b = g[0][0], g[0][1]
        c, d = g[1][0], g[1][1]
        swap = {a: b, b: a, c: d, d: c}
        for r in range(H):
            for col in range(W):
                if r < 2 and col < 2:
                    continue
                v = g[r][col]
                if v in swap:
                    g[r][col] = swap[v]
        return g
    for p in pairs:
        if apply(p["input"]) != p["output"]:
            return None
    return apply(test_inp)


def _solve_shape_lookup_3x3(pairs, test_inp):
    """
    27a28665: 3x3 input -> 1x1 output.
    Shape (frozenset of non-zero cell positions, color-agnostic) -> integer output.
    Build map from all training pairs, look up test.
    """
    # Guard: all training inputs must be 3x3, all outputs must be 1x1
    for p in pairs:
        if len(p["input"]) != 3 or len(p["input"][0]) != 3:
            return None
        if len(p["output"]) != 1 or len(p["output"][0]) != 1:
            return None
    # Guard: test input must be 3x3
    if len(test_inp) != 3 or len(test_inp[0]) != 3:
        return None

    def _extract_shape(grid):
        s = set()
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                if v != 0:
                    s.add((r, c))
        return frozenset(s)

    shape_map = {}
    for p in pairs:
        inp_shape = _extract_shape(p["input"])
        out_val = p["output"][0][0]
        shape_map[inp_shape] = out_val

    test_shape = _extract_shape(test_inp)
    if test_shape in shape_map:
        return [[shape_map[test_shape]]]
    return None


def _solve_row_complement_noise(pairs, test_inp):
    """
    2de01db2: variable-width grid.
    Per row: find main color (most common non-0, non-7);
    complement 0 <-> main; 7-positions and noise -> main.
    """
    import collections as _col

    def _transform(row):
        counts = _col.Counter(v for v in row if v != 0 and v != 7)
        if not counts:
            return list(row)
        main = counts.most_common(1)[0][0]
        result = []
        for v in row:
            if v == 7 or (v != 0 and v != main):
                result.append(main)
            elif v == main:
                result.append(0)
            else:  # v == 0
                result.append(main)
        return result

    for p in pairs:
        for ri, row in enumerate(p["input"]):
            pred = _transform(row)
            if pred != list(p["output"][ri]):
                return None

    return [_transform(row) for row in test_inp]


def _solve_3section_stack(pairs, test_inp):
    """
    25c199f5: 5x17 -> 5x5.
    3 sections of 5x5 (cols 0-4, 6-10, 12-16), separator col5=6 col11=6, bg=7.
    Output: BG-pad prepended + [sec2 non-bg rows] + [sec1 non-bg rows] + [sec0 non-bg rows].
    """
    H, W = len(test_inp), (len(test_inp[0]) if test_inp else 0)
    if H != 5 or W != 17:
        return None
    BG = 7

    def _get_sections(grid):
        s0 = [list(grid[r][0:5])  for r in range(5)]
        s1 = [list(grid[r][6:11]) for r in range(5)]
        s2 = [list(grid[r][12:17])for r in range(5)]
        return s0, s1, s2

    def _non_bg(sec):
        return [row for row in sec if any(v != BG for v in row)]

    def _predict(grid):
        s0, s1, s2 = _get_sections(grid)
        rows = _non_bg(s2) + _non_bg(s1) + _non_bg(s0)
        bg_row = [BG] * 5
        while len(rows) < 5:
            rows.insert(0, bg_row[:])
        return rows[:5]

    for p in pairs:
        gi = p["input"]
        if len(gi) != 5 or len(gi[0]) != 17:
            return None
        if _predict(gi) != [list(r) for r in p["output"]]:
            return None

    return _predict(test_inp)



def _solve_shape_to_color_3x3_sections(pairs, test_inp):
    """
    17cae0c1: Input is 3x9 with three 3x3 sections side-by-side.
    Each section contains 5s in a specific shape.
    From training pairs, build shape->color map (frozenset of (r,c) of 5s -> color).
    Output = 3x9 where each row of section i is filled with the mapped color.
    """
    if len(test_inp) != 3 or len(test_inp[0]) != 9:
        return None

    def get_shape(grid, sec):
        pts = set()
        for r in range(3):
            for c in range(3):
                if grid[r][sec * 3 + c] == 5:
                    pts.add((r, c))
        return frozenset(pts)

    shape_map = {}
    for p in pairs:
        inp, out = p['input'], p['output']
        if len(inp) != 3 or len(inp[0]) != 9:
            return None
        if len(out) != 3 or len(out[0]) != 9:
            return None
        for sec in range(3):
            shape = get_shape(inp, sec)
            color = out[0][sec * 3]
            if shape in shape_map and shape_map[shape] != color:
                return None
            shape_map[shape] = color

    pred_row = []
    for sec in range(3):
        shape = get_shape(test_inp, sec)
        color = shape_map.get(shape)
        if color is None:
            return None
        pred_row += [color] * 3

    return [pred_row[:] for _ in range(3)]


def _solve_uniform_row_encode(pairs, test):
    """25d8a9c8: each row all-same→[5,5,5,...], mixed→[0,0,0,...]"""
    if not pairs:
        return None
    ncols = len(pairs[0]['input'][0])
    for p in pairs:
        inp, exp = p['input'], p['output']
        for r, row in enumerate(inp):
            val = 5 if len(set(row)) == 1 else 0
            if exp[r] != [val] * ncols:
                return None
    ncols_t = len(test[0])
    out = []
    for row in test:
        val = 5 if len(set(row)) == 1 else 0
        out.append([val] * ncols_t)
    return out


def _solve_count_2x2_ones_blocks(pairs, test):
    """1fad071e: count 2×2 all-1 blocks → 1×5 row with that many leading 1s"""
    if not pairs:
        return None

    def _count(grid):
        R, C = len(grid), len(grid[0])
        cnt = 0
        for r in range(R - 1):
            for c in range(C - 1):
                if grid[r][c] == 1 and grid[r][c+1] == 1 and                    grid[r+1][c] == 1 and grid[r+1][c+1] == 1:
                    cnt += 1
        return cnt

    for p in pairs:
        if len(p['output']) != 1 or len(p['output'][0]) != 5:
            return None
        exp_cnt = sum(p['output'][0])
        if _count(p['input']) != exp_cnt:
            return None
        row = p['output'][0]
        if not all(row[i] >= row[i+1] for i in range(4)):
            return None

    cnt = _count(test)
    if cnt > 5:
        return None
    return [[1]*cnt + [0]*(5-cnt)]


def _solve_trie_rows(pairs, test):
    """Trie row-mapping: 100% train coverage required."""
    if not pairs: return None
    trie = build_row_trie(pairs)
    for p in pairs:
        for inp_row, exp_row in zip(p['input'], p['output']):
            pred = trie.get(tuple(inp_row))
            if pred is None or list(pred) != exp_row:
                return None
    result = trie_predict(trie, test)
    return result if _valid_grid(result) else None


# ════════════════════════════════════════════════════════════
# NEW: Signal Processing solvers
# (Engineering: crop, filter, downsample)
# ════════════════════════════════════════════════════════════

def _solve_crop_background(pairs, test):
    """
    Signal Processing: crop grid to bounding box of non-background content.
    Covers SHRINK and CROP_SUBGRID categories.
    Tries bg=0 and bg=most-common-color.
    """
    if not pairs: return None
    for bg in [0, _most_common_color(pairs[0]['input'])]:
        try:
            ok = True
            for p in pairs:
                cropped = _crop_to_bbox(p['input'], bg)
                if cropped is None or cropped != p['output']:
                    ok = False; break
            if ok:
                result = _crop_to_bbox(test, bg)
                if result is not None and _valid_grid(result):
                    return result
        except Exception:
            pass
    return None


def _solve_select_half(pairs, test):
    """
    Signal Processing: output is one of the 4 halves (top/bottom/left/right).
    Covers SHRINK and CROP_SUBGRID categories.
    """
    if not pairs: return None

    def halves(g):
        rows, cols = len(g), len(g[0])
        return {
            'top':    [row[:] for row in g[:rows//2]],
            'bottom': [row[:] for row in g[rows//2:]],
            'left':   [row[:cols//2] for row in g],
            'right':  [row[cols//2:] for row in g],
        }

    for hname in ['top', 'bottom', 'left', 'right']:
        try:
            if all(halves(p['input'])[hname] == p['output'] for p in pairs):
                result = halves(test)[hname]
                if _valid_grid(result):
                    return result
        except Exception:
            pass
    return None


def _solve_select_quarter(pairs, test):
    """
    Signal Processing: output is one of 4 quadrants.
    Covers SHRINK and CROP_SUBGRID for even-sized inputs.
    """
    if not pairs: return None

    def quarters(g):
        rows, cols = len(g), len(g[0])
        hr, hc = rows // 2, cols // 2
        return {
            'TL': [row[:hc] for row in g[:hr]],
            'TR': [row[hc:] for row in g[:hr]],
            'BL': [row[:hc] for row in g[hr:]],
            'BR': [row[hc:] for row in g[hr:]],
        }

    for qname in ['TL', 'TR', 'BL', 'BR']:
        try:
            if all(quarters(p['input'])[qname] == p['output'] for p in pairs):
                result = quarters(test)[qname]
                if _valid_grid(result):
                    return result
        except Exception:
            pass
    return None


def _solve_remove_border(pairs, test):
    """
    Signal Processing: remove the 1-pixel outer border.
    Covers SHRINK (output = input[1:-1, 1:-1]).
    """
    if not pairs: return None

    def rm_border(g):
        rows, cols = len(g), len(g[0])
        if rows <= 2 or cols <= 2: return None
        return [row[1:-1] for row in g[1:-1]]

    for p in pairs:
        expected = rm_border(p['input'])
        if expected is None or expected != p['output']:
            return None
    result = rm_border(test)
    return result if result is not None and _valid_grid(result) else None


def _solve_add_border(pairs, test):
    """
    Signal Processing: add a 1-pixel outer border with detected color.
    Covers CONTAINS_INPUT (output is input padded by a frame).
    """
    if not pairs: return None

    def add_border(g, color):
        rows, cols = len(g), len(g[0])
        top_bot = [[color] * (cols + 2)]
        mid = [[color] + row[:] + [color] for row in g]
        return top_bot + mid + top_bot

    for bc in range(10):
        try:
            if all(add_border(p['input'], bc) == p['output'] for p in pairs):
                result = add_border(test, bc)
                if _valid_grid(result):
                    return result
        except Exception:
            pass
    return None


def _solve_stride_crop(pairs, test):
    """
    Signal Processing: extract every r_stride-th row and c_stride-th col.
    Covers SHAPE_CHANGE tasks (e.g., 5×17 → 5×5 with stride 3).
    """
    if not pairs: return None
    for r_stride in range(1, 8):
        for c_stride in range(1, 8):
            if r_stride == 1 and c_stride == 1:
                continue
            try:
                def extract(g, rs=r_stride, cs=c_stride):
                    return [row[::cs] for row in g[::rs]]
                if all(extract(p['input']) == p['output'] for p in pairs):
                    result = extract(test)
                    if _valid_grid(result):
                        return result
            except Exception:
                pass
    return None


def _solve_extract_subgrid_offset(pairs, test):
    """
    Signal Processing: output is a contiguous subgrid at fixed (r0,c0) offset.
    Covers CROP_SUBGRID / SHRINK when a single region is extracted.
    Only fires when exactly one offset is consistent across ALL pairs.
    """
    if not pairs: return None
    p0 = pairs[0]
    out_h = len(p0['output'])
    out_w = len(p0['output'][0]) if p0['output'] else 0
    inp_h = len(p0['input'])
    inp_w = len(p0['input'][0]) if p0['input'] else 0
    if out_h >= inp_h and out_w >= inp_w:
        return None  # not a crop (no shrinkage)

    valid_offsets = []
    for r0 in range(max(1, inp_h - out_h + 1)):
        for c0 in range(max(1, inp_w - out_w + 1)):
            try:
                def extract(g, r=r0, c=c0, h=out_h, w=out_w):
                    return [row[c:c+w] for row in g[r:r+h]]
                if all(extract(p['input']) == p['output'] for p in pairs):
                    valid_offsets.append((r0, c0))
            except Exception:
                pass

    if len(valid_offsets) == 1:
        r0, c0 = valid_offsets[0]
        result = [row[c0:c0+out_w] for row in test[r0:r0+out_h]]
        if _valid_grid(result):
            return result
    return None


# ════════════════════════════════════════════════════════════
# NEW: Computer Vision solvers
# (Engineering: blob analysis, component filtering)
# ════════════════════════════════════════════════════════════

def _solve_largest_component(pairs, test):
    """
    CV: keep only the single largest connected component (by cell count),
    set everything else to background.
    Covers COLOR_REDUCTION tasks.
    """
    if not pairs: return None
    for conn in [4, 8]:
        for bg in [0]:
            def _tfm(g, bg_c=bg, cn=conn):
                ccs = [(color, cells)
                       for color, cells in _find_ccs(g, cn)
                       if color != bg_c]
                if not ccs: return None
                largest = max(ccs, key=lambda x: len(x[1]))
                rows, cols = len(g), len(g[0])
                out = [[bg_c] * cols for _ in range(rows)]
                for rr, cc in largest[1]:
                    out[rr][cc] = largest[0]
                return out

            try:
                if all(_tfm(p['input']) == p['output'] for p in pairs):
                    res = _tfm(test)
                    if res is not None and _valid_grid(res):
                        return res
            except Exception:
                pass
    return None


def _solve_smallest_component(pairs, test):
    """
    CV: keep only the single smallest connected component.
    Covers COLOR_REDUCTION (find the 'signal' amid noise).
    """
    if not pairs: return None
    for conn in [4, 8]:
        for bg in [0]:
            def _tfm(g, bg_c=bg, cn=conn):
                ccs = [(color, cells)
                       for color, cells in _find_ccs(g, cn)
                       if color != bg_c]
                if not ccs: return None
                smallest = min(ccs, key=lambda x: len(x[1]))
                rows, cols = len(g), len(g[0])
                out = [[bg_c] * cols for _ in range(rows)]
                for rr, cc in smallest[1]:
                    out[rr][cc] = smallest[0]
                return out

            try:
                if all(_tfm(p['input']) == p['output'] for p in pairs):
                    res = _tfm(test)
                    if res is not None and _valid_grid(res):
                        return res
            except Exception:
                pass
    return None


def _solve_color_isolate(pairs, test):
    """
    CV: keep only cells of one specific color; replace all others with bg.
    Covers COLOR_REDUCTION (isolate one signal color).
    """
    if not pairs: return None
    for bg in [0]:
        for keep_color in range(10):
            if keep_color == bg: continue
            # keep_color must appear in all training outputs
            if not all(any(keep_color == c
                           for row in p['output'] for c in row)
                       for p in pairs):
                continue
            def _tfm(g, kc=keep_color, bc=bg):
                return [[c if c == kc else bc for c in row] for row in g]
            try:
                if all(_tfm(p['input']) == p['output'] for p in pairs):
                    if any(keep_color == c for row in test for c in row):
                        res = _tfm(test)
                        if _valid_grid(res):
                            return res
            except Exception:
                pass
    return None


def _solve_color_swap(pairs, test):
    """
    CV: swap exactly two colors (a ↔ b), all other cells unchanged.
    Covers COLOR_REDUCTION and local color-change tasks.
    """
    if not pairs: return None
    p0 = pairs[0]
    flat_in  = [c for row in p0['input']  for c in row]
    flat_out = [c for row in p0['output'] for c in row]
    if len(flat_in) != len(flat_out): return None

    swaps = {}
    for ci, co in zip(flat_in, flat_out):
        if ci == co: continue
        if ci in swaps:
            if swaps[ci] != co: return None
        else:
            swaps[ci] = co

    if len(swaps) != 2: return None
    items = list(swaps.items())
    if not (items[0][0] == items[1][1] and items[1][0] == items[0][1]):
        return None
    a, b = items[0][0], items[0][1]

    def _tfm(g):
        return [[b if c == a else (a if c == b else c) for c in row] for row in g]

    try:
        if all(_tfm(p['input']) == p['output'] for p in pairs):
            res = _tfm(test)
            if _valid_grid(res): return res
    except Exception:
        pass
    return None


def _solve_color_invert(pairs, test):
    """
    CV: invert each color value  c → (9 - c).
    Covers MAJOR_CHANGE and some COLOR_REDUCTION tasks.
    """
    if not pairs: return None
    def _tfm(g):
        return [[9 - c for c in row] for row in g]
    try:
        if all(_tfm(p['input']) == p['output'] for p in pairs):
            res = _tfm(test)
            if _valid_grid(res): return res
    except Exception:
        pass
    return None


def _solve_color_complement_bg(pairs, test):
    """
    CV: swap background color with primary foreground color.
    E.g. bg=0 fg=5 → 0→5, 5→0, others unchanged.
    """
    if not pairs: return None
    p0 = pairs[0]
    cnt_in  = _color_counts(p0['input'])
    cnt_out = _color_counts(p0['output'])
    if not cnt_in or not cnt_out: return None
    bg_in  = cnt_in.most_common(1)[0][0]
    bg_out = cnt_out.most_common(1)[0][0]
    if bg_in == bg_out: return None

    # Other most-common colors must be consistent
    def _tfm(g, a=bg_in, b=bg_out):
        return [[b if c == a else (a if c == b else c) for c in row]
                for row in g]

    try:
        if all(_tfm(p['input']) == p['output'] for p in pairs):
            res = _tfm(test)
            if _valid_grid(res): return res
    except Exception:
        pass
    return None


def _solve_propagate_nonbg(pairs, test):
    """
    CV / Diffusion: non-bg cells spread one step (BFS) into bg cells.
    Covers COLOR_EXPANSION tasks.
    """
    if not pairs: return None

    def spread_once(g, bg=0):
        rows, cols = len(g), len(g[0])
        out = [row[:] for row in g]
        for r in range(rows):
            for c in range(cols):
                if g[r][c] == bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols and g[nr][nc] != bg:
                            out[r][c] = g[nr][nc]
                            break
        return out

    for bg in [0]:
        # Determine how many spread steps are needed from pair 0
        for steps in range(1, 6):
            def _tfm(g, bg_c=bg, s=steps):
                current = g
                for _ in range(s):
                    current = spread_once(current, bg_c)
                return current
            try:
                if all(_tfm(p['input']) == p['output'] for p in pairs):
                    res = _tfm(test)
                    if _valid_grid(res): return res
            except Exception:
                pass
    return None


def _solve_erosion(pairs, test):
    """
    Morphological erosion: foreground cells that have any bg neighbor become bg.
    Nature: crystal dissolution / cell death at boundary.
    """
    if not pairs: return None

    def erode(g, bg=0):
        rows, cols = len(g), len(g[0])
        out = [row[:] for row in g]
        for r in range(rows):
            for c in range(cols):
                if g[r][c] != bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols and g[nr][nc] == bg:
                            out[r][c] = bg; break
        return out

    for steps in range(1, 4):
        def _tfm(g, s=steps):
            current = g
            for _ in range(s):
                current = erode(current)
            return current
        try:
            if all(_tfm(p['input']) == p['output'] for p in pairs):
                res = _tfm(test)
                if _valid_grid(res): return res
        except Exception:
            pass
    return None


def _solve_dilation(pairs, test):
    """
    Morphological dilation: bg cells adjacent to foreground become foreground.
    Nature: crystal growth / cell proliferation at boundary.
    """
    if not pairs: return None

    def dilate(g, bg=0):
        rows, cols = len(g), len(g[0])
        out = [row[:] for row in g]
        for r in range(rows):
            for c in range(cols):
                if g[r][c] == bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols and g[nr][nc] != bg:
                            out[r][c] = g[nr][nc]; break
        return out

    for steps in range(1, 4):
        def _tfm(g, s=steps):
            current = g
            for _ in range(s):
                current = dilate(current)
            return current
        try:
            if all(_tfm(p['input']) == p['output'] for p in pairs):
                res = _tfm(test)
                if _valid_grid(res): return res
        except Exception:
            pass
    return None


# ════════════════════════════════════════════════════════════
# NEW: Cellular Automata solvers
# (Nature: Conway Game of Life, Wolfram rules)
# ════════════════════════════════════════════════════════════

def _solve_ca_4d(pairs, test):
    """
    CA: directional 4-neighbor rule (N,S,W,E) with bv=0.
    Covers LOCAL_CHANGE tasks where orientation matters.
    """
    if not pairs: return None
    if any(len(p['input']) != len(p['output']) or
           len(p['input'][0]) != len(p['output'][0]) for p in pairs):
        return None
    rule = _build_ca_rule(pairs, _ca_nbrs_4)
    if rule is None: return None
    res = _apply_ca(test, rule, _ca_nbrs_4)
    return res if res is not None and _valid_grid(res) else None


def _solve_ca_4s(pairs, test):
    """
    CA: sorted (direction-invariant) 4-neighbor rule.
    Covers LOCAL_CHANGE tasks where rule is symmetric (majority, spread).
    """
    if not pairs: return None
    if any(len(p['input']) != len(p['output']) or
           len(p['input'][0]) != len(p['output'][0]) for p in pairs):
        return None
    rule = _build_ca_rule(pairs, _ca_nbrs_4s)
    if rule is None: return None
    res = _apply_ca(test, rule, _ca_nbrs_4s)
    return res if res is not None and _valid_grid(res) else None


def _solve_ca_8d(pairs, test):
    """
    CA: directional 8-neighbor (Moore neighborhood) rule.
    Covers LOCAL_CHANGE tasks with diagonal interactions.
    """
    if not pairs: return None
    if any(len(p['input']) != len(p['output']) or
           len(p['input'][0]) != len(p['output'][0]) for p in pairs):
        return None
    rule = _build_ca_rule(pairs, _ca_nbrs_8)
    if rule is None: return None
    res = _apply_ca(test, rule, _ca_nbrs_8)
    return res if res is not None and _valid_grid(res) else None


def _solve_ca_8s(pairs, test):
    """
    CA: sorted (direction-invariant) 8-neighbor rule.
    """
    if not pairs: return None
    if any(len(p['input']) != len(p['output']) or
           len(p['input'][0]) != len(p['output'][0]) for p in pairs):
        return None
    rule = _build_ca_rule(pairs, _ca_nbrs_8s)
    if rule is None: return None
    res = _apply_ca(test, rule, _ca_nbrs_8s)
    return res if res is not None and _valid_grid(res) else None


def _solve_ca_center_only(pairs, test):
    """
    CA degenerate: rule depends only on center value (= bijective color map).
    Captures simple per-cell recoloring not caught by backtrack_color_map.
    """
    if not pairs: return None
    if any(len(p['input']) != len(p['output']) or
           len(p['input'][0]) != len(p['output'][0]) for p in pairs):
        return None
    rule = {}
    for p in pairs:
        inp, out = p['input'], p['output']
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                key = inp[r][c]
                val = out[r][c]
                if key in rule and rule[key] != val: return None
                rule[key] = val
    if not rule: return None
    rows, cols = len(test), len(test[0])
    res = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if test[r][c] not in rule: return None
            res[r][c] = rule[test[r][c]]
    return res if _valid_grid(res) else None


def _solve_ca_4d_alt_bv(pairs, test):
    """
    CA 4-neighbor directional with bv = most_common_color of input.
    Covers tasks where border context matters.
    """
    if not pairs: return None
    if any(len(p['input']) != len(p['output']) or
           len(p['input'][0]) != len(p['output'][0]) for p in pairs):
        return None
    bv = _most_common_color(pairs[0]['input'])
    if bv == 0: return None   # duplicate of _solve_ca_4d

    def nbrs_bv(g, r, c, bv_=bv):
        return _ca_nbrs_4(g, r, c, bv=bv_)

    rule = _build_ca_rule(pairs, nbrs_bv)
    if rule is None: return None
    res = _apply_ca(test, rule, nbrs_bv)
    return res if res is not None and _valid_grid(res) else None


# ════════════════════════════════════════════════════════════
# NEW: Crystal Growth / Morphogenesis solvers
# (Nature: symmetry completion, Turing patterns)
# ════════════════════════════════════════════════════════════

def _solve_sym_h(pairs, test):
    """
    Crystal growth: complete horizontal (left-right) mirror symmetry.
    Covers SYMH_COMPLETE tasks.
    """
    if not pairs: return None
    for bg in [0, _most_common_color(pairs[0]['input'])]:
        try:
            if all(_sym_h_complete(p['input'], bg) == p['output']
                   for p in pairs):
                result = _sym_h_complete(test, bg)
                if result != test and _valid_grid(result):
                    return result
        except Exception:
            pass
    return None


def _solve_sym_v(pairs, test):
    """
    Crystal growth: complete vertical (top-bottom) mirror symmetry.
    Covers SYMV_COMPLETE tasks.
    """
    if not pairs: return None
    for bg in [0, _most_common_color(pairs[0]['input'])]:
        try:
            if all(_sym_v_complete(p['input'], bg) == p['output']
                   for p in pairs):
                result = _sym_v_complete(test, bg)
                if result != test and _valid_grid(result):
                    return result
        except Exception:
            pass
    return None


def _solve_sym_rot180(pairs, test):
    """Crystal growth: complete 180° rotational symmetry."""
    if not pairs: return None
    for bg in [0, _most_common_color(pairs[0]['input'])]:
        try:
            if all(_sym_rot180_complete(p['input'], bg) == p['output']
                   for p in pairs):
                result = _sym_rot180_complete(test, bg)
                if result != test and _valid_grid(result):
                    return result
        except Exception:
            pass
    return None


def _solve_sym_diagonal(pairs, test):
    """Crystal growth: complete diagonal (transpose) symmetry."""
    if not pairs: return None
    if any(len(p['input']) != len(p['input'][0]) for p in pairs):
        return None  # only square grids
    for bg in [0, _most_common_color(pairs[0]['input'])]:
        try:
            if all(_sym_diag_complete(p['input'], bg) == p['output']
                   for p in pairs):
                result = _sym_diag_complete(test, bg)
                if result != test and _valid_grid(result):
                    return result
        except Exception:
            pass
    return None


# ════════════════════════════════════════════════════════════
# NEW: Pattern / Structure solvers
# (Information theory, signal processing)
# ════════════════════════════════════════════════════════════

def _solve_stamp_upscale(pairs, test):
    """
    Information theory: each input color maps to a specific k×k block.
    Covers UPSCALE tasks where the expansion uses a color-keyed lookup.
    """
    if not pairs: return None
    p0 = pairs[0]
    ir, ic = len(p0['input']), len(p0['input'][0]) if p0['input'] else 0
    or_, oc = len(p0['output']), len(p0['output'][0]) if p0['output'] else 0
    if ir == 0 or ic == 0 or or_ == 0 or oc == 0: return None
    if or_ % ir != 0 or oc % ic != 0: return None
    kr, kc = or_ // ir, oc // ic
    if not (2 <= kr <= 5 and 2 <= kc <= 5): return None

    color_map = {}
    for r in range(ir):
        for c in range(ic):
            color = p0['input'][r][c]
            block = [p0['output'][r*kr+dr][c*kc:c*kc+kc] for dr in range(kr)]
            if color in color_map:
                if color_map[color] != block: return None
            else:
                color_map[color] = block

    for p in pairs[1:]:
        ir2, ic2 = len(p['input']), len(p['input'][0]) if p['input'] else 0
        or2, oc2 = len(p['output']), len(p['output'][0]) if p['output'] else 0
        if or2 != ir2*kr or oc2 != ic2*kc: return None
        for r in range(ir2):
            for c in range(ic2):
                color = p['input'][r][c]
                block = [p['output'][r*kr+dr][c*kc:c*kc+kc] for dr in range(kr)]
                if color not in color_map or color_map[color] != block:
                    return None

    tr, tc = len(test), len(test[0]) if test else 0
    if tr == 0 or tc == 0: return None
    result = [[0] * (tc * kc) for _ in range(tr * kr)]
    for r in range(tr):
        for c in range(tc):
            color = test[r][c]
            if color not in color_map: return None
            block = color_map[color]
            for dr in range(kr):
                for dc in range(kc):
                    result[r*kr+dr][c*kc+dc] = block[dr][dc]
    return result if _valid_grid(result) else None


def _solve_color_parity(pairs, test):
    """
    Pattern: output color depends on (r + c) % 2 or r % 2 or c % 2.
    Covers checkerboard / stripe tasks.
    """
    if not pairs: return None
    if any(len(p['input']) != len(p['output']) or
           len(p['input'][0]) != len(p['output'][0]) for p in pairs):
        return None

    for parity_fn_name in ['rc', 'r', 'c']:
        if parity_fn_name == 'rc':
            pfn = lambda r, c: (r + c) % 2
        elif parity_fn_name == 'r':
            pfn = lambda r, c: r % 2
        else:
            pfn = lambda r, c: c % 2

        # Build rule: (input_color, parity) → output_color
        rule = {}
        ok = True
        for p in pairs:
            inp, out = p['input'], p['output']
            rows, cols = len(inp), len(inp[0])
            for r in range(rows):
                for c in range(cols):
                    key = (inp[r][c], pfn(r, c))
                    val = out[r][c]
                    if key in rule and rule[key] != val:
                        ok = False; break
                    rule[key] = val
                if not ok: break
            if not ok: break
        if not ok or not rule: continue

        rows, cols = len(test), len(test[0])
        res = [[0]*cols for _ in range(rows)]
        valid = True
        for r in range(rows):
            for c in range(cols):
                key = (test[r][c], pfn(r, c))
                if key not in rule:
                    valid = False; break
                res[r][c] = rule[key]
            if not valid: break
        if valid and _valid_grid(res):
            return res
    return None


def _solve_row_col_repeat(pairs, test):
    """
    Pattern: output = input with each row/col repeated k times.
    """
    if not pairs: return None
    p0 = pairs[0]
    ir, ic = len(p0['input']), len(p0['input'][0])
    or_, oc = len(p0['output']), len(p0['output'][0])

    # Row repeat
    if or_ % ir == 0 and oc == ic:
        k = or_ // ir
        if k >= 2:
            def rr_tfm(g, k_=k):
                out = []
                for row in g:
                    for _ in range(k_):
                        out.append(row[:])
                return out
            try:
                if all(rr_tfm(p['input']) == p['output'] for p in pairs):
                    res = rr_tfm(test)
                    if _valid_grid(res): return res
            except Exception:
                pass

    # Column repeat
    if ir == or_ and oc % ic == 0:
        k = oc // ic
        if k >= 2:
            def cr_tfm(g, k_=k):
                return [[c for c in row for _ in range(k_)] for row in g]
            try:
                if all(cr_tfm(p['input']) == p['output'] for p in pairs):
                    res = cr_tfm(test)
                    if _valid_grid(res): return res
            except Exception:
                pass
    return None


# ════════════════════════════════════════════════════════════
# NEW: Physics / Graph Theory solvers
# (object motion, field equations)
# ════════════════════════════════════════════════════════════

def _solve_object_translate(pairs, test):
    """
    Graph Theory: detect uniform translation vector applied to ALL non-bg cells.
    Covers LOCAL_CHANGE / MOVEMENT tasks.
    """
    if not pairs: return None

    for bg in [0]:
        # Find translation from first pair
        p0 = pairs[0]
        inp_cells = {(r, c): p0['input'][r][c]
                     for r in range(len(p0['input']))
                     for c in range(len(p0['input'][0]))
                     if p0['input'][r][c] != bg}
        out_cells = {(r, c): p0['output'][r][c]
                     for r in range(len(p0['output']))
                     for c in range(len(p0['output'][0]))
                     if p0['output'][r][c] != bg}
        if not inp_cells or not out_cells: continue
        if len(inp_cells) != len(out_cells): continue

        # Try all candidate (dr, dc) by comparing a single seed
        ir0, ic0 = next(iter(inp_cells))
        found_dr_dc = None
        rows_in, cols_in = len(p0['input']), len(p0['input'][0])

        for (or0, oc0), ov in out_cells.items():
            if ov != inp_cells[(ir0, ic0)]: continue
            dr, dc = or0 - ir0, oc0 - ic0
            # Validate on all non-bg input cells
            if all(
                (r+dr, c+dc) in out_cells and
                out_cells[(r+dr, c+dc)] == v
                for (r, c), v in inp_cells.items()
            ):
                found_dr_dc = (dr, dc)
                break

        if found_dr_dc is None: continue
        dr, dc = found_dr_dc

        # Verify all pairs
        ok = True
        for p in pairs[1:]:
            ic2 = {(r, c): p['input'][r][c]
                   for r in range(len(p['input']))
                   for c in range(len(p['input'][0]))
                   if p['input'][r][c] != bg}
            oc2 = {(r, c): p['output'][r][c]
                   for r in range(len(p['output']))
                   for c in range(len(p['output'][0]))
                   if p['output'][r][c] != bg}
            if len(ic2) != len(oc2):
                ok = False; break
            if not all(
                (r+dr, c+dc) in oc2 and oc2[(r+dr, c+dc)] == v
                for (r, c), v in ic2.items()
            ):
                ok = False; break
        if not ok: continue

        # Apply to test
        t_rows, t_cols = len(test), len(test[0])
        result = [[bg] * t_cols for _ in range(t_rows)]
        for r in range(t_rows):
            for c in range(t_cols):
                if test[r][c] != bg:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < t_rows and 0 <= nc < t_cols:
                        result[nr][nc] = test[r][c]
        if _valid_grid(result):
            return result
    return None


def _solve_gravity_nonzero(pairs, test):
    """
    Physics: gravity for non-zero values toward 4 directions, bg = most common.
    Extension of existing gravity to non-zero background.
    """
    if not pairs: return None
    bg = _most_common_color(pairs[0]['input'])
    if bg == 0: return None  # duplicate of existing gravity

    def grav(g, direction, bg_c):
        g2 = [[bg_c if v == bg_c else v for v in row] for row in g]
        rows, cols = len(g2), len(g2[0])
        if direction == 'down':
            for c in range(cols):
                col_v = [g2[r][c] for r in range(rows) if g2[r][c] != bg_c]
                emp = rows - len(col_v)
                for r in range(rows):
                    g2[r][c] = bg_c if r < emp else col_v[r - emp]
        elif direction == 'up':
            for c in range(cols):
                col_v = [g2[r][c] for r in range(rows) if g2[r][c] != bg_c]
                for r in range(rows):
                    g2[r][c] = col_v[r] if r < len(col_v) else bg_c
        elif direction == 'left':
            for r in range(rows):
                rv = [c for c in g2[r] if c != bg_c]
                g2[r] = rv + [bg_c] * (cols - len(rv))
        elif direction == 'right':
            for r in range(rows):
                rv = [c for c in g2[r] if c != bg_c]
                g2[r] = [bg_c] * (cols - len(rv)) + rv
        return g2

    for d in ['down', 'up', 'left', 'right']:
        try:
            if all(grav(p['input'], d, bg) == p['output'] for p in pairs):
                res = grav(test, d, bg)
                if _valid_grid(res): return res
        except Exception:
            pass
    return None


def _solve_majority_fill(pairs, test):
    """
    Control Theory: replace each bg cell with the majority non-bg neighbor color.
    Covers COLOR_REDUCTION / hole-filling tasks.
    """
    if not pairs: return None
    if any(len(p['input']) != len(p['output']) or
           len(p['input'][0]) != len(p['output'][0]) for p in pairs):
        return None

    def maj_fill(g, bg=0):
        rows, cols = len(g), len(g[0])
        out = [row[:] for row in g]
        for r in range(rows):
            for c in range(cols):
                if g[r][c] == bg:
                    nbr_colors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols and g[nr][nc] != bg:
                            nbr_colors.append(g[nr][nc])
                    if nbr_colors:
                        cnt = Counter(nbr_colors)
                        out[r][c] = cnt.most_common(1)[0][0]
        return out

    for bg in [0]:
        for steps in range(1, 5):
            def _tfm(g, bg_c=bg, s=steps):
                current = g
                for _ in range(s):
                    current = maj_fill(current, bg_c)
                return current
            try:
                if all(_tfm(p['input']) == p['output'] for p in pairs):
                    res = _tfm(test)
                    if _valid_grid(res): return res
            except Exception:
                pass
    return None


# ════════════════════════════════════════════════════════════
# NEW: Chain / Compound solvers
# (Program synthesis: depth-2 compositions)
# ════════════════════════════════════════════════════════════

def _solve_chain_geo_remap(pairs, test):
    """
    Program synthesis: geometric transform → color remap.
    Covers tasks needing rotation + recoloring.
    """
    if not pairs: return None
    geo_names = ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'flip_d']
    for geo in geo_names:
        try:
            new_pairs = []
            for p in pairs:
                t = _apply_transform(p['input'], geo)
                if t is None: break
                new_pairs.append({'input': t, 'output': p['output']})
            else:
                t_test = _apply_transform(test, geo)
                remap = backtrack_color_map(new_pairs, t_test)
                if remap is not None and _valid_grid(remap):
                    return remap
        except Exception:
            pass
    return None


def _solve_chain_remap_geo(pairs, test):
    """
    Program synthesis: color remap → geometric transform.
    """
    if not pairs: return None
    geo_names = ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'flip_d']

    # Find remap first (if any)
    remap_map = {}
    for p in pairs:
        flat_in  = [c for row in p['input']  for c in row]
        for ci in flat_in:
            if ci not in remap_map:
                remap_map[ci] = None

    # Try color remap then geo
    remap = backtrack_color_map(pairs, test)
    if remap is None: return None

    remap_pairs = [
        {'input': [[remap_map.get(c, c) if remap_map else c for c in row]
                   for row in p['input']],
         'output': p['output']}
        for p in pairs
    ]
    # Build remap pairs properly
    remap_pairs2 = []
    mapping = {}
    for p in pairs:
        flat_in  = [c for row in p['input']  for c in row]
        flat_out = [c for row in p['output'] for c in row]
        if len(flat_in) != len(flat_out): return None
        for ci, co in zip(flat_in, flat_out):
            if ci in mapping and mapping[ci] != co: return None
            mapping[ci] = co

    for p in pairs:
        recolored_in = [[mapping.get(c, c) for c in row] for row in p['input']]
        remap_pairs2.append({'input': recolored_in, 'output': p['output']})

    recolored_test = [[mapping.get(c, c) for c in row] for row in test]

    for geo in geo_names:
        try:
            if all(_apply_transform(rp['input'], geo) == rp['output']
                   for rp in remap_pairs2):
                res = _apply_transform(recolored_test, geo)
                if res is not None and _valid_grid(res):
                    return res
        except Exception:
            pass
    return None


def _solve_scale_then_remap(pairs, test):
    """
    Program synthesis: scale up → color remap.
    Covers upscale + recoloring.
    """
    if not pairs: return None
    p0 = pairs[0]
    ir, ic = len(p0['input']), len(p0['input'][0])
    or_, oc = len(p0['output']), len(p0['output'][0])
    if or_ <= ir or oc <= ic: return None

    for factor in range(2, 6):
        if ir * factor != or_ or ic * factor != oc: continue
        scaled_pairs = []
        for p in pairs:
            scaled_inp = scale_up(p['input'], factor)
            scaled_pairs.append({'input': scaled_inp, 'output': p['output']})
        scaled_test = scale_up(test, factor)
        remap = backtrack_color_map(scaled_pairs, scaled_test)
        if remap is not None and _valid_grid(remap):
            return remap
    return None


# ── Ordered solver pipeline (precision-first ordering) ───────

# ════════════════════════════════════════════════════════════
# NEW TARGETED SOLVERS  (diagnostically derived from failing 123)
# ════════════════════════════════════════════════════════════

# ── Separator detection helpers ───────────────────────────────
def _find_full_sep_cols(grid):
    """Column indices where every cell has the same value (uniform color)."""
    rows, cols = len(grid), len(grid[0])
    return [c for c in range(cols)
            if len(set(grid[r][c] for r in range(rows))) == 1]


def _find_full_sep_rows(grid):
    """Row indices where every cell has the same value (uniform color)."""
    return [r for r, row in enumerate(grid) if len(set(row)) == 1]


def _split_cols(grid, sep_cols):
    """Split grid into sub-grids at column separators; skip separator cols."""
    cols = len(grid[0])
    seps = set(sep_cols)
    parts, start = [], 0
    for c in range(cols + 1):
        if c == cols or c in seps:
            if c > start:
                parts.append([row[start:c] for row in grid])
            start = c + 1
    return parts


def _split_rows(grid, sep_rows):
    """Split grid into sub-grids at row separators; skip separator rows."""
    rows = len(grid)
    seps = set(sep_rows)
    parts, start = [], 0
    for r in range(rows + 1):
        if r == rows or r in seps:
            if r > start:
                parts.append(grid[start:r])
            start = r + 1
    return parts


def _grid_bg(grid):
    """Most common cell value = background."""
    all_c = [v for row in grid for v in row]
    return Counter(all_c).most_common(1)[0][0]


# ── 1. Fractal self-stamp ─────────────────────────────────────
def _solve_fractal_stamp_self(pairs, test):
    """
    N×M input → (N*N)×(M*M) output.
    For each non-bg cell (r,c), paste the entire input into block
    (r*N:(r+1)*N, c*M:(c+1)*M).  Bg cells → all-bg block.
    FIX: always use bg=0 (sparse grids have non-zero majority but 0 is true bg).
    Confirmed on task 007bbfb7.
    """
    if not pairs:
        return None

    def _apply(grid):
        nr, nc = len(grid), len(grid[0])
        bg = 0  # ARC convention: 0 is background for fractal-stamp tasks
        out = [[bg] * (nc * nc) for _ in range(nr * nr)]
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] != bg:
                    for dr in range(nr):
                        for dc in range(nc):
                            out[r * nr + dr][c * nc + dc] = grid[dr][dc]
        return out

    for p in pairs:
        nr, nc = len(p['input']), len(p['input'][0])
        if len(p['output']) != nr * nr or len(p['output'][0]) != nc * nc:
            return None
        if _apply(p['input']) != p['output']:
            return None
    return _apply(test)


# ── 2. Stack content rows from separator-column panels (bottom-up) ──
def _solve_overlay_sep_col(pairs, test):
    """
    Split input by separator columns into panels.
    For each panel keep only rows that contain ≥1 non-bg cell.
    Stack panels vertically with panel0 at the bottom:
      output (top→bottom) = content(panelN-1) … content(panel1) content(panel0).
    Handles task 25c199f5.
    """
    if not pairs:
        return None

    def _apply(grid):
        bg = _grid_bg(grid)
        seps = _find_full_sep_cols(grid)
        if not seps:
            return None
        parts = _split_cols(grid, seps)
        if len(parts) < 2:
            return None
        widths = {len(p[0]) for p in parts}
        if len(widths) != 1:
            return None
        # Extract content rows from each panel (rows with ≥1 non-bg cell)
        content = []
        for part in parts:
            rows_with_content = [row for row in part if any(v != bg for v in row)]
            content.append(rows_with_content)
        # Stack: panel0 at bottom → reversed order when reading top-to-bottom
        stacked = []
        for panel_rows in reversed(content):
            stacked.extend(panel_rows)
        return stacked if stacked else None

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None or pred != p['output']:
            return None
    return _apply(test)


# ── 3. Overlay sub-grids split by separator rows ─────────────
def _solve_overlay_sep_row(pairs, test):
    """
    Separator rows divide input into equal-height parts.
    Overlay all parts: first non-bg value wins at each cell.
    """
    if not pairs:
        return None

    def _apply(grid):
        bg = _grid_bg(grid)
        seps = _find_full_sep_rows(grid)
        if not seps:
            return None
        parts = _split_rows(grid, seps)
        if len(parts) < 2:
            return None
        shapes = {(len(p), len(p[0])) for p in parts}
        if len(shapes) != 1:
            return None
        h, w = shapes.pop()
        merged = [[bg] * w for _ in range(h)]
        for part in parts:
            for r in range(h):
                for c in range(w):
                    if merged[r][c] == bg and part[r][c] != bg:
                        merged[r][c] = part[r][c]
        return merged

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None or pred != p['output']:
            return None
    return _apply(test)


# ── 4. AND both halves at single separator column ─────────────
def _solve_and_halves_sep_col(pairs, test):
    """
    Exactly one separator column divides input into two equal halves.
    Output: cells that are non-bg in BOTH halves → active color; rest bg.
    FIX: brute-force bg and fill-color detection so that grids where
    0 is NOT the majority (task 0520fde7) are handled correctly.
    """
    if not pairs:
        return None
    bg_color = None
    active_color = None

    def _find_sep(grid):
        seps = _find_full_sep_cols(grid)
        return seps[0] if len(seps) == 1 else None

    def _apply(grid, bg, fc):
        s = _find_sep(grid)
        if s is None:
            return None
        left  = [row[:s]      for row in grid]
        right = [row[s + 1:]  for row in grid]
        if not left or not right or len(left[0]) != len(right[0]):
            return None
        rows, hw = len(grid), len(left[0])
        out = [[bg] * hw for _ in range(rows)]
        for r in range(rows):
            for c in range(hw):
                if left[r][c] != bg and right[r][c] != bg:
                    out[r][c] = fc
        return out

    for p in pairs:
        s = _find_sep(p['input'])
        if s is None:
            return None
        left  = [row[:s]     for row in p['input']]
        right = [row[s + 1:] for row in p['input']]
        if not left or not right or len(left[0]) != len(right[0]):
            return None
        rows, hw = len(p['input']), len(left[0])
        if len(p['output']) != rows or len(p['output'][0]) != hw:
            return None

        inp_colors = list({v for row in p['input'] for v in row})
        out_colors = list({v for row in p['output'] for v in row})

        if bg_color is None:
            # Brute-force: try every (bg_cand, fc_cand) combination
            found = False
            for bg_cand in inp_colors:
                for fc_cand in out_colors:
                    pred = _apply(p['input'], bg_cand, fc_cand)
                    if pred is not None and pred == p['output']:
                        bg_color  = bg_cand
                        active_color = fc_cand
                        found = True
                        break
                if found:
                    break
            if not found:
                return None
        else:
            pred = _apply(p['input'], bg_color, active_color)
            if pred is None or pred != p['output']:
                return None

    if bg_color is None or active_color is None:
        return None
    return _apply(test, bg_color, active_color)


# ── 5. Connect same-color cells within same row ───────────────
def _solve_connect_row(pairs, test):
    """
    For each row: groups of same-color non-bg cells separated by only bg
    → fill the gap (including endpoints) with that color.
    Handles tasks 22eb0ac0, and the row-direction of 070dd51e.
    """
    if not pairs:
        return None

    def _apply(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        g = [row[:] for row in grid]
        changed = False
        for r in range(rows):
            by_color = defaultdict(list)
            for c in range(cols):
                if g[r][c] != bg:
                    by_color[g[r][c]].append(c)
            for color, positions in by_color.items():
                if len(positions) < 2:
                    continue
                mn, mx = min(positions), max(positions)
                # Only if all between them is bg or same color
                if all(g[r][c] in (bg, color) for c in range(mn, mx + 1)):
                    for c in range(mn, mx + 1):
                        if g[r][c] != color:
                            g[r][c] = color
                            changed = True
        return g if changed else None

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None or pred != p['output']:
            return None
    return _apply(test)


# ── 6. Connect same-color cells within same column ────────────
def _solve_connect_col(pairs, test):
    """
    For each column: same-color non-bg cells separated by only bg
    → fill the gap with that color.
    Handles 17b80ad2, 253bf280, and col-direction of 070dd51e.
    """
    if not pairs:
        return None

    def _apply(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        g = [row[:] for row in grid]
        changed = False
        for c in range(cols):
            by_color = defaultdict(list)
            for r in range(rows):
                if g[r][c] != bg:
                    by_color[g[r][c]].append(r)
            for color, positions in by_color.items():
                if len(positions) < 2:
                    continue
                mn, mx = min(positions), max(positions)
                if all(g[r][c] in (bg, color) for r in range(mn, mx + 1)):
                    for r in range(mn, mx + 1):
                        if g[r][c] != color:
                            g[r][c] = color
                            changed = True
        return g if changed else None

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None:
            if p['input'] == p['output']:
                continue  # identity pair
            return None
        if pred != p['output']:
            return None
    return _apply(test)


# ── 7. Connect same-color cells in both rows AND columns ──────
def _solve_connect_row_col(pairs, test):
    """
    Run row-connect and column-connect INDEPENDENTLY on the original grid,
    then merge with column-result priority:
      col non-bg > row non-bg > original.
    FIX: sequential overwrite corrupted crossing cells; now uses original grid
    for both passes and merges cleanly.  Handles 070dd51e.
    """
    if not pairs:
        return None

    def _row_connect(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        g = [row[:] for row in grid]
        for r in range(rows):
            by_color = defaultdict(list)
            for c in range(cols):
                if grid[r][c] != bg:
                    by_color[grid[r][c]].append(c)
            for color, positions in by_color.items():
                if len(positions) < 2:
                    continue
                mn, mx = min(positions), max(positions)
                if all(grid[r][c] in (bg, color) for c in range(mn, mx + 1)):
                    for c in range(mn, mx + 1):
                        g[r][c] = color
        return g

    def _col_connect(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        g = [row[:] for row in grid]
        for c in range(cols):
            by_color = defaultdict(list)
            for r in range(rows):
                if grid[r][c] != bg:
                    by_color[grid[r][c]].append(r)
            for color, positions in by_color.items():
                if len(positions) < 2:
                    continue
                mn, mx = min(positions), max(positions)
                if all(grid[r][c] in (bg, color) for r in range(mn, mx + 1)):
                    for r in range(mn, mx + 1):
                        g[r][c] = color
        return g

    def _apply(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        row_res = _row_connect(grid)
        col_res = _col_connect(grid)
        result = [[grid[r][c] for c in range(cols)] for r in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if col_res[r][c] != bg:
                    result[r][c] = col_res[r][c]
                elif row_res[r][c] != bg:
                    result[r][c] = row_res[r][c]
        return result

    for p in pairs:
        pred = _apply(p['input'])
        if pred == p['input'] and p['input'] == p['output']:
            continue  # identity pair
        if pred != p['output']:
            return None

    result = _apply(test)
    return result if result != test else None


# ── 8. Fill gap between exactly-2 same-color cells per column ─
def _solve_fill_pair_col(pairs, test):
    """
    For each color appearing exactly 2 times in the same column:
    fill all cells strictly between the two occurrences with a fill color
    learned from training.  Handles 253bf280.
    """
    if not pairs:
        return None
    fill_map = {}   # dot_color → fill_color

    for p in pairs:
        bg = _grid_bg(p['input'])
        rows, cols = len(p['input']), len(p['input'][0])
        # Build expected output from learned fill_map
        expected = [row[:] for row in p['input']]
        learned_new = {}
        for c in range(cols):
            by_color = defaultdict(list)
            for r in range(rows):
                v = p['input'][r][c]
                if v != bg:
                    by_color[v].append(r)
            for color, rs in by_color.items():
                if len(rs) != 2:
                    continue
                r1, r2 = min(rs), max(rs)
                if r2 - r1 <= 1:
                    continue
                # Determine fill color from output
                out_vals = {p['output'][r][c] for r in range(r1 + 1, r2)
                            if p['output'][r][c] != bg}
                if len(out_vals) != 1:
                    return None
                fc = list(out_vals)[0]
                if color in fill_map and fill_map[color] != fc:
                    return None
                if color not in fill_map:
                    learned_new[color] = fc
                for r in range(r1 + 1, r2):
                    expected[r][c] = fc
        fill_map.update(learned_new)
        if expected != p['output']:
            return None

    if not fill_map:
        return None
    # Apply to test
    bg = _grid_bg(test)
    rows, cols = len(test), len(test[0])
    result = [row[:] for row in test]
    touched = False
    for c in range(cols):
        by_color = defaultdict(list)
        for r in range(rows):
            v = test[r][c]
            if v != bg:
                by_color[v].append(r)
        for color, rs in by_color.items():
            if len(rs) != 2 or color not in fill_map:
                continue
            r1, r2 = min(rs), max(rs)
            for r in range(r1 + 1, r2):
                result[r][c] = fill_map[color]
                touched = True
    return result if touched else None


# ── 9. Fill enclosed bg with a new learned color ─────────────
def _solve_enclosed_new_color(pairs, test):
    """
    Input has 2 colors (bg + fg wall).
    Bg cells unreachable from any border → filled with a new color
    that doesn't appear in the input (learned from training).
    Handles 31adaf00.
    """
    if not pairs:
        return None
    fill_color = None

    def _enclosed_mask(grid, bg):
        rows, cols = len(grid), len(grid[0])
        visited = set()
        stack = []
        for r in range(rows):
            for c in (0, cols - 1):
                if grid[r][c] == bg and (r, c) not in visited:
                    visited.add((r, c))
                    stack.append((r, c))
        for c in range(cols):
            for r in (0, rows - 1):
                if grid[r][c] == bg and (r, c) not in visited:
                    visited.add((r, c))
                    stack.append((r, c))
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols \
                        and (nr, nc) not in visited \
                        and grid[nr][nc] == bg:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        return [(r, c) for r in range(rows) for c in range(cols)
                if grid[r][c] == bg and (r, c) not in visited]

    for p in pairs:
        inp_colors = {v for row in p['input'] for v in row}
        if len(inp_colors) > 2:
            return None
        bg = _grid_bg(p['input'])
        enclosed = _enclosed_mask(p['input'], bg)
        if not enclosed:
            return None
        fc_vals = {p['output'][r][c] for r, c in enclosed}
        if len(fc_vals) != 1:
            return None
        fc = list(fc_vals)[0]
        if fill_color is None:
            fill_color = fc
        elif fill_color != fc:
            return None
        expected = [row[:] for row in p['input']]
        for r, c in enclosed:
            expected[r][c] = fill_color
        if expected != p['output']:
            return None

    if fill_color is None:
        return None
    bg = _grid_bg(test)
    enclosed = _enclosed_mask(test, bg)
    if not enclosed:
        return None
    result = [row[:] for row in test]
    for r, c in enclosed:
        result[r][c] = fill_color
    return result


# ── 10. Extract the single non-empty sub-grid (2D separator grid) ─
def _solve_content_subgrid_sep(pairs, test):
    """
    Grid divided by full-row AND full-col separators into sub-cells.
    Exactly one sub-cell has non-bg content → output = that sub-cell.
    Handles 2dc579da.
    """
    if not pairs:
        return None

    def _get_content_cell(grid):
        bg = _grid_bg(grid)
        sep_rows = _find_full_sep_rows(grid)
        sep_cols = _find_full_sep_cols(grid)
        if not sep_rows and not sep_cols:
            return None
        row_parts = _split_rows(grid, sep_rows)
        if not row_parts:
            return None
        content = None
        for rp in row_parts:
            col_parts = _split_cols(rp, sep_cols)
            for cp in col_parts:
                if any(cp[r][c] != bg
                       for r in range(len(cp)) for c in range(len(cp[0]))):
                    if content is not None:
                        return None  # multiple non-empty sub-cells
                    content = cp
        return content

    for p in pairs:
        cc = _get_content_cell(p['input'])
        if cc is None or cc != p['output']:
            return None
    return _get_content_cell(test)


# ── 11. Overlay same-sized scattered objects ──────────────────
def _solve_overlay_objects(pairs, test):
    """
    Find all connected-component bounding boxes in input (scattered objects).
    If all bounding boxes are the same size → overlay them (first non-bg wins).
    Handles 25e02866, 20818e16.
    """
    if not pairs:
        return None

    def _extract_bboxes(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        boxes = []
        for sr in range(rows):
            for sc in range(cols):
                if not visited[sr][sc] and grid[sr][sc] != bg:
                    stack = [(sr, sc)]
                    visited[sr][sc] = True
                    comp = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols \
                                    and not visited[nr][nc] \
                                    and grid[nr][nc] != bg:
                                visited[nr][nc] = True
                                stack.append((nr, nc))
                                comp.append((nr, nc))
                    r0 = min(r for r, c in comp)
                    r1 = max(r for r, c in comp)
                    c0 = min(c for r, c in comp)
                    c1 = max(c for r, c in comp)
                    boxes.append([grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)])
        return boxes, bg

    def _apply(grid):
        boxes, bg = _extract_bboxes(grid)
        if len(boxes) < 2:
            return None
        shapes = {(len(b), len(b[0])) for b in boxes}
        if len(shapes) != 1:
            return None
        h, w = shapes.pop()
        merged = [[bg] * w for _ in range(h)]
        for box in boxes:
            for r in range(h):
                for c in range(w):
                    if merged[r][c] == bg and box[r][c] != bg:
                        merged[r][c] = box[r][c]
        return merged

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None or pred != p['output']:
            return None
    return _apply(test)


# ── 12. Output unique separator-line colors (sorted) ─────────
def _solve_sep_line_colors(pairs, test):
    """
    Collect all unique non-bg separator colors from BOTH row and column
    separator lines; sort ascending; output as 1-row array [[c1, c2, …]].
    Also handles column-vector output [[c1],[c2],…].
    FIX: previous version picked only row OR col seps with wrong ordering.
    Handles 22425bda.
    """
    if not pairs:
        return None

    def _all_sep_colors(grid):
        bg = _grid_bg(grid)
        colors = set()
        for r in _find_full_sep_rows(grid):
            v = grid[r][0]
            if v != bg:
                colors.add(v)
        for c in _find_full_sep_cols(grid):
            v = grid[0][c]
            if v != bg:
                colors.add(v)
        return sorted(colors)

    # Determine output format from first pair
    p0_colors = _all_sep_colors(pairs[0]['input'])
    if not p0_colors:
        return None
    p0_out = pairs[0]['output']
    if p0_out == [p0_colors]:
        fmt = 'row'
    elif p0_out == [[c] for c in p0_colors]:
        fmt = 'col'
    else:
        return None

    for p in pairs:
        colors = _all_sep_colors(p['input'])
        if not colors:
            return None
        expected = [colors] if fmt == 'row' else [[c] for c in colors]
        if p['output'] != expected:
            return None

    colors = _all_sep_colors(test)
    if not colors:
        return None
    return [colors] if fmt == 'row' else [[c] for c in colors]


# ── 13. Fill entire row when endpoints share same color ───────
def _solve_fill_row_endpoints(pairs, test):
    """
    For each row, for each color present, fill the span from the leftmost to
    rightmost occurrence of that color. Processes each color independently,
    so rows with multiple colors are handled correctly.
    Handles 22eb0ac0, 22168020 and similar.
    """
    if not pairs:
        return None

    def _apply(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        g = [row[:] for row in grid]
        changed = False
        for r in range(rows):
            by_color = {}
            for c in range(cols):
                v = g[r][c]
                if v != bg:
                    by_color.setdefault(v, []).append(c)
            for color, positions in by_color.items():
                if len(positions) < 2:
                    continue
                lo, hi = min(positions), max(positions)
                for c in range(lo, hi + 1):
                    if g[r][c] != color:
                        g[r][c] = color
                        changed = True
        return g if changed else None

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None:
            if [list(r) for r in p['input']] == [list(r) for r in p['output']]:
                continue  # identity pair — no fills needed
            return None
        if pred != [list(r) for r in p['output']]:
            return None
    return _apply(test)


# ── 14. Color each full column top-to-bottom from dot dots ────
def _solve_color_columns_full(pairs, test):
    """
    Scattered single-pixel colored dots → color their entire column
    from topmost to bottommost dot of that color.
    Handles 17b80ad2.
    """
    if not pairs:
        return None

    def _apply(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        g = [row[:] for row in grid]
        changed = False
        for c in range(cols):
            by_color = defaultdict(list)
            for r in range(rows):
                if g[r][c] != bg:
                    by_color[g[r][c]].append(r)
            for color, rs in by_color.items():
                if len(rs) < 2:
                    continue
                r0, r1 = min(rs), max(rs)
                for r in range(r0, r1 + 1):
                    if g[r][c] != color:
                        g[r][c] = color
                        changed = True
        return g if changed else None

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None or pred != p['output']:
            return None
    return _apply(test)


# ── 15. Crop to tightest non-bg bounding box (generalized) ────
def _solve_crop_to_content(pairs, test):
    """
    Output = tightest bounding box of all non-bg cells.
    More aggressive than crop_background: works even when bg ≠ 0.
    """
    if not pairs:
        return None

    def _crop(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        non_bg = [(r, c) for r in range(rows) for c in range(cols)
                  if grid[r][c] != bg]
        if not non_bg:
            return None
        r0 = min(r for r, c in non_bg)
        r1 = max(r for r, c in non_bg)
        c0 = min(c for r, c in non_bg)
        c1 = max(c for r, c in non_bg)
        if r0 == 0 and r1 == rows - 1 and c0 == 0 and c1 == cols - 1:
            return None  # already full size — not a crop task
        return [grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)]

    for p in pairs:
        pred = _crop(p['input'])
        if pred is None or pred != p['output']:
            return None
    return _crop(test)


# ── 16. Pattern continuation across separator sections ───────
def _solve_pattern_sequence_sep(pairs, test):
    """
    Grid divided by full-row separators into equal sections.
    Each section contains exactly one row with a contiguous colored block.
    Detect mode:
      A) lengths grow arithmetically (start fixed)
      B) start shifts arithmetically (length fixed)
    Extrapolate to produce the next section.
    FIX: replaces broken centroid-shift logic with explicit 1D-sequence analysis.
    Handles 351d6448.
    """
    if not pairs:
        return None

    def _get_sections_patterns(grid):
        seps = _find_full_sep_rows(grid)
        if not seps:
            return None
        parts = _split_rows(grid, seps)
        shapes = {(len(p), len(p[0])) for p in parts}
        if len(shapes) != 1:
            return None
        h, w = shapes.pop()
        patterns = []
        for sec in parts:
            bg = _grid_bg(sec)
            nonbg_rows = []
            for ri, row in enumerate(sec):
                cells = [(c, v) for c, v in enumerate(row) if v != bg]
                if cells:
                    nonbg_rows.append((ri, cells))
            if len(nonbg_rows) != 1:
                return None
            ri, cells = nonbg_rows[0]
            cs  = sorted(c for c, v in cells)
            vs  = list({v for c, v in cells})
            # must be contiguous single-color block
            if cs != list(range(cs[0], cs[-1] + 1)) or len(vs) != 1:
                return None
            patterns.append({'row': ri, 'start': cs[0],
                             'length': len(cs), 'color': vs[0]})
        return parts, patterns, h, w

    def _predict_next(patterns, h, w, bg):
        if len(patterns) < 2:
            return None
        if len({p['row'] for p in patterns}) != 1:
            return None
        if len({p['color'] for p in patterns}) != 1:
            return None
        starts  = [p['start']  for p in patterns]
        lengths = [p['length'] for p in patterns]
        color   = patterns[0]['color']
        pat_row = patterns[0]['row']
        len_deltas   = [lengths[i+1] - lengths[i]  for i in range(len(lengths)-1)]
        start_deltas = [starts[i+1]  - starts[i]   for i in range(len(starts)-1)]
        mode = None
        if (len(set(len_deltas)) == 1 and len_deltas[0] != 0
                and len(set(starts)) == 1):
            mode       = 'length_grow'
            len_delta  = len_deltas[0]
            next_start  = starts[-1]
            next_length = lengths[-1] + len_delta
        elif (len(set(start_deltas)) == 1 and start_deltas[0] != 0
                and len(set(lengths)) == 1):
            mode        = 'start_shift'
            start_delta = start_deltas[0]
            next_start  = starts[-1] + start_delta
            next_length = lengths[-1]
        else:
            return None
        if next_start < 0 or next_start + next_length > w or next_length <= 0:
            return None
        sec = [[bg] * w for _ in range(h)]
        for c in range(next_start, next_start + next_length):
            sec[pat_row][c] = color
        return sec

    for p in pairs:
        info = _get_sections_patterns(p['input'])
        if info is None:
            return None
        sections, patterns, h, w = info
        bg = _grid_bg(sections[-1])
        predicted = _predict_next(patterns, h, w, bg)
        if predicted is None or predicted != p['output']:
            return None

    info = _get_sections_patterns(test)
    if info is None:
        return None
    sections, patterns, h, w = info
    bg = _grid_bg(sections[-1])
    return _predict_next(patterns, h, w, bg)

# ── NEW-A: Voronoi/territory column segment fill ─────────────
def _solve_segment_fill_col(pairs, test):
    """
    For each column, sort non-bg dot positions by row into a sequence.
    Segment i covers rows (r_{i-1}+1) … r_i (inclusive); first segment starts at row 0.
    Fill each segment with the color of its bottom-boundary dot.
    After the last dot, fill remaining rows with the last dot's color.
    FIX for 17b80ad2: different-color dots in same column get territory-fill,
    not same-color-span fill.
    """
    if not pairs:
        return None

    def _apply(grid):
        bg = _grid_bg(grid)
        rows, cols = len(grid), len(grid[0])
        g = [row[:] for row in grid]
        changed = False
        for c in range(cols):
            dots = sorted([(r, grid[r][c]) for r in range(rows)
                           if grid[r][c] != bg])
            if not dots:
                continue
            prev_r = -1
            for dot_r, dot_color in dots:
                for r in range(prev_r + 1, dot_r + 1):
                    if g[r][c] != dot_color:
                        g[r][c] = dot_color
                        changed = True
                prev_r = dot_r
            # Trailing segment: last dot's color fills to grid bottom
            last_r, last_color = dots[-1]
            for r in range(last_r + 1, rows):
                if g[r][c] != last_color:
                    g[r][c] = last_color
                    changed = True
        return g if changed else None

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None or pred != p['output']:
            return None
    return _apply(test)


# ── NEW-B: Fill 2×2 all-background blocks with a learned color ─
def _solve_fill_2x2_blocks(pairs, test):
    """
    Find every 2×2 sub-block in which all 4 cells are background;
    fill all such cells with a fill color learned from training.
    FIX for 31adaf00: replaces incorrect flood-fill-from-border algorithm
    with exact 2×2-block detection.
    """
    if not pairs:
        return None
    fill_color = None

    def _2x2_bg_cells(grid, bg):
        rows, cols = len(grid), len(grid[0])
        cells = set()
        for r in range(rows - 1):
            for c in range(cols - 1):
                if (grid[r][c] == bg and grid[r][c+1] == bg and
                        grid[r+1][c] == bg and grid[r+1][c+1] == bg):
                    cells |= {(r, c), (r, c+1), (r+1, c), (r+1, c+1)}
        return cells

    for p in pairs:
        bg = _grid_bg(p['input'])
        cells = _2x2_bg_cells(p['input'], bg)
        if not cells:
            return None
        fc_vals = {p['output'][r][c] for r, c in cells}
        if len(fc_vals) != 1:
            return None
        fc = list(fc_vals)[0]
        if fill_color is None:
            fill_color = fc
        elif fill_color != fc:
            return None
        expected = [row[:] for row in p['input']]
        for r, c in cells:
            expected[r][c] = fill_color
        if expected != p['output']:
            return None

    if fill_color is None:
        return None
    bg = _grid_bg(test)
    cells = _2x2_bg_cells(test, bg)
    if not cells:
        return None
    result = [row[:] for row in test]
    for r, c in cells:
        result[r][c] = fill_color
    return result


# ── NEW-C: Connect endpoint pairs in rows AND columns ─────────
def _solve_connect_endpoints(pairs, test):
    """
    For each color appearing exactly 2 times in the same row or column:
    fill all cells strictly between the two occurrences with a fill color
    learned from training.  Supports both row pairs and column pairs.
    Identity-pair safe.
    FIX for 253bf280: extends _solve_fill_pair_col with row support.
    """
    if not pairs:
        return None
    fill_map = {}  # dot_color → fill_color

    for p in pairs:
        bg = _grid_bg(p['input'])
        rows, cols = len(p['input']), len(p['input'][0])
        expected = [row[:] for row in p['input']]
        learned_new = {}

        def _infer_fill(dot_color, pos1, pos2, axis):
            """Return fill color from output cells between pos1 and pos2."""
            if axis == 'col':
                r1, r2 = min(pos1, pos2), max(pos1, pos2)
                c_fixed = dot_color   # reuse var name — see call site
                # caller passes column index via closure — handled below
                return None  # handled inline
            return None

        # Column pairs
        for c in range(cols):
            by_color = defaultdict(list)
            for r in range(rows):
                v = p['input'][r][c]
                if v != bg:
                    by_color[v].append(r)
            for color, rs in by_color.items():
                if len(rs) != 2:
                    continue
                r1, r2 = min(rs), max(rs)
                if r2 - r1 <= 1:
                    continue
                out_vals = {p['output'][r][c] for r in range(r1 + 1, r2)
                            if p['output'][r][c] != bg}
                if len(out_vals) != 1:
                    return None
                fc = list(out_vals)[0]
                if color in fill_map and fill_map[color] != fc:
                    return None
                if color in learned_new and learned_new[color] != fc:
                    return None
                learned_new.setdefault(color, fc)
                for r in range(r1 + 1, r2):
                    expected[r][c] = fc

        # Row pairs
        for r in range(rows):
            by_color = defaultdict(list)
            for c in range(cols):
                v = p['input'][r][c]
                if v != bg:
                    by_color[v].append(c)
            for color, cs in by_color.items():
                if len(cs) != 2:
                    continue
                c1, c2 = min(cs), max(cs)
                if c2 - c1 <= 1:
                    continue
                out_vals = {p['output'][r][c] for c in range(c1 + 1, c2)
                            if p['output'][r][c] != bg}
                if len(out_vals) != 1:
                    return None
                fc = list(out_vals)[0]
                if color in fill_map and fill_map[color] != fc:
                    return None
                if color in learned_new and learned_new[color] != fc:
                    return None
                learned_new.setdefault(color, fc)
                for c in range(c1 + 1, c2):
                    expected[r][c] = fc

        fill_map.update(learned_new)
        if expected == p['input'] and p['input'] == p['output']:
            continue  # identity pair
        if expected != p['output']:
            return None

    if not fill_map:
        return None

    bg = _grid_bg(test)
    rows, cols = len(test), len(test[0])
    result = [row[:] for row in test]
    touched = False

    for c in range(cols):
        by_color = defaultdict(list)
        for r in range(rows):
            v = test[r][c]
            if v != bg:
                by_color[v].append(r)
        for color, rs in by_color.items():
            if len(rs) != 2 or color not in fill_map:
                continue
            r1, r2 = min(rs), max(rs)
            for r in range(r1 + 1, r2):
                result[r][c] = fill_map[color]
                touched = True

    for r in range(rows):
        by_color = defaultdict(list)
        for c in range(cols):
            v = test[r][c]
            if v != bg:
                by_color[v].append(c)
        for color, cs in by_color.items():
            if len(cs) != 2 or color not in fill_map:
                continue
            c1, c2 = min(cs), max(cs)
            for c in range(c1 + 1, c2):
                result[r][c] = fill_map[color]
                touched = True

    return result if touched else None




# ---------------------------------------------------------------------------
# BATCH-2 NEW SOLVERS: diagonal_stripe, fractal_stamp_complement,
#   stamp_max_freq, stamp_uniform_axis, mirror_mosaic_2x2
# ---------------------------------------------------------------------------

def _solve_diagonal_stripe(pairs, test):
    def _get_seq(grid):
        cells = [(r, c, grid[r][c])
                 for r in range(len(grid))
                 for c in range(len(grid[0]))
                 if grid[r][c] != 0]
        if not cells: return None, None
        for k in range(2, 8):
            mapping = {}
            ok = True
            for r, c, v in cells:
                key = (r + c) % k
                if key in mapping and mapping[key] != v:
                    ok = False; break
                mapping[key] = v
            if ok and len(mapping) == k:
                seq = [mapping[i] for i in range(k)]
                return seq, k
        return None, None

    for p in pairs:
        seq, k = _get_seq(p['input'])
        if seq is None: return None
        rows, cols = len(p['input']), len(p['input'][0])
        expected = [[seq[(r+c)%k] for c in range(cols)] for r in range(rows)]
        if expected != p['output']: return None

    seq, k = _get_seq(test)
    if seq is None: return None
    rows, cols = len(test), len(test[0])
    return [[seq[(r+c)%k] for c in range(cols)] for r in range(rows)]


def _solve_fractal_stamp_complement(pairs, test):
    def _apply(grid):
        nr, nc = len(grid), len(grid[0])
        bg = 0
        colors = set(v for row in grid for v in row if v != bg)
        if len(colors) != 1: return None
        color = next(iter(colors))
        comp = [[color if v == bg else bg for v in row] for row in grid]
        out = [[bg] * (nc * nc) for _ in range(nr * nr)]
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == color:
                    for dr in range(nr):
                        for dc in range(nc):
                            out[r*nr+dr][c*nc+dc] = comp[dr][dc]
        return out

    for p in pairs:
        nr, nc = len(p['input']), len(p['input'][0])
        if len(p['output']) != nr*nr or len(p['output'][0]) != nc*nc:
            return None
        if _apply(p['input']) != p['output']: return None
    return _apply(test)


def _solve_stamp_max_freq(pairs, test):
    def _apply(grid):
        nr, nc = len(grid), len(grid[0])
        bg = 0
        freq = Counter(v for row in grid for v in row if v != bg)
        if not freq: return None
        max_color = freq.most_common(1)[0][0]
        positions = [(r, c) for r in range(nr) for c in range(nc)
                     if grid[r][c] == max_color]
        out = [[bg]*(nc*nc) for _ in range(nr*nr)]
        for r, c in positions:
            for dr in range(nr):
                for dc in range(nc):
                    out[r*nr+dr][c*nc+dc] = grid[dr][dc]
        return out

    for p in pairs:
        nr, nc = len(p['input']), len(p['input'][0])
        if len(p['output']) != nr*nr or len(p['output'][0]) != nc*nc:
            return None
        if _apply(p['input']) != p['output']: return None
    return _apply(test)


def _solve_stamp_uniform_axis(pairs, test):
    def _apply(grid):
        nr, nc = len(grid), len(grid[0])
        bg = 0
        for r in range(nr):
            row = [grid[r][c] for c in range(nc)]
            if len(set(row)) == 1 and row[0] != bg:
                out = [[bg]*(nc*nc) for _ in range(nr*nr)]
                for bc in range(nr):
                    for dr in range(nr):
                        for dc in range(nc):
                            out[r*nr+dr][bc*nc+dc] = grid[dr][dc]
                return out
        for c in range(nc):
            col = [grid[r][c] for r in range(nr)]
            if len(set(col)) == 1 and col[0] != bg:
                out = [[bg]*(nc*nc) for _ in range(nr*nr)]
                for br in range(nr):
                    for dr in range(nr):
                        for dc in range(nc):
                            out[br*nr+dr][c*nc+dc] = grid[dr][dc]
                return out
        return None

    for p in pairs:
        nr, nc = len(p['input']), len(p['input'][0])
        if len(p['output']) != nr*nr or len(p['output'][0]) != nc*nc:
            return None
        if _apply(p['input']) != p['output']: return None
    return _apply(test)


def _solve_mirror_mosaic_2x2(pairs, test):
    def _apply(grid):
        nr, nc = len(grid), len(grid[0])
        orig   = [row[:] for row in grid]
        flipud = grid[::-1]
        fliplr = [row[::-1] for row in grid]
        rot180 = [row[::-1] for row in grid[::-1]]
        out = [[0]*(nc*2) for _ in range(nr*2)]
        for r in range(nr):
            for c in range(nc):
                out[r][c]       = rot180[r][c]
                out[r][c+nc]    = flipud[r][c]
                out[r+nr][c]    = fliplr[r][c]
                out[r+nr][c+nc] = orig[r][c]
        return out

    for p in pairs:
        nr, nc = len(p['input']), len(p['input'][0])
        if len(p['output']) != nr*2 or len(p['output'][0]) != nc*2:
            return None
        if _apply(p['input']) != p['output']: return None
    return _apply(test)




def _solve_cross_draw_markers(pairs, test):
    """140c817e: draw full cross (row+col) per sparse marker; center→2, 4 diag neighbors→3."""
    from collections import Counter
    def _apply(grid):
        R, C = len(grid), len(grid[0])
        flat = [v for row in grid for v in row]
        bg = Counter(flat).most_common(1)[0][0]
        marker_cells = {}
        for r in range(R):
            for c in range(C):
                v = grid[r][c]
                if v != bg:
                    marker_cells[(r, c)] = v
        if not marker_cells:
            return None
        out = [row[:] for row in grid]
        for (mr, mc), color in marker_cells.items():
            for c in range(C):
                if out[mr][c] == bg:
                    out[mr][c] = color
            for r in range(R):
                if out[r][mc] == bg:
                    out[r][mc] = color
            out[mr][mc] = 2
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = mr+dr, mc+dc
                if 0 <= nr < R and 0 <= nc < C and out[nr][nc] == bg:
                    out[nr][nc] = 3
        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None
    return _apply(test)


def _solve_voronoi_row_frame(pairs, test):
    """0f63c0b9: row-Voronoi frame: assign each row to nearest seed row; seed rows + row 0 + row R-1 fully painted; others only col-0 and col-C-1 painted."""
    from collections import Counter
    def _apply(grid):
        R, C = len(grid), len(grid[0])
        flat = [v for row in grid for v in row]
        bg = Counter(flat).most_common(1)[0][0]
        seeds = {}
        for r in range(R):
            for c in range(C):
                if grid[r][c] != bg:
                    seeds[r] = grid[r][c]
                    break
        if not seeds:
            return None
        seed_rows = sorted(seeds.keys())
        def nearest_seed(r):
            best, best_d = None, 10**9
            for sr in seed_rows:
                d = abs(r - sr)
                if d < best_d or (d == best_d and sr < best):
                    best_d = d; best = sr
            return best
        zone = {r: nearest_seed(r) for r in range(R)}
        out = [[bg]*C for _ in range(R)]
        for r in range(R):
            color = seeds[zone[r]]
            is_boundary = (r == 0 or r == R-1 or r in seeds)
            if is_boundary:
                for c in range(C):
                    out[r][c] = color
            else:
                out[r][0] = color
                out[r][C-1] = color
        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None
    return _apply(test)


def _solve_recolor_8_by_key_pattern(pairs, test):
    """009d5c81: find 3×3 key pattern of 1s → lookup color; replace 8s with color, 1s with 0."""
    KEY_MAP = {
        frozenset({(0,0),(0,1),(0,2),(1,0),(1,2),(2,1)}): 7,
        frozenset({(0,0),(0,2),(1,1),(2,0),(2,1),(2,2)}): 3,
        frozenset({(0,1),(1,0),(1,1),(1,2),(2,1)}): 2,
    }
    def extract_key(grid):
        ones = [(r,c) for r,row in enumerate(grid) for c,v in enumerate(row) if v == 1]
        if not ones:
            return None
        r0 = min(r for r,c in ones); c0 = min(c for r,c in ones)
        return frozenset((r-r0, c-c0) for r,c in ones)

    def apply_color(grid, color):
        return [[color if v==8 else (0 if v==1 else v) for v in row] for row in grid]

    for p in pairs:
        key = extract_key(p['input'])
        if key not in KEY_MAP:
            return None
        if apply_color(p['input'], KEY_MAP[key]) != p['output']:
            return None
    key = extract_key(test)
    if key not in KEY_MAP:
        return None
    return apply_color(test, KEY_MAP[key])


def _solve_grid_and_intersection(pairs, test):
    """0520fde7: split at separator column (from training); AND both halves → 2; else 0."""
    def find_sep(grid):
        R, C = len(grid), len(grid[0])
        for c in range(C):
            col_vals = {grid[r][c] for r in range(R)}
            if len(col_vals) == 1 and 0 not in col_vals:
                return c, next(iter(col_vals))
        return None, None

    sep_col, sep_color = find_sep(pairs[0]['input'])
    if sep_col is None:
        return None
    for p in pairs[1:]:
        sc, sv = find_sep(p['input'])
        if sc != sep_col or sv != sep_color:
            return None

    def solve_at(grid, sc):
        R, C = len(grid), len(grid[0])
        if sc >= C:
            return None
        left  = [[grid[r][c] for c in range(sc)] for r in range(R)]
        right = [[grid[r][c] for c in range(sc+1, C)] for r in range(R)]
        if not left or not right or len(left[0]) != len(right[0]):
            return None
        W = len(left[0])
        return [[2 if left[r][c] != 0 and right[r][c] != 0 else 0
                 for c in range(W)] for r in range(R)]

    for p in pairs:
        if solve_at(p['input'], sep_col) != p['output']:
            return None
    return solve_at(test, sep_col)


def _solve_grid_nor(pairs, test):
    """1b2d62fb: split at separator column (derived from training); NOR both halves → 8; else 0."""
    def find_sep(grid):
        R, C = len(grid), len(grid[0])
        for c in range(C):
            col_vals = {grid[r][c] for r in range(R)}
            if len(col_vals) == 1 and 0 not in col_vals:
                return c, next(iter(col_vals))
        return None, None

    # Derive separator column and color from training (must be consistent)
    sep_col, sep_color = find_sep(pairs[0]['input'])
    if sep_col is None:
        return None
    for p in pairs[1:]:
        sc, sv = find_sep(p['input'])
        if sc != sep_col or sv != sep_color:
            return None

    def solve_at(grid, sc):
        R, C = len(grid), len(grid[0])
        if sc >= C:
            return None
        left  = [[grid[r][c] for c in range(sc)] for r in range(R)]
        right = [[grid[r][c] for c in range(sc+1, C)] for r in range(R)]
        if not left or not right or len(left[0]) != len(right[0]):
            return None
        W = len(left[0])
        return [[8 if left[r][c] == 0 and right[r][c] == 0 else 0
                 for c in range(W)] for r in range(R)]

    for p in pairs:
        if solve_at(p['input'], sep_col) != p['output']:
            return None
    return solve_at(test, sep_col)



def _solve_recolor_small_components(pairs, test_inp):
    """12eac192 + 1e5d6875: small connected components (size<=2) → color 3; large ones unchanged."""
    from collections import deque
    def solve(grid):
        rows, cols = len(grid), len(grid[0])
        visited = [[False]*cols for _ in range(rows)]
        result = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0 or visited[r][c]:
                    continue
                color = grid[r][c]
                # BFS to find component
                q = deque([(r, c)])
                visited[r][c] = True
                component = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            component.append((nr, nc))
                if len(component) <= 2:
                    for (cr2, cc2) in component:
                        result[cr2][cc2] = 3
        return result
    for p in pairs:
        if solve(p['input']) != p['output']:
            return None
    return solve(test_inp)


def _solve_most_frequent_below_sep(pairs, test_inp):
    """27a77e38: find separator row of all-5s; count values above; place most-frequent below at last_row,mid_col."""
    def solve(grid):
        rows, cols = len(grid), len(grid[0])
        sep_row = None
        for r in range(rows):
            if all(grid[r][c] == 5 for c in range(cols)):
                sep_row = r
                break
        if sep_row is None:
            return None
        # count values above separator (exclude 0 and 5)
        from collections import Counter
        cnt = Counter()
        for r in range(sep_row):
            for c in range(cols):
                v = grid[r][c]
                if v != 0:
                    cnt[v] += 1
        if not cnt:
            return None
        most_freq = cnt.most_common(1)[0][0]
        result = [row[:] for row in grid]
        last_row = rows - 1
        mid_col = cols // 2
        result[last_row][mid_col] = most_freq
        return result
    for p in pairs:
        out = solve(p['input'])
        if out is None or out != p['output']:
            return None
    return solve(test_inp)


def _solve_template_color_markers(pairs, test_inp):
    """12997ef3: shape of 1s as template; isolated non-zero non-1 cells are color markers.
       If markers in same row → stack horizontally; if same col → stack vertically."""
    def get_template(grid):
        """Return bounding-box-normalized shape of 1s as list of (dr,dc) offsets."""
        cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 1]
        if not cells:
            return None, None, None
        minr = min(r for r,c in cells)
        minc = min(c for r,c in cells)
        offsets = tuple(sorted((r-minr, c-minc) for r,c in cells))
        maxdr = max(dr for dr,dc in offsets)
        maxdc = max(dc for dr,dc in offsets)
        return offsets, maxdr+1, maxdc+1

    def get_markers(grid):
        """Return list of (r, c, color) for isolated non-zero non-1 cells."""
        markers = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                v = grid[r][c]
                if v != 0 and v != 1:
                    markers.append((r, c, v))
        return markers

    def build_output(offsets, h, w, markers):
        # Determine layout
        rows_m = [r for r,c,v in markers]
        cols_m = [c for r,c,v in markers]
        n = len(markers)
        if len(set(rows_m)) == 1:
            # horizontal: sort by col
            markers_sorted = sorted(markers, key=lambda x: x[1])
            out_h, out_w = h, w * n
            out = [[0]*out_w for _ in range(out_h)]
            for i, (r, c, color) in enumerate(markers_sorted):
                for (dr, dc) in offsets:
                    out[dr][i*w + dc] = color
        elif len(set(cols_m)) == 1:
            # vertical: sort by row
            markers_sorted = sorted(markers, key=lambda x: x[0])
            out_h, out_w = h * n, w
            out = [[0]*out_w for _ in range(out_h)]
            for i, (r, c, color) in enumerate(markers_sorted):
                for (dr, dc) in offsets:
                    out[i*h + dr][dc] = color
        else:
            return None
        return out

    def solve(grid):
        offsets, h, w = get_template(grid)
        if offsets is None:
            return None
        markers = get_markers(grid)
        if not markers:
            return None
        return build_output(offsets, h, w, markers)

    for p in pairs:
        out = solve(p['input'])
        if out is None or out != p['output']:
            return None
    return solve(test_inp)


def _solve_replace_ones_by_matching_shape(pairs, test_inp):
    """2a5f8217: replace connected components of 1s with matched non-1 component of identical shape."""
    from collections import deque
    def get_components(grid, target_colors=None):
        rows, cols = len(grid), len(grid[0])
        visited = [[False]*cols for _ in range(rows)]
        comps = []
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v == 0 or visited[r][c]:
                    continue
                if target_colors is not None and v not in target_colors:
                    continue
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==v:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                minr = min(r2 for r2,c2 in cells)
                minc = min(c2 for r2,c2 in cells)
                shape = tuple(sorted((r2-minr, c2-minc) for r2,c2 in cells))
                comps.append({'color': v, 'cells': cells, 'shape': shape})
        return comps

    def solve(grid):
        ones_comps = get_components(grid, target_colors={1})
        if not ones_comps:
            return None
        other_comps = get_components(grid, target_colors=None)
        other_comps = [c for c in other_comps if c['color'] != 1]
        shape_to_color = {}
        for comp in other_comps:
            shape_to_color[comp['shape']] = comp['color']
        result = [row[:] for row in grid]
        for ocomp in ones_comps:
            matched_color = shape_to_color.get(ocomp['shape'])
            if matched_color is None:
                return None
            for (r, c) in ocomp['cells']:
                result[r][c] = matched_color
        return result

    for p in pairs:
        out = solve(p['input'])
        if out is None or out != p['output']:
            return None
    return solve(test_inp)


def _solve_cross_diagonal_markers(pairs, test_inp):
    """0ca9ddb6: value 1 → add 7 at 4 orthogonal neighbors; value 2 → add 4 at 4 diagonal neighbors."""
    import copy
    def solve(grid):
        rows, cols = len(grid), len(grid[0])
        result = copy.deepcopy(grid)
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v == 1:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols:
                            result[nr][nc] = 7
                elif v == 2:
                    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols:
                            result[nr][nc] = 4
        return result
    for p in pairs:
        if solve(p['input']) != p['output']:
            return None
    return solve(test_inp)


def _solve_rotate_column_colors(pairs, test_inp):
    """2601afb7: find vertical color runs anchored to bottom; cyclic-right-shift colors and lengths."""
    def get_col_runs(grid):
        rows, cols = len(grid), len(grid[0])
        bg = 7  # background is 7
        runs = []
        for c in range(cols):
            col_vals = [grid[r][c] for r in range(rows)]
            non_bg = [(r, v) for r, v in enumerate(col_vals) if v != bg]
            if non_bg:
                # find contiguous run from bottom
                color = col_vals[rows-1] if col_vals[rows-1] != bg else non_bg[-1][1]
                length = 0
                for r in range(rows-1, -1, -1):
                    if col_vals[r] != bg:
                        length += 1
                    else:
                        break
                runs.append((c, color, length))
        return runs

    def solve(grid):
        rows, cols = len(grid), len(grid[0])
        bg = 7
        runs = get_col_runs(grid)
        if not runs:
            return None
        col_indices = [r[0] for r in runs]
        colors = [r[1] for r in runs]
        lengths = [r[2] for r in runs]
        n = len(runs)
        # cyclic right-shift: new_colors[i] = colors[(i-1)%n], new_lengths[i] = lengths[(i-1)%n]
        new_colors = [colors[(i-1) % n] for i in range(n)]
        new_lengths = [lengths[(i-1) % n] for i in range(n)]
        result = [[bg]*cols for _ in range(rows)]
        for i, ci in enumerate(col_indices):
            clr = new_colors[i]
            lng = new_lengths[i]
            for r in range(rows-1, rows-1-lng, -1):
                if r >= 0:
                    result[r][ci] = clr
        return result

    for p in pairs:
        out = solve(p['input'])
        if out is None or out != p['output']:
            return None
    return solve(test_inp)


# ── TILE 3×3 WITH DIAGONAL MARKER (310f3251) ─────────────────────────────────
def _solve_tile_3x3_with_marker(pairs, test_inp):
    """
    For each non-bg cell at (r,c), place marker M at ((r-1)%H,(c-1)%W) if bg.
    Tile the modified grid 3×3. Output size = (3H, 3W).
    """
    from collections import Counter

    def bg_of(grid):
        flat = [v for row in grid for v in row]
        return Counter(flat).most_common(1)[0][0]

    def apply_marker(grid, bg, M):
        H, W = len(grid), len(grid[0])
        g = [row[:] for row in grid]
        for r in range(H):
            for c in range(W):
                if grid[r][c] != bg:
                    mr, mc = (r - 1) % H, (c - 1) % W
                    if g[mr][mc] == bg:
                        g[mr][mc] = M
        return g

    def tile_3(grid):
        return [row * 3 for row in grid] * 3

    if not pairs:
        return None
    # Determine M
    M = None
    for pair in pairs:
        inp, out = pair['input'], pair['output']
        H, W = len(inp), len(inp[0])
        if len(out) != 3 * H or len(out[0]) != 3 * W:
            return None
        in_vals = {v for row in inp for v in row}
        out_vals = {v for row in out for v in row}
        extra = out_vals - in_vals
        if len(extra) != 1:
            return None
        m = next(iter(extra))
        if M is None:
            M = m
        elif M != m:
            return None
    if M is None:
        return None
    # Verify all training pairs
    for pair in pairs:
        inp, out = pair['input'], pair['output']
        bg = bg_of(inp)
        if tile_3(apply_marker(inp, bg, M)) != out:
            return None
    # Apply to test
    bg = bg_of(test_inp)
    return tile_3(apply_marker(test_inp, bg, M))


# ── TILE 3×3 ALTERNATING HORIZONTAL FLIP (00576224) ──────────────────────────
def _solve_tile_3x3_alt_hflip(pairs, test_inp):
    """
    Tile input 3 copies wide per row. Alternate each H-row block between
    original (even block) and horizontally reversed (odd block).
    Output size = (3H, 3W).
    """
    if not pairs:
        return None

    def make_output(grid):
        H = len(grid)
        out = []
        for block in range(3):
            for r in range(H):
                row = grid[r]
                if block % 2 == 1:
                    row = row[::-1]
                out.append(row * 3)
        return out

    for pair in pairs:
        inp, out = pair['input'], pair['output']
        H, W = len(inp), len(inp[0])
        if len(out) != 3 * H or len(out[0]) != 3 * W:
            return None
        if make_output(inp) != out:
            return None
    return make_output(test_inp)


# ── EXTEND PERIOD TO 9 ROWS WITH COLOR REMAP (017c7c7b) ──────────────────────
def _solve_extend_period_to_9rows(pairs, test_inp):
    """
    Input rows repeat with period P; replace color A->B;
    output = 9 rows cycling through the remapped period.
    Guards: all outputs have 9 rows; exactly one non-bg color remapped.
    """
    if not pairs:
        return None

    def find_period(rows):
        n = len(rows)
        for p in range(1, n + 1):
            if all(rows[i] == rows[i % p] for i in range(n)):
                return p
        return n

    # Guard: all outputs have 9 rows
    for pair in pairs:
        if len(pair['output']) != 9:
            return None
    # test input should NOT already have 9 rows (no-op guard)
    if len(test_inp) == 9:
        return None

    # Determine remap from all pairs
    remap = None
    for pair in pairs:
        inp, out = pair['input'], pair['output']
        if len(inp[0]) != len(out[0]):
            return None
        in_vals = {v for row in inp for v in row}
        out_vals = {v for row in out for v in row}
        in_nonbg = in_vals - {0}
        out_nonbg = out_vals - {0}
        if len(in_nonbg) != 1 or len(out_nonbg) != 1:
            return None
        A, B = next(iter(in_nonbg)), next(iter(out_nonbg))
        r = (A, B)
        if remap is None:
            remap = r
        elif remap != r:
            return None
    if remap is None:
        return None
    A, B = remap

    def apply_remap(row):
        return [B if v == A else v for v in row]

    # Verify all training pairs
    for pair in pairs:
        inp, out = pair['input'], pair['output']
        period = find_period(inp)
        expected = [apply_remap(inp[i % period]) for i in range(9)]
        if expected != out:
            return None

    # Apply to test
    period = find_period(test_inp)
    return [apply_remap(test_inp[i % period]) for i in range(9)]


def _solve_nearest_wall_color(pairs, test_inp):
    """
    2204b7a8: Input has 'wall' rows or columns (full solid-color rows/cols at edges,
    non-zero color). Interior has dots of a specific color (non-zero, non-wall).
    Output: replace each dot with the color of the nearest wall (min distance to
    a wall row/col), keeping everything else unchanged.
    """
    if not pairs:
        return None

    def find_walls(grid):
        H = len(grid)
        W = len(grid[0]) if H > 0 else 0
        wall_rows = {}  # row_idx -> color (non-zero only)
        wall_cols = {}  # col_idx -> color (non-zero only)
        for r in range(H):
            colors = set(grid[r])
            if len(colors) == 1 and list(colors)[0] != 0:
                wall_rows[r] = list(colors)[0]
        for c in range(W):
            col_colors = set(grid[r][c] for r in range(H))
            if len(col_colors) == 1 and list(col_colors)[0] != 0:
                wall_cols[c] = list(col_colors)[0]
        return wall_rows, wall_cols

    def find_dot_color(grid, wall_rows, wall_cols):
        H = len(grid)
        W = len(grid[0]) if H > 0 else 0
        wall_row_set = set(wall_rows.keys())
        wall_col_set = set(wall_cols.keys())
        freq = {}
        for r in range(H):
            if r in wall_row_set:
                continue
            for c in range(W):
                if c in wall_col_set:
                    continue
                v = grid[r][c]
                if v != 0:
                    freq[v] = freq.get(v, 0) + 1
        if not freq:
            return None
        return max(freq, key=lambda k: freq[k])

    def apply_nearest_wall(grid, wall_rows, wall_cols, dot_color):
        import copy
        H = len(grid)
        W = len(grid[0]) if H > 0 else 0
        out = copy.deepcopy(grid)
        for r in range(H):
            for c in range(W):
                if grid[r][c] == dot_color:
                    best_dist = 10**9
                    best_color = dot_color
                    for wr_idx, wr_color in wall_rows.items():
                        d = abs(r - wr_idx)
                        if d < best_dist:
                            best_dist = d
                            best_color = wr_color
                    for wc_idx, wc_color in wall_cols.items():
                        d = abs(c - wc_idx)
                        if d < best_dist:
                            best_dist = d
                            best_color = wc_color
                    out[r][c] = best_color
        return out

    # Validate on training pairs
    for p in pairs:
        inp = p["input"]
        expected = p["output"]
        wr, wc = find_walls(inp)
        if not wr and not wc:
            return None
        dot_c = find_dot_color(inp, wr, wc)
        if dot_c is None:
            return None
        result = apply_nearest_wall(inp, wr, wc, dot_c)
        if result != expected:
            return None

    # Apply to test
    wr, wc = find_walls(test_inp)
    if not wr and not wc:
        return None
    dot_c = find_dot_color(test_inp, wr, wc)
    if dot_c is None:
        return None
    return apply_nearest_wall(test_inp, wr, wc, dot_c)


def _solve_largest_blob_to_3x3(pairs, test_inp):
    """
    3194b014: 20x20 input → 3x3 uniform output.
    Top-3 most-frequent colors are background/noise.
    Among remaining colors, the one with the most cells → 3x3 uniform of that color.
    """
    if not pairs:
        return None

    def get_dominant_color(grid):
        freq = {}
        for row in grid:
            for v in row:
                freq[v] = freq.get(v, 0) + 1
        if len(freq) < 4:
            return None
        sorted_colors = sorted(freq.keys(), key=lambda k: -freq[k])
        # Top-3 are noise/bg — skip them
        remaining = sorted_colors[3:]
        if not remaining:
            return None
        return max(remaining, key=lambda k: freq[k])

    # Validate on training pairs
    for p in pairs:
        inp = p["input"]
        expected = p["output"]
        H_out = len(expected)
        W_out = len(expected[0]) if H_out > 0 else 0
        flat = [expected[r][c] for r in range(H_out) for c in range(W_out)]
        if len(set(flat)) != 1:
            return None
        expected_color = flat[0]
        dom = get_dominant_color(inp)
        if dom != expected_color:
            return None

    # Apply to test
    dom = get_dominant_color(test_inp)
    if dom is None:
        return None
    H_out = len(pairs[0]["output"])
    W_out = len(pairs[0]["output"][0]) if H_out > 0 else 3
    return [[dom] * W_out for _ in range(H_out)]



def _solve_xor_halves(pairs, test_inp):
    """XOR top/bottom halves: exactly one nonzero → output_color, else 0.
    Handles optional divider row (uniform non-zero at row out_h).
    Covers: 31d5ba1a (no divider), 3428a4f5 (with divider)."""
    import numpy as _np
    def solve_one(inp, out_h, out_w, oc):
        A = _np.array(inp); H, W = A.shape
        if H == 2*out_h+1:
            div = A[out_h]
            if len(set(div.tolist()))!=1 or div[0]==0: return None
            top, bot = A[:out_h], A[out_h+1:]
        elif H == 2*out_h:
            top, bot = A[:out_h], A[out_h:]
        else:
            return None
        if top.shape!=(out_h,out_w) or bot.shape!=(out_h,out_w): return None
        res = _np.zeros((out_h,out_w),dtype=int)
        for r in range(out_h):
            for c in range(out_w):
                res[r,c] = oc if (int(top[r,c]!=0)+int(bot[r,c]!=0))==1 else 0
        return res.tolist()
    out_h=len(pairs[0]['output']); out_w=len(pairs[0]['output'][0])
    oc_set=set()
    for p in pairs: oc_set.update(v for v in _np.array(p['output']).flatten() if v!=0)
    if len(oc_set)!=1: return None
    oc=list(oc_set)[0]
    for p in pairs:
        if solve_one(p['input'],out_h,out_w,oc)!=p['output']: return None
    return solve_one(test_inp,out_h,out_w,oc)


def _solve_nor_halves(pairs, test_inp):
    """NOR top/bottom halves: both zero → output_color, else 0.
    Divider row (uniform non-zero) at index out_h. Covers: 0c9aba6e."""
    import numpy as _np
    def solve_one(inp, out_h, out_w, oc):
        A = _np.array(inp); H, W = A.shape
        if H == 2*out_h+1:
            div = A[out_h]
            if len(set(div.tolist()))!=1 or div[0]==0: return None
            top, bot = A[:out_h], A[out_h+1:]
        elif H == 2*out_h:
            top, bot = A[:out_h], A[out_h:]
        else:
            return None
        if top.shape!=(out_h,out_w) or bot.shape!=(out_h,out_w): return None
        res = _np.zeros((out_h,out_w),dtype=int)
        for r in range(out_h):
            for c in range(out_w):
                res[r,c] = oc if (top[r,c]==0 and bot[r,c]==0) else 0
        return res.tolist()
    out_h=len(pairs[0]['output']); out_w=len(pairs[0]['output'][0])
    oc_set=set()
    for p in pairs: oc_set.update(v for v in _np.array(p['output']).flatten() if v!=0)
    if len(oc_set)!=1: return None
    oc=list(oc_set)[0]
    for p in pairs:
        if solve_one(p['input'],out_h,out_w,oc)!=p['output']: return None
    return solve_one(test_inp,out_h,out_w,oc)


def _solve_tile_repeat_third(pairs, test_inp):
    """Output = first W//3 columns (input tiles 3x horizontally). Covers: 2dee498d."""
    import numpy as _np
    def solve_one(inp):
        A = _np.array(inp); H, W = A.shape
        if W%3!=0: return None
        return A[:,:W//3].tolist()
    for p in pairs:
        if solve_one(p['input'])!=p['output']: return None
    return solve_one(test_inp)


def _solve_bbox_tile_horizontal(pairs, test_inp):
    """Extract bounding box of non-zero signal; tile 2x horizontally. Covers: 28bf18c6."""
    import numpy as _np
    def solve_one(inp):
        A = _np.array(inp)
        rows = _np.any(A!=0,axis=1); cols = _np.any(A!=0,axis=0)
        if not rows.any() or not cols.any(): return None
        r0,r1 = _np.where(rows)[0][[0,-1]]; c0,c1 = _np.where(cols)[0][[0,-1]]
        bbox = A[r0:r1+1, c0:c1+1]
        return _np.concatenate([bbox,bbox],axis=1).tolist()
    for p in pairs:
        if solve_one(p['input'])!=p['output']: return None
    return solve_one(test_inp)


def _solve_staircase_blob_count(pairs, test_inp):
    """
    2753e76c: Count 4-connected blobs per non-zero color.
    Sort colors by blob-count descending → row order + filled cells per row.
    Output: N rows x max_blobs cols; row k is right-aligned, filled with color[k].
    """
    import numpy as _np
    from collections import deque as _deque
    def _count_blobs(arr, color):
        visited = set()
        count = 0
        positions = set(zip(*_np.where(arr == color)))
        for pos in positions:
            if pos not in visited:
                count += 1
                q = _deque([pos])
                while q:
                    r, c = q.popleft()
                    if (r,c) in visited: continue
                    visited.add((r,c))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nb=(r+dr,c+dc)
                        if nb in positions and nb not in visited:
                            q.append(nb)
        return count
    def solve_one(inp):
        A = _np.array(inp)
        nz_colors = set(A[A!=0].flatten().tolist())
        if not nz_colors: return None
        blobs = {c: _count_blobs(A, c) for c in nz_colors}
        order = sorted(nz_colors, key=lambda c: (-blobs[c], c))
        W = blobs[order[0]]
        N = len(order)
        out = _np.zeros((N, W), dtype=int)
        for k, color in enumerate(order):
            b = blobs[color]
            out[k, W-b:] = color
        return out.tolist()
    for p in pairs:
        if solve_one(p['input']) != p['output']: return None
    return solve_one(test_inp)


def _solve_minority_bbox_fill(pairs, test_inp):
    """
    23b5c85d: Find the least-frequent non-zero color in the input.
    Compute its bounding box. Return a rectangle of that size filled
    entirely with that color.
    """
    import numpy as _np
    from collections import Counter as _Counter
    def solve_one(inp):
        A = _np.array(inp)
        nz = A[A != 0].flatten()
        if len(nz) == 0: return None
        cnt = _Counter(nz.tolist())
        min_count = min(cnt.values())
        candidates = sorted([c for c, v in cnt.items() if v == min_count])
        color = candidates[0]
        rows, cols = _np.where(A == color)
        if len(rows) == 0: return None
        h = int(rows.max() - rows.min()) + 1
        w = int(cols.max() - cols.min()) + 1
        return _np.full((h, w), color, dtype=int).tolist()
    for p in pairs:
        if solve_one(p['input']) != p['output']: return None
    return solve_one(test_inp)


def _solve_grid_divider_sections(pairs, test_inp):
    """
    1190e5a7: Minority color forms full-row and full-col divider lines.
    Count sections between horizontal dividers × sections between vertical dividers
    → output shape (h_sections × v_sections), filled with majority color.
    """
    import numpy as _np
    from collections import Counter as _Counter
    def solve_one(inp):
        A = _np.array(inp)
        H, W = A.shape
        cnt = _Counter(A.flatten().tolist())
        # Find colors forming full rows
        colors_full_rows = set()
        for r in range(H):
            row = A[r, :]
            if _np.all(row == row[0]) and row[0] != 0:
                colors_full_rows.add(int(row[0]))
        # Find colors forming full cols
        colors_full_cols = set()
        for c in range(W):
            col = A[:, c]
            if _np.all(col == col[0]) and col[0] != 0:
                colors_full_cols.add(int(col[0]))
        div_colors = colors_full_rows & colors_full_cols
        if not div_colors:
            div_colors = colors_full_rows | colors_full_cols
        if not div_colors: return None
        div_color = min(div_colors, key=lambda c: cnt.get(c, 0))
        div_rows = [r for r in range(H) if _np.all(A[r, :] == div_color)]
        div_cols = [c for c in range(W) if _np.all(A[:, c] == div_color)]
        h_sections = len(div_rows) + 1
        v_sections = len(div_cols) + 1
        remaining = {c: v for c, v in cnt.items() if c != div_color and c != 0}
        if not remaining: return None
        maj_color = max(remaining, key=lambda c: remaining[c])
        return _np.full((h_sections, v_sections), maj_color, dtype=int).tolist()
    for p in pairs:
        if solve_one(p['input']) != p['output']: return None
    return solve_one(test_inp)



# ── _solve_quadrant_blobs ────────────────────────────────────────────────────
def _solve_quadrant_blobs(pairs, test_inp):
    """
    19bb5feb: Dominant non-zero color forms a rectangle; other non-dominant
    colored blobs are placed in its quadrants. Output is 2x2 with each blob
    color at its quadrant position (UL/UR/LL/LR), 0 for empty.
    """
    import numpy as _np
    from collections import Counter
    def solve_one(inp):
        A = _np.array(inp)
        flat = A.flatten().tolist()
        cnt = Counter(flat)
        nz_cnt = {c: v for c, v in cnt.items() if c != 0}
        if not nz_cnt:
            return None
        dom = max(nz_cnt, key=lambda c: nz_cnt[c])
        rows, cols = _np.where(A == dom)
        if len(rows) == 0:
            return None
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        row_mid = (r0 + r1) / 2.0
        col_mid = (c0 + c1) / 2.0
        out = [[0, 0], [0, 0]]
        for c in set(flat) - {0, dom}:
            rs, cs = _np.where(A == c)
            if len(rs) == 0:
                continue
            cr = (rs.min() + rs.max()) / 2.0
            cc = (cs.min() + cs.max()) / 2.0
            ri = 0 if cr < row_mid else 1
            ci = 0 if cc < col_mid else 1
            out[ri][ci] = int(c)
        return out

    for p in pairs:
        if solve_one(p['input']) != p['output']:
            return None
    return solve_one(test_inp)


# ── _solve_minority_bbox_extract ─────────────────────────────────────────────
def _solve_minority_bbox_extract(pairs, test_inp):
    """
    0b148d64: Exactly 2 non-zero colors; minority (less frequent) = signal,
    majority = background. Extract bounding box of signal; replace majority
    pixels with 0 inside crop; return crop.
    """
    import numpy as _np
    from collections import Counter
    def solve_one(inp):
        A = _np.array(inp)
        flat = A.flatten().tolist()
        cnt = Counter(flat)
        nz_cnt = {c: v for c, v in cnt.items() if c != 0}
        if len(nz_cnt) != 2:
            return None
        sig = min(nz_cnt, key=lambda c: nz_cnt[c])
        bg  = max(nz_cnt, key=lambda c: nz_cnt[c])
        rows, cols = _np.where(A == sig)
        if len(rows) == 0:
            return None
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        crop = A[r0:r1+1, c0:c1+1].copy()
        crop[crop == bg] = 0
        return crop.tolist()

    for p in pairs:
        if solve_one(p['input']) != p['output']:
            return None
    return solve_one(test_inp)



# ── _solve_checkerboard_grid_on_empty ────────────────────────────────────────
def _solve_checkerboard_grid_on_empty(pairs, test_inp):
    """
    332efdb3: Input is all-zero; output is a grid pattern:
    out[r][c] = color if (r%2==0 or c%2==0) else 0.
    Color is constant across all training outputs.
    """
    from collections import Counter
    def get_color(grid):
        flat = [v for row in grid for v in row]
        nz = [v for v in flat if v != 0]
        if not nz: return None
        return Counter(nz).most_common(1)[0][0]

    def check_all_zero(grid):
        return all(v == 0 for row in grid for v in row)

    def make_grid(H, W, color):
        return [[color if (r%2==0 or c%2==0) else 0 for c in range(W)] for r in range(H)]

    if not all(check_all_zero(p['input']) for p in pairs):
        return None
    color = get_color(pairs[0]['output'])
    if color is None:
        return None
    for p in pairs:
        H, W = len(p['output']), len(p['output'][0])
        if make_grid(H, W, color) != p['output']:
            return None
    H, W = len(test_inp), len(test_inp[0])
    return make_grid(H, W, color)


# ── _solve_spiral_on_empty ────────────────────────────────────────────────────
def _solve_spiral_on_empty(pairs, test_inp):
    """
    28e73c20: Input is all-zero; output is a clockwise spiral starting at (0,0)
    drawn with a fixed color. Arm lengths: N, N-1,N-1, N-3,N-3, ... > 0.
    """
    from collections import Counter

    def check_all_zero(grid):
        return all(v == 0 for row in grid for v in row)

    def get_color(grid):
        flat = [v for row in grid for v in row]
        nz = [v for v in flat if v != 0]
        if not nz: return None
        return Counter(nz).most_common(1)[0][0]

    def make_spiral(H, W, col):
        o = [[0]*W for _ in range(H)]
        N = max(H, W)
        arms = [N]
        l = N - 1
        while l > 0:
            arms += [l, l]
            l -= 2
        dirs = [(0,1),(1,0),(0,-1),(-1,0)]
        r, c, d = 0, 0, 0
        for arm_len in arms:
            dr, dc = dirs[d % 4]
            for step in range(arm_len):
                if 0 <= r < H and 0 <= c < W:
                    o[r][c] = col
                if step < arm_len - 1:
                    r += dr; c += dc
            d += 1
            dr2, dc2 = dirs[d % 4]
            r += dr2; c += dc2
        return o

    if not all(check_all_zero(p['input']) for p in pairs):
        return None
    color = get_color(pairs[0]['output'])
    if color is None:
        return None
    for p in pairs:
        H, W = len(p['output']), len(p['output'][0])
        if make_spiral(H, W, color) != p['output']:
            return None
    H, W = len(test_inp), len(test_inp[0])
    return make_spiral(H, W, color)


def _solve_color_shift_signal(pairs, test_inp):
    """
    342dd610: Each non-background (non-8) pixel shifts by a fixed per-color vector.
    Derive color->shift map from training pairs (must be consistent across all pairs).
    Background=8 stays 8.
    """
    import numpy as _np
    from collections import Counter

    def _derive_shifts(inp, out):
        H, W = len(inp), len(inp[0])
        in_pix, out_pix = {}, {}
        for r in range(H):
            for c in range(W):
                v = inp[r][c]
                if v != 8:
                    in_pix.setdefault(v, []).append((r, c))
                v2 = out[r][c]
                if v2 != 8:
                    out_pix.setdefault(v2, []).append((r, c))
        sm = {}
        for color in in_pix:
            if color not in out_pix:
                return None
            ip = sorted(in_pix[color])
            op = sorted(out_pix[color])
            if len(ip) != len(op):
                return None
            shifts = set((ro - ri, co - ci) for (ri, ci), (ro, co) in zip(ip, op))
            if len(shifts) != 1:
                return None
            sm[color] = shifts.pop()
        return sm

    # Build global shift map validated across all training pairs
    global_shift = {}
    for p in pairs:
        sm = _derive_shifts(p['input'], p['output'])
        if sm is None:
            return None
        for col, sh in sm.items():
            if col in global_shift and global_shift[col] != sh:
                return None
            global_shift[col] = sh

    if not global_shift:
        return None

    def _apply(grid):
        H, W = len(grid), len(grid[0])
        out = [[8] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                v = grid[r][c]
                if v != 8:
                    if v not in global_shift:
                        return None
                    dr, dc = global_shift[v]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = v
        return out

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None or pred != p['output']:
            return None

    return _apply(test_inp)


def _solve_shift_left_fill5(pairs, test_inp):
    """
    32e9702f: Background=0 becomes 5 in output. Each non-0 signal cell
    shifts left by 1 column. Cells at col=0 are lost (off the edge).
    """
    import numpy as _np
    from collections import Counter

    def _apply(grid):
        H = len(grid)
        W = len(grid[0])
        out = [[5] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                v = grid[r][c]
                if v != 0:
                    nc = c - 1
                    if nc >= 0:
                        out[r][nc] = v
        return out

    for p in pairs:
        pred = _apply(p['input'])
        if pred != p['output']:
            return None

    return _apply(test_inp)


def _solve_perimeter_tile(pairs, test_inp):
    """
    30f42897: One contiguous group of N non-bg cells on the perimeter defines
    the signal. Period = 2N. Tile with P/(2N) copies at offsets start, start+2N, ...
    Perimeter traversal: top L->R, right T->B (excl top), bottom R->L (excl right),
    left B->T (excl both corners). Background=8.
    Signal may wrap around perimeter end-to-start; use circular start detection.
    """
    import numpy as _np
    from collections import Counter

    def _perim(H, W):
        pos = []
        for c in range(W):
            pos.append((0, c))
        for r in range(1, H):
            pos.append((r, W - 1))
        for c in range(W - 2, -1, -1):
            pos.append((H - 1, c))
        for r in range(H - 2, 0, -1):
            pos.append((r, 0))
        return pos

    def _apply(grid, bg=8):
        H, W = len(grid), len(grid[0])
        pos = _perim(H, W)
        P = len(pos)
        vals = [grid[r][c] for r, c in pos]

        # Circular start: first non-bg cell that follows a bg cell (mod P)
        start = None
        for k in range(P):
            if vals[k] != bg and vals[(k - 1) % P] == bg:
                start = k
                break
        if start is None:
            return None

        # Collect signal circularly
        signal = []
        j = 0
        while vals[(start + j) % P] != bg and j < P:
            signal.append(vals[(start + j) % P])
            j += 1
        N = len(signal)

        if N == 0 or P % (2 * N) != 0:
            return None

        # Tile: P/(2N) copies each separated by N bg cells
        out_vals = [bg] * P
        num_copies = P // (2 * N)
        for k in range(num_copies):
            cs = (start + k * 2 * N) % P
            for j in range(N):
                out_vals[(cs + j) % P] = signal[j]

        out = [row[:] for row in grid]
        for i, (r, c) in enumerate(pos):
            out[r][c] = out_vals[i]
        return out

    for p in pairs:
        pred = _apply(p['input'])
        if pred is None or pred != p['output']:
            return None

    return _apply(test_inp)


def _solve_complement_blocks(pairs, test_inp):
    """
    1c0d0a4b: Find all-zero divider rows and divider cols.
    For each cell:
      - if row or col is a divider -> output 0
      - elif input==0               -> output 2
      - else (input==8)             -> output 0
    """
    import numpy as _np
    from collections import Counter

    def _apply(grid):
        H, W = len(grid), len(grid[0])
        div_rows = {r for r in range(H) if all(grid[r][c] == 0 for c in range(W))}
        div_cols = {c for c in range(W) if all(grid[r][c] == 0 for r in range(H))}
        out = [[0] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                if r in div_rows or c in div_cols:
                    out[r][c] = 0
                elif grid[r][c] == 0:
                    out[r][c] = 2
                else:
                    out[r][c] = 0
        return out

    for p in pairs:
        pred = _apply(p['input'])
        if pred != p['output']:
            return None

    return _apply(test_inp)


# new_solvers.py — Additional DSA solver functions for ARC-AGI-2
# Each function follows signature: solver(pairs, test_inp) -> grid | None
# Verified against training examples before implementation.

import collections


def _get_bg(grid):
    """Return most frequent color (background)."""
    counts = collections.Counter()
    for row in grid:
        counts.update(row)
    return counts.most_common(1)[0][0]


def _copy_grid(grid):
    return [list(row) for row in grid]


# ─────────────────────────────────────────────────────────────
# 1. 178fcbfb — isolated dot → full row (colors 3,1) or full col (color 2)
# ─────────────────────────────────────────────────────────────
def _solve_row_col_from_single_dots(pairs, test_inp):
    """
    Any isolated single dot of color 3 or 1 → extend as full row with that color.
    Any isolated single dot of color 2 → extend as full column with that color.
    Multiple dots coexist; each extends independently.
    Verified on task 178fcbfb.
    """
    ROW_COLORS = {3, 1}
    COL_COLORS = {2}

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])
        out = _copy_grid(grid)

        # Find all non-bg cells
        for r in range(R):
            for c in range(C):
                v = grid[r][c]
                if v == bg:
                    continue
                if v in ROW_COLORS:
                    for cc in range(C):
                        out[r][cc] = v
                elif v in COL_COLORS:
                    for rr in range(R):
                        out[rr][c] = v
        return out

    # Validate on training pairs
    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 2. 21f83797 — two isolated 2-dots → cross lines + interior rectangle fill
# ─────────────────────────────────────────────────────────────
def _solve_two_dot_cross_rect_fill(pairs, test_inp):
    """
    Find exactly 2 isolated dots of color 2.
    Draw full row + full column through each (4 lines total).
    Fill interior rectangle formed by crossing lines with color 1.
    Verified on task 21f83797.
    """
    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        dots = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 2]
        if len(dots) != 2:
            return None

        (r1, c1), (r2, c2) = dots
        if r1 == r2 or c1 == c2:
            return None  # need distinct rows and cols

        rmin, rmax = min(r1, r2), max(r1, r2)
        cmin, cmax = min(c1, c2), max(c1, c2)

        out = _copy_grid(grid)
        # Interior fill first (so lines overwrite if needed)
        for r in range(rmin + 1, rmax):
            for c in range(cmin + 1, cmax):
                out[r][c] = 1
        # Full rows
        for c in range(C):
            out[r1][c] = 2
            out[r2][c] = 2
        # Full cols
        for r in range(R):
            out[r][c1] = 2
            out[r][c2] = 2
        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 3. 14b8e18c — closed hollow rectangle of 6s → corner outward marks of 2
# ─────────────────────────────────────────────────────────────
def _solve_closed_rect_corner_marks(pairs, test_inp):
    """
    Find each connected component of color 6 that forms a valid closed hollow rectangle:
    all perimeter cells of bounding box are 6, all interior cells are bg.
    For each valid rectangle, place color 2 at 2 outward-adjacent cells per corner (8 total).
    Verified on task 14b8e18c.
    """
    def _find_components(grid, color):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        components = []
        for sr in range(R):
            for sc in range(C):
                if grid[sr][sc] == color and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != color:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            stack.append((r+dr, c+dc))
                    components.append(comp)
        return components

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])
        out = _copy_grid(grid)

        comps = _find_components(grid, 6)
        for comp in comps:
            rs = [r for r, c in comp]
            cs = [c for r, c in comp]
            r1, r2 = min(rs), max(rs)
            c1, c2 = min(cs), max(cs)

            if r2 - r1 < 2 or c2 - c1 < 2:
                continue  # too small

            # Check all perimeter cells are 6
            valid = True
            cell_set = set(comp)
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    is_perim = (r == r1 or r == r2 or c == c1 or c == c2)
                    is_interior = (r1 < r < r2 and c1 < c < c2)
                    if is_perim and (r, c) not in cell_set:
                        valid = False
                        break
                    if is_interior and grid[r][c] != bg:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            # Place 2s outward from each corner (2 cells per corner)
            corner_marks = [
                (r1 - 1, c1), (r1, c1 - 1),   # top-left
                (r1 - 1, c2), (r1, c2 + 1),   # top-right
                (r2 + 1, c1), (r2, c1 - 1),   # bottom-left
                (r2 + 1, c2), (r2, c2 + 1),   # bottom-right
            ]
            for mr, mc in corner_marks:
                if 0 <= mr < R and 0 <= mc < C:
                    out[mr][mc] = 2

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 4. 292dd178 — C-shaped (3-sided) hollow rect of 1s → fill + extend outward
# ─────────────────────────────────────────────────────────────
def _solve_open_rect_fill_exterior(pairs, test_inp):
    """
    Find bounding box of 1s. Exactly 3 of the 4 borders are complete rows/cols of 1s.
    One border has exactly one gap cell.
    Fill all interior cells with 2.
    Fill the gap cell with 2.
    Extend outward from the gap position in perpendicular direction to grid edge with 2.
    Verified on task 292dd178.
    """
    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        cells_1 = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 1]
        if not cells_1:
            return None

        rs = [r for r, c in cells_1]
        cs = [c for r, c in cells_1]
        r1, r2 = min(rs), max(rs)
        c1, c2 = min(cs), max(cs)

        if r2 - r1 < 2 or c2 - c1 < 2:
            return None

        cell_set = set(cells_1)

        # Check borders: collect missing cells per border
        top_missing = [(r1, c) for c in range(c1, c2 + 1) if (r1, c) not in cell_set]
        bot_missing = [(r2, c) for c in range(c1, c2 + 1) if (r2, c) not in cell_set]
        left_missing = [(r, c1) for r in range(r1, r2 + 1) if (r, c1) not in cell_set]
        right_missing = [(r, c2) for r in range(r1, r2 + 1) if (r, c2) not in cell_set]

        # Exactly 3 borders complete (0 missing), 1 border has exactly 1 gap
        borders = [
            ('top', top_missing, 'row'),
            ('bot', bot_missing, 'row'),
            ('left', left_missing, 'col'),
            ('right', right_missing, 'col'),
        ]

        gap_border = None
        for name, missing, kind in borders:
            if len(missing) == 1:
                if gap_border is not None:
                    return None  # more than one gap border
                gap_border = (name, missing[0], kind)
            elif len(missing) != 0:
                return None  # border not clean

        if gap_border is None:
            return None

        name, (gr, gc), kind = gap_border

        out = _copy_grid(grid)

        # Fill interior with 2
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                out[r][c] = 2

        # Fill gap with 2
        out[gr][gc] = 2

        # Extend outward from gap
        if name == 'top':
            # gap is on top border → extend upward
            for r in range(gr - 1, -1, -1):
                out[r][gc] = 2
        elif name == 'bot':
            # gap is on bottom border → extend downward
            for r in range(gr + 1, R):
                out[r][gc] = 2
        elif name == 'left':
            # gap is on left border → extend leftward
            for c in range(gc - 1, -1, -1):
                out[gr][c] = 2
        elif name == 'right':
            # gap is on right border → extend rightward
            for c in range(gc + 1, C):
                out[gr][c] = 2

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 5. 351d6448 — arithmetic sequence rows separated by 5-rows → next row
# ─────────────────────────────────────────────────────────────
def _solve_arith_seq_5_separator(pairs, test_inp):
    """
    Grid sections separated by full-width rows of 5s.
    Each section has a row of non-bg cells with increasing length (arithmetic).
    Output: single 3-row block (blank, next-length row, blank) continuing the sequence.
    Verified on task 351d6448.
    """
    def _parse_sections(grid, bg):
        R, C = len(grid), len(grid[0])
        sections = []
        current = []
        for r in range(R):
            if all(grid[r][c] == 5 for c in range(C)):
                if current:
                    sections.append(current)
                    current = []
            else:
                current.append(grid[r])
        if current:
            sections.append(current)
        return sections

    def _get_filled_row(section, bg):
        """Return (color, length) of the non-bg row in section."""
        for row in section:
            non_bg = [v for v in row if v != bg]
            if non_bg:
                return non_bg[0], len(non_bg)
        return None, 0

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        sections = _parse_sections(grid, bg)
        if len(sections) < 2:
            return None

        entries = []
        for sec in sections:
            color, length = _get_filled_row(sec, bg)
            if color is None:
                return None
            entries.append((color, length))

        # Check arithmetic sequence of lengths
        lengths = [l for _, l in entries]
        if len(lengths) < 2:
            return None
        diff = lengths[1] - lengths[0]
        for i in range(1, len(lengths)):
            if lengths[i] - lengths[i-1] != diff:
                return None

        # Next length
        next_len = lengths[-1] + diff
        color = entries[-1][0]

        if next_len <= 0 or next_len > C:
            return None

        blank = [bg] * C
        seq_row = [color] * next_len + [bg] * (C - next_len)
        return [blank, seq_row, blank]

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 6. 358ba94e — select blob with most interior background cells
# ─────────────────────────────────────────────────────────────
def _solve_select_most_interior_nulls_region(pairs, test_inp):
    """
    Multiple same-sized rectangular blobs of color C with interior background cells.
    Output: the blob with the most interior background cells.
    Verified on task 358ba94e.
    """
    def _find_blobs(grid, bg):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        blobs = []
        for sr in range(R):
            for sc in range(C):
                v = grid[sr][sc]
                if v != bg and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    color = v
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != color:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            stack.append((r+dr, c+dc))
                    rs = [r for r, c in comp]
                    cs = [c for r, c in comp]
                    blobs.append({
                        'color': color,
                        'cells': set(comp),
                        'r1': min(rs), 'r2': max(rs),
                        'c1': min(cs), 'c2': max(cs),
                    })
        return blobs

    def _interior_bg_count(blob, grid, bg):
        r1, r2, c1, c2 = blob['r1'], blob['r2'], blob['c1'], blob['c2']
        count = 0
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if grid[r][c] == bg:
                    count += 1
        return count

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        blobs = _find_blobs(grid, bg)
        if not blobs:
            return None

        best = max(blobs, key=lambda b: _interior_bg_count(b, grid, bg))
        r1, r2, c1, c2 = best['r1'], best['r2'], best['c1'], best['c2']

        out = [[bg] * (c2 - c1 + 1) for _ in range(r2 - r1 + 1)]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                out[r - r1][c - c1] = grid[r][c]
        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 7. 1c786137 — extract interior of rectangle bordered by color 4
# ─────────────────────────────────────────────────────────────
def _solve_extract_rect_interior(pairs, test_inp):
    """
    Find rectangle bordered by color 4. Output: interior content (strip the 4-border).
    Verified on task 1c786137.
    """
    def _apply(grid):
        R, C = len(grid), len(grid[0])

        # Find bounding box of color 4
        cells_4 = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 4]
        if not cells_4:
            return None

        rs = [r for r, c in cells_4]
        cs = [c for r, c in cells_4]
        r1, r2 = min(rs), max(rs)
        c1, c2 = min(cs), max(cs)

        if r2 - r1 < 2 or c2 - c1 < 2:
            return None

        # Return interior
        return [list(grid[r][c1+1:c2]) for r in range(r1+1, r2)]

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 8. 25e02866 — overlay multiple same-bordered rectangles of 3s
# ─────────────────────────────────────────────────────────────
def _solve_overlay_rect_interiors(pairs, test_inp):
    """
    Multiple same-bordered rectangles of 3s. Overlay/merge interiors:
    any non-bg, non-3 cell from any rectangle appears in output at its interior position.
    Verified on task 25e02866.
    """
    def _find_rect_components(grid, border_color, bg):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        rects = []
        for sr in range(R):
            for sc in range(C):
                if grid[sr][sc] == border_color and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != border_color:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            stack.append((r+dr, c+dc))
                    rs = [r for r, c in comp]
                    cs = [c for r, c in comp]
                    rects.append((min(rs), max(rs), min(cs), max(cs)))
        return rects

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        rects = _find_rect_components(grid, 3, bg)
        if len(rects) < 2:
            return None

        # All rects should be same size
        sizes = [(r2 - r1, c2 - c1) for r1, r2, c1, c2 in rects]
        if len(set(sizes)) != 1:
            return None

        h, w = sizes[0]
        if h < 2 or w < 2:
            return None

        # Build output of interior size
        interior_h = h - 1
        interior_w = w - 1
        out = [[bg] * interior_w for _ in range(interior_h)]

        for r1, r2, c1, c2 in rects:
            for r in range(r1 + 1, r2):
                for c in range(c1 + 1, c2):
                    v = grid[r][c]
                    if v != bg and v != 3:
                        out[r - r1 - 1][c - c1 - 1] = v

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 9. 1b59e163 / 20981f0e — template stamp to marker dots
# ─────────────────────────────────────────────────────────────
def _solve_shape_stamp_to_marker(pairs, test_inp):
    """
    Template shapes of color 1 (or 9) each with a unique interior color marker.
    Isolated same-color single dots elsewhere → stamp copy of template at that location.
    Original shapes remain; marker dots removed; stamped copies added.
    Verified on tasks 1b59e163 and 20981f0e.
    """
    def _find_components_any(grid, bg):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        comps = []
        for sr in range(R):
            for sc in range(C):
                if grid[sr][sc] != bg and not visited[sr][sc]:
                    comp = []
                    color = grid[sr][sc]
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != color:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            stack.append((r+dr, c+dc))
                    comps.append({'color': color, 'cells': comp})
        return comps

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        comps = _find_components_any(grid, bg)

        # Identify template shape color: the color that forms large multi-cell shapes
        # Templates are multi-cell shapes of one color (typically 1 or 9)
        # Markers are single isolated cells of unique colors inside templates
        # Lone dots are single isolated cells of same color as markers outside templates

        # Group components by color
        by_color = collections.defaultdict(list)
        for comp in comps:
            by_color[comp['color']].append(comp)

        # Find the dominant shape color (large components)
        shape_color = None
        for color, clist in by_color.items():
            if color == bg:
                continue
            large = [c for c in clist if len(c['cells']) > 1]
            if large:
                # Could be the template color
                shape_color = color
                break

        if shape_color is None:
            return None

        shape_comps = [c for c in by_color[shape_color] if len(c['cells']) > 1]
        if not shape_comps:
            return None

        # For each template (large shape), find its interior marker color
        # Interior = cells within bounding box not of shape_color and not bg
        templates = []
        for sc in shape_comps:
            cells = sc['cells']
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            r1, r2, c1, c2 = min(rs), max(rs), min(cs), max(cs)

            # Find interior marker
            marker_color = None
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    v = grid[r][c]
                    if v != bg and v != shape_color:
                        marker_color = v
            if marker_color is not None:
                templates.append({
                    'cells': cells,
                    'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2,
                    'marker_color': marker_color,
                })

        if not templates:
            return None

        # Build map: marker_color → template
        marker_to_template = {}
        for t in templates:
            marker_to_template[t['marker_color']] = t

        # Find lone single dots (size-1 components) whose color matches a marker
        lone_dots = []
        for color, clist in by_color.items():
            if color in marker_to_template:
                singletons = [c for c in clist if len(c['cells']) == 1]
                for s in singletons:
                    # Make sure it's not inside a template bounding box
                    sr, sc2 = s['cells'][0]
                    tmpl = marker_to_template[color]
                    if not (tmpl['r1'] <= sr <= tmpl['r2'] and tmpl['c1'] <= sc2 <= tmpl['c2']):
                        lone_dots.append({'color': color, 'r': sr, 'c': sc2})

        out = _copy_grid(grid)

        # Remove all lone dots
        for dot in lone_dots:
            out[dot['r']][dot['c']] = bg

        # Stamp copies of templates at lone dot locations
        for dot in lone_dots:
            tmpl = marker_to_template[dot['color']]
            # Find the marker position within template
            marker_r, marker_c = None, None
            for r in range(tmpl['r1'], tmpl['r2'] + 1):
                for c in range(tmpl['c1'], tmpl['c2'] + 1):
                    if grid[r][c] == dot['color']:
                        marker_r, marker_c = r, c
                        break
                if marker_r is not None:
                    break

            if marker_r is None:
                continue

            dr = dot['r'] - marker_r
            dc = dot['c'] - marker_c

            for (tr, tc) in tmpl['cells']:
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < R and 0 <= nc < C:
                    out[nr][nc] = shape_color

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 10. 2601afb7 — sort columns by height of non-bg content (tallest leftmost)
# ─────────────────────────────────────────────────────────────
def _solve_sort_cols_by_height(pairs, test_inp):
    """
    Sort columns by height of non-background content.
    Tallest column goes to leftmost position.
    Verified on task 2601afb7.
    """
    def _col_height(grid, c, bg):
        for r in range(len(grid)):
            if grid[r][c] != bg:
                return len(grid) - r
        return 0

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        heights = [(_col_height(grid, c, bg), c) for c in range(C)]
        sorted_cols = [c for _, c in sorted(heights, key=lambda x: -x[0])]

        out = [[grid[r][sorted_cols[c]] for c in range(C)] for r in range(R)]
        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 11. 13713586 — fill toward 5-wall (full row or col of 5s), closer wins
# ─────────────────────────────────────────────────────────────
def _solve_forward_fill_toward_5_wall(pairs, test_inp):
    """
    5-wall is a full row or full column of 5s.
    Each non-bg, non-5 cell fills toward the wall along its row (wall=col) or col (wall=row).
    In overlap zones, cell closer to wall takes precedence (paint farthest→closest).
    Verified on task 13713586.
    """
    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])
        out = _copy_grid(grid)

        # Find wall: full row of 5s or full col of 5s
        wall_row = None
        wall_col = None
        for r in range(R):
            if all(grid[r][c] == 5 for c in range(C)):
                wall_row = r
                break
        for c in range(C):
            if all(grid[r][c] == 5 for r in range(R)):
                wall_col = c
                break

        if wall_row is None and wall_col is None:
            return None

        if wall_row is not None:
            # Wall is a row; each colored cell fills toward wall_row along its column
            for c in range(C):
                # Collect all non-bg, non-5 cells in this column
                colored_cells = []
                for r in range(R):
                    v = grid[r][c]
                    if v != bg and v != 5:
                        colored_cells.append((r, v))

                # Sort by distance to wall (farthest first so closer overwrites)
                colored_cells.sort(key=lambda x: -abs(x[0] - wall_row))

                for (r, v) in colored_cells:
                    # Fill from r toward wall_row
                    if r < wall_row:
                        step_range = range(r, wall_row)
                    else:
                        step_range = range(r, wall_row, -1)
                    for rr in step_range:
                        out[rr][c] = v

        if wall_col is not None:
            # Wall is a column; each colored cell fills toward wall_col along its row
            for r in range(R):
                colored_cells = []
                for c in range(C):
                    v = grid[r][c]
                    if v != bg and v != 5:
                        colored_cells.append((c, v))

                colored_cells.sort(key=lambda x: -abs(x[0] - wall_col))

                for (c, v) in colored_cells:
                    if c < wall_col:
                        step_range = range(c, wall_col)
                    else:
                        step_range = range(c, wall_col, -1)
                    for cc in step_range:
                        out[r][cc] = v

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 12. 1caeab9d — all blocks slide to row range of color-1 block
# ─────────────────────────────────────────────────────────────
def _solve_gravity_color1_anchor(pairs, test_inp):
    """
    All rectangular blocks (non-bg) slide to share the row range of the block
    whose pixel has color value 1. Column positions are preserved.
    Verified on task 1caeab9d.
    """
    def _find_rect_blocks(grid, bg):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        blocks = []
        for sr in range(R):
            for sc in range(C):
                v = grid[sr][sc]
                if v != bg and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != v:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            stack.append((r+dr, c+dc))
                    rs = [r for r, c in comp]
                    cs = [c for r, c in comp]
                    blocks.append({
                        'color': v,
                        'cells': comp,
                        'r1': min(rs), 'r2': max(rs),
                        'c1': min(cs), 'c2': max(cs),
                    })
        return blocks

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        blocks = _find_rect_blocks(grid, bg)
        if not blocks:
            return None

        # Find anchor block: the one with color value 1
        anchor = None
        for b in blocks:
            if b['color'] == 1:
                anchor = b
                break

        if anchor is None:
            return None

        anchor_r1, anchor_r2 = anchor['r1'], anchor['r2']
        block_h = anchor_r2 - anchor_r1  # height - 1

        out = [[bg] * C for _ in range(R)]

        for b in blocks:
            h = b['r2'] - b['r1']
            # Place block at anchor row range (top-aligned to anchor_r1)
            dr = anchor_r1 - b['r1']
            for (r, c) in b['cells']:
                nr = r + dr
                if 0 <= nr < R:
                    out[nr][c] = b['color']

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 13. 29700607 — header row maps colors → cols; dots extend row + column
# ─────────────────────────────────────────────────────────────
def _solve_header_row_col_expand(pairs, test_inp):
    """
    Header row maps colors to column positions.
    Scattered same-color single dots → extend each dot's row as full row;
    continue the mapped column downward from the header.
    Verified on task 29700607.
    """
    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        # Find header row: first row with multiple non-bg cells at distinct columns
        header_row = None
        color_to_col = {}
        for r in range(R):
            non_bg = [(c, grid[r][c]) for c in range(C) if grid[r][c] != bg]
            if len(non_bg) >= 2:
                header_row = r
                color_to_col = {v: c for c, v in non_bg}
                break

        if header_row is None:
            return None

        out = _copy_grid(grid)

        # Find lone dots (single cells not in header row)
        for r in range(R):
            if r == header_row:
                continue
            non_bg_in_row = [(c, grid[r][c]) for c in range(C) if grid[r][c] != bg]
            if len(non_bg_in_row) == 1:
                dot_c, dot_color = non_bg_in_row[0]
                # Extend full row with dot_color
                for c in range(C):
                    out[r][c] = dot_color
                # Extend mapped column downward from header+1
                if dot_color in color_to_col:
                    mapped_c = color_to_col[dot_color]
                    for rr in range(header_row + 1, r):
                        out[rr][mapped_c] = dot_color

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)




# new_solvers.py — Additional DSA solver functions for ARC-AGI-2
# Each function follows signature: solver(pairs, test_inp) -> grid | None
# Verified against training examples before implementation.

import collections


def _get_bg(grid):
    """Return most frequent color (background)."""
    counts = collections.Counter()
    for row in grid:
        counts.update(row)
    return counts.most_common(1)[0][0]


def _copy_grid(grid):
    return [list(row) for row in grid]


# ─────────────────────────────────────────────────────────────
# 1. 178fcbfb — isolated dot → full row (colors 3,1) or full col (color 2)
# ─────────────────────────────────────────────────────────────
def _solve_row_col_from_single_dots(pairs, test_inp):
    """
    Any isolated single dot of color 3 or 1 → extend as full row with that color.
    Any isolated single dot of color 2 → extend as full column with that color.
    Multiple dots coexist; each extends independently.
    Verified on task 178fcbfb.
    """
    ROW_COLORS = {3, 1}
    COL_COLORS = {2}

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])
        out = _copy_grid(grid)

        for r in range(R):
            for c in range(C):
                v = grid[r][c]
                if v == bg:
                    continue
                if v in ROW_COLORS:
                    for cc in range(C):
                        out[r][cc] = v
                elif v in COL_COLORS:
                    for rr in range(R):
                        out[rr][c] = v
        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 2. 21f83797 — two isolated 2-dots → cross lines + interior rectangle fill
# ─────────────────────────────────────────────────────────────
def _solve_two_dot_cross_rect_fill(pairs, test_inp):
    """
    Find exactly 2 isolated dots of color 2.
    Draw full row + full column through each (4 lines total).
    Fill interior rectangle formed by crossing lines with color 1.
    Verified on task 21f83797.
    """
    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        dots = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 2]
        if len(dots) != 2:
            return None

        (r1, c1), (r2, c2) = dots
        if r1 == r2 or c1 == c2:
            return None

        rmin, rmax = min(r1, r2), max(r1, r2)
        cmin, cmax = min(c1, c2), max(c1, c2)

        out = _copy_grid(grid)
        for r in range(rmin + 1, rmax):
            for c in range(cmin + 1, cmax):
                out[r][c] = 1
        for c in range(C):
            out[r1][c] = 2
            out[r2][c] = 2
        for r in range(R):
            out[r][c1] = 2
            out[r][c2] = 2
        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 3. 14b8e18c — closed hollow square of any color → corner outward marks of 2
#    FIXED: dynamic color, square bbox, remove interior check
# ─────────────────────────────────────────────────────────────
def _solve_closed_rect_corner_marks(pairs, test_inp):
    """
    For each non-bg color, find connected components.
    Component is valid if its bounding box is SQUARE (r2-r1 == c2-c1 >= 1)
    and the component cells are exactly the bounding box perimeter.
    For each valid component place color 2 at 2 outward cells per corner.
    Verified on task 14b8e18c.
    """
    def _find_components(grid, color):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        components = []
        for sr in range(R):
            for sc in range(C):
                if grid[sr][sc] == color and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != color:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((r + dr, c + dc))
                    components.append(comp)
        return components

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])
        out = _copy_grid(grid)

        all_colors = set(v for row in grid for v in row if v != bg)

        for color in all_colors:
            for comp in _find_components(grid, color):
                rs = [r for r, c in comp]
                cs = [c for r, c in comp]
                r1, r2 = min(rs), max(rs)
                c1, c2 = min(cs), max(cs)

                # Must be square and at least 2×2 bounding box
                if r2 - r1 != c2 - c1:
                    continue
                if r2 - r1 < 1:
                    continue

                # Compute expected perimeter
                perim = set()
                for r in range(r1, r2 + 1):
                    perim.add((r, c1))
                    perim.add((r, c2))
                for c in range(c1, c2 + 1):
                    perim.add((r1, c))
                    perim.add((r2, c))

                # Component must be exactly the perimeter (no extra cells, no missing)
                if set(comp) != perim:
                    continue

                # Place 2s outward from each corner
                corner_marks = [
                    (r1 - 1, c1), (r1, c1 - 1),
                    (r1 - 1, c2), (r1, c2 + 1),
                    (r2 + 1, c1), (r2, c1 - 1),
                    (r2 + 1, c2), (r2, c2 + 1),
                ]
                for mr, mc in corner_marks:
                    if 0 <= mr < R and 0 <= mc < C:
                        out[mr][mc] = 2

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 4. 292dd178 — C-shaped (3-sided) hollow rect of 1s → fill + extend outward
#    FIXED: per-component processing (multiple C-shapes per grid)
# ─────────────────────────────────────────────────────────────
def _solve_open_rect_fill_exterior(pairs, test_inp):
    """
    Find all connected components of color 1.
    For each component: exactly 3 of 4 borders complete, 1 border has 1 gap.
    Fill interior + gap + extend outward from gap with color 2.
    Verified on task 292dd178.
    """
    def _find_components(grid, color):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        components = []
        for sr in range(R):
            for sc in range(C):
                if grid[sr][sc] == color and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != color:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((r + dr, c + dc))
                    components.append(comp)
        return components

    def _process_component(comp, grid, out):
        """Apply C-shape fill logic for one component. Returns True on success."""
        R, C = len(grid), len(grid[0])
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        r1, r2 = min(rs), max(rs)
        c1, c2 = min(cs), max(cs)

        if r2 - r1 < 2 or c2 - c1 < 2:
            return False

        cell_set = set(comp)

        top_missing = [(r1, c) for c in range(c1, c2 + 1) if (r1, c) not in cell_set]
        bot_missing = [(r2, c) for c in range(c1, c2 + 1) if (r2, c) not in cell_set]
        left_missing = [(r, c1) for r in range(r1, r2 + 1) if (r, c1) not in cell_set]
        right_missing = [(r, c2) for r in range(r1, r2 + 1) if (r, c2) not in cell_set]

        borders = [
            ('top', top_missing),
            ('bot', bot_missing),
            ('left', left_missing),
            ('right', right_missing),
        ]

        gap_border = None
        for name, missing in borders:
            if len(missing) == 1:
                if gap_border is not None:
                    return False
                gap_border = (name, missing[0])
            elif len(missing) != 0:
                return False

        if gap_border is None:
            return False

        name, (gr, gc) = gap_border

        # Fill interior
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                out[r][c] = 2

        # Fill gap
        out[gr][gc] = 2

        # Extend outward
        if name == 'top':
            for r in range(gr - 1, -1, -1):
                out[r][gc] = 2
        elif name == 'bot':
            for r in range(gr + 1, R):
                out[r][gc] = 2
        elif name == 'left':
            for c in range(gc - 1, -1, -1):
                out[gr][c] = 2
        elif name == 'right':
            for c in range(gc + 1, C):
                out[gr][c] = 2

        return True

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        comps = _find_components(grid, 1)
        if not comps:
            return None

        out = _copy_grid(grid)
        any_success = False
        for comp in comps:
            if _process_component(comp, grid, out):
                any_success = True

        return out if any_success else None

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 5. 351d6448 — arithmetic sequence rows separated by 5-rows → next row
#    FIXED: track start_col; handle Case A (same content, arithmetic starts)
#           and Case B (same start, arithmetic lengths)
# ─────────────────────────────────────────────────────────────
def _solve_arith_seq_5_separator(pairs, test_inp):
    """
    Grid sections separated by full-width rows of 5s.
    Each section has one non-bg row.
    Case A: same content, start positions shift arithmetically.
    Case B: same start, lengths grow arithmetically.
    Output: 3-row block [blank, next_row, blank].
    Verified on task 351d6448.
    """
    def _parse_sections(grid, bg):
        R, C = len(grid), len(grid[0])
        sections = []
        current = []
        for r in range(R):
            if all(grid[r][c] == 5 for c in range(C)):
                if current:
                    sections.append(current)
                    current = []
            else:
                current.append(grid[r])
        if current:
            sections.append(current)
        return sections

    def _get_row_info(section, bg):
        """Return (start_col, content_list) of the non-bg row in section."""
        for row in section:
            non_bg_indices = [c for c, v in enumerate(row) if v != bg]
            if non_bg_indices:
                start = min(non_bg_indices)
                content = [row[c] for c in non_bg_indices]
                return start, content
        return None, None

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        sections = _parse_sections(grid, bg)
        if len(sections) < 2:
            return None

        entries = []
        for sec in sections:
            start, content = _get_row_info(sec, bg)
            if start is None:
                return None
            entries.append((start, content))

        starts = [s for s, _ in entries]
        contents = [c for _, c in entries]

        blank = [bg] * C

        # Case A: same content, arithmetic start positions
        if all(c == contents[0] for c in contents):
            if len(starts) >= 2:
                diff = starts[1] - starts[0]
                arith_ok = all(starts[i] - starts[i - 1] == diff
                               for i in range(1, len(starts)))
                if arith_ok and diff != 0:
                    next_start = starts[-1] + diff
                    if 0 <= next_start and next_start + len(contents[0]) <= C:
                        seq_row = [bg] * C
                        for i, v in enumerate(contents[0]):
                            seq_row[next_start + i] = v
                        return [blank, seq_row, blank]

        # Case B: same start position, lengths grow arithmetically (same color)
        if all(s == starts[0] for s in starts):
            lengths = [len(c) for c in contents]
            if len(lengths) >= 2:
                diff = lengths[1] - lengths[0]
                arith_ok = all(lengths[i] - lengths[i - 1] == diff
                               for i in range(1, len(lengths)))
                if arith_ok and diff != 0:
                    next_len = lengths[-1] + diff
                    color = contents[-1][0] if contents[-1] else bg
                    if 0 < next_len <= C:
                        seq_row = [color] * next_len + [bg] * (C - next_len)
                        if starts[0] > 0:
                            # place at start position
                            seq_row = [bg] * starts[0] + [color] * next_len
                            seq_row = seq_row[:C] + [bg] * max(0, C - len(seq_row))
                        return [blank, seq_row, blank]

        return None

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 6. 358ba94e — select blob with UNIQUE interior-null count
#    FIXED: use Counter to find blob with unique count (freq == 1)
# ─────────────────────────────────────────────────────────────
def _solve_select_most_interior_nulls_region(pairs, test_inp):
    """
    Multiple rectangular blobs of the same color with interior background cells.
    Output: the blob whose interior-bg count is UNIQUE (appears only once).
    Verified on task 358ba94e.
    """
    def _find_blobs(grid, bg):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        blobs = []
        for sr in range(R):
            for sc in range(C):
                v = grid[sr][sc]
                if v != bg and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    color = v
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != color:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((r + dr, c + dc))
                    rs2 = [r for r, c in comp]
                    cs2 = [c for r, c in comp]
                    blobs.append({
                        'color': color,
                        'cells': set(comp),
                        'r1': min(rs2), 'r2': max(rs2),
                        'c1': min(cs2), 'c2': max(cs2),
                    })
        return blobs

    def _interior_bg_count(blob, grid, bg):
        r1, r2, c1, c2 = blob['r1'], blob['r2'], blob['c1'], blob['c2']
        count = 0
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if grid[r][c] == bg:
                    count += 1
        return count

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        blobs = _find_blobs(grid, bg)
        if not blobs:
            return None

        counts = [_interior_bg_count(b, grid, bg) for b in blobs]
        freq = collections.Counter(counts)

        # Select the blob whose interior-bg count is unique (freq == 1)
        unique_blobs = [(b, cnt) for b, cnt in zip(blobs, counts) if freq[cnt] == 1]
        if len(unique_blobs) != 1:
            return None

        best = unique_blobs[0][0]
        r1, r2, c1, c2 = best['r1'], best['r2'], best['c1'], best['c2']

        out = [[bg] * (c2 - c1 + 1) for _ in range(r2 - r1 + 1)]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                out[r - r1][c - c1] = grid[r][c]
        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 7. 1c786137 — extract interior of rectangle (dynamic border color)
#    FIXED: try all non-bg colors, check complete perimeter
# ─────────────────────────────────────────────────────────────
def _solve_extract_rect_interior(pairs, test_inp):
    """
    For each non-bg color, check if all its cells form the exact perimeter of
    a bounding box. If so, extract and return the interior content.
    Verified on task 1c786137.
    """
    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        all_colors = set(v for row in grid for v in row if v != bg)

        for color in all_colors:
            cells = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == color]
            if not cells:
                continue

            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            r1, r2 = min(rs), max(rs)
            c1, c2 = min(cs), max(cs)

            if r2 - r1 < 2 or c2 - c1 < 2:
                continue  # need interior space

            # Compute expected perimeter
            perim = set()
            for r in range(r1, r2 + 1):
                perim.add((r, c1))
                perim.add((r, c2))
            for c in range(c1, c2 + 1):
                perim.add((r1, c))
                perim.add((r2, c))

            if set(cells) != perim:
                continue

            # Extract interior
            interior = [list(grid[r][c1 + 1:c2]) for r in range(r1 + 1, r2)]
            if interior:
                return interior

        return None

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 8. 25e02866 — overlay multiple same-bordered rectangles
#    FIXED: dynamic border color, full h×w output (including border)
# ─────────────────────────────────────────────────────────────
def _solve_overlay_rect_interiors(pairs, test_inp):
    """
    Find border color: non-bg color whose components are all same-sized valid
    rectangle perimeters (≥2 components). Build one full h×w output:
    perimeter cells = border_color, interior cells = bg initially.
    Overlay non-bg, non-border interior markers from all rects at relative offsets.
    Verified on task 25e02866.
    """
    def _find_rect_components(grid, border_color):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        rects = []
        for sr in range(R):
            for sc in range(C):
                if grid[sr][sc] == border_color and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != border_color:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((r + dr, c + dc))
                    rs2 = [r for r, c in comp]
                    cs2 = [c for r, c in comp]
                    rects.append({
                        'comp': set(comp),
                        'r1': min(rs2), 'r2': max(rs2),
                        'c1': min(cs2), 'c2': max(cs2),
                    })
        return rects

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        all_colors = set(v for row in grid for v in row if v != bg)

        border_color = None
        best_rects = None

        for color in all_colors:
            rects = _find_rect_components(grid, color)
            if len(rects) < 2:
                continue

            # All rects must be same size
            sizes = [(r['r2'] - r['r1'], r['c2'] - r['c1']) for r in rects]
            if len(set(sizes)) != 1:
                continue

            h_span, w_span = sizes[0]
            if h_span < 2 or w_span < 2:
                continue

            # Each component must be exactly the bounding box perimeter
            all_valid = True
            for rect in rects:
                r1, r2, c1, c2 = rect['r1'], rect['r2'], rect['c1'], rect['c2']
                perim = set()
                for r in range(r1, r2 + 1):
                    perim.add((r, c1)); perim.add((r, c2))
                for c in range(c1, c2 + 1):
                    perim.add((r1, c)); perim.add((r2, c))
                if rect['comp'] != perim:
                    all_valid = False
                    break

            if not all_valid:
                continue

            border_color = color
            best_rects = rects
            break

        if border_color is None or best_rects is None:
            return None

        rects = best_rects
        h = rects[0]['r2'] - rects[0]['r1'] + 1
        w = rects[0]['c2'] - rects[0]['c1'] + 1

        # Build full h×w output: bg interior + border perimeter
        out = [[bg] * w for _ in range(h)]
        for r in range(h):
            out[r][0] = border_color
            out[r][w - 1] = border_color
        for c in range(w):
            out[0][c] = border_color
            out[h - 1][c] = border_color

        # Overlay interior markers from all rects
        for rect in rects:
            r1, r2, c1, c2 = rect['r1'], rect['r2'], rect['c1'], rect['c2']
            for r in range(r1 + 1, r2):
                for c in range(c1 + 1, c2):
                    v = grid[r][c]
                    if v != bg and v != border_color:
                        out[r - r1][c - c1] = v

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 9. 1b59e163 — shapes move to align marker with isolated colored dots
#    REPLACES _solve_shape_stamp_to_marker
# ─────────────────────────────────────────────────────────────
def _solve_shape_move_to_dots(pairs, test_inp):
    """
    Color-1 shapes each have a unique interior marker color (non-bg, non-1, non-5).
    Isolated single dots of those marker colors appear outside shapes.
    Move each shape so its marker aligns with the matching isolated dot.
    Erase original shapes/markers/isolated-5-dots; output shows only moved shapes.
    Verified on task 1b59e163.
    """
    def _find_components(grid, bg):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        comps = []
        for sr in range(R):
            for sc in range(C):
                v = grid[sr][sc]
                if v != bg and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    color = v
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != color:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((r + dr, c + dc))
                    comps.append({'color': color, 'cells': comp})
        return comps

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])
        shape_color = 1

        comps = _find_components(grid, bg)

        # Find large shape components of color 1
        shape_comps = [c for c in comps if c['color'] == shape_color and len(c['cells']) > 1]
        if not shape_comps:
            return None

        # For each shape find its interior marker (non-bg, non-1, non-5)
        templates = []
        for sc in shape_comps:
            cells = sc['cells']
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            r1, r2 = min(rs), max(rs)
            c1, c2 = min(cs), max(cs)

            marker_r = marker_c = marker_color = None
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    v = grid[r][c]
                    if v != bg and v != shape_color and v != 5:
                        marker_r, marker_c, marker_color = r, c, v
                        break
                if marker_r is not None:
                    break

            if marker_r is None:
                continue

            templates.append({
                'cells': cells,
                'marker_r': marker_r, 'marker_c': marker_c,
                'marker_color': marker_color,
                'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2,
            })

        if not templates:
            return None

        marker_to_tmpl = {t['marker_color']: t for t in templates}

        # Find isolated dots matching marker colors outside template bounding boxes
        target_dots = {}
        for comp in comps:
            color = comp['color']
            if color not in marker_to_tmpl:
                continue
            if len(comp['cells']) != 1:
                continue
            dot_r, dot_c = comp['cells'][0]
            tmpl = marker_to_tmpl[color]
            # Must be outside template bounding box
            if not (tmpl['r1'] <= dot_r <= tmpl['r2'] and tmpl['c1'] <= dot_c <= tmpl['c2']):
                target_dots[color] = (dot_r, dot_c)

        if not target_dots:
            return None

        # Build output: all bg, stamp moved shapes (color 1 cells only, no markers)
        out = [[bg] * C for _ in range(R)]

        for color, (dot_r, dot_c) in target_dots.items():
            tmpl = marker_to_tmpl[color]
            offset_r = dot_r - tmpl['marker_r']
            offset_c = dot_c - tmpl['marker_c']

            for (r, c) in tmpl['cells']:
                nr, nc = r + offset_r, c + offset_c
                if 0 <= nr < R and 0 <= nc < C:
                    out[nr][nc] = shape_color

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 10. 20981f0e — center color-1 shapes within color-2 divider cells
# ─────────────────────────────────────────────────────────────
def _solve_center_shapes_in_cells(pairs, test_inp):
    """
    Color-2 divider rows and columns partition the grid into cells.
    Color-1 shapes are off-center in each cell.
    Center each shape within its cell (integer division).
    Verified on task 20981f0e.
    """
    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        # Find divider rows (any row containing color 2)
        div_rows = [r for r in range(R) if any(grid[r][c] == 2 for c in range(C))]
        div_cols = [c for c in range(C) if any(grid[r][c] == 2 for r in range(R))]

        if not div_rows or not div_cols:
            return None

        # Cell boundaries: sentinels at -1 and R/C
        row_bounds = [-1] + div_rows + [R]
        col_bounds = [-1] + div_cols + [C]

        out = _copy_grid(grid)

        for i in range(len(row_bounds) - 1):
            for j in range(len(col_bounds) - 1):
                cell_r1 = row_bounds[i] + 1
                cell_r2 = row_bounds[i + 1] - 1
                cell_c1 = col_bounds[j] + 1
                cell_c2 = col_bounds[j + 1] - 1

                if cell_r1 > cell_r2 or cell_c1 > cell_c2:
                    continue

                cell_h = cell_r2 - cell_r1 + 1
                cell_w = cell_c2 - cell_c1 + 1

                # Find color-1 cells in this cell
                shape_cells = [
                    (r, c) for r in range(cell_r1, cell_r2 + 1)
                    for c in range(cell_c1, cell_c2 + 1)
                    if grid[r][c] == 1
                ]

                if not shape_cells:
                    continue

                sr1 = min(r for r, c in shape_cells)
                sr2 = max(r for r, c in shape_cells)
                sc1 = min(c for r, c in shape_cells)
                sc2 = max(c for r, c in shape_cells)

                sh = sr2 - sr1 + 1
                sw = sc2 - sc1 + 1

                # Centered top-left
                new_r1 = cell_r1 + (cell_h - sh) // 2
                new_c1 = cell_c1 + (cell_w - sw) // 2

                offset_r = new_r1 - sr1
                offset_c = new_c1 - sc1

                if offset_r == 0 and offset_c == 0:
                    continue

                # Erase original
                for (r, c) in shape_cells:
                    out[r][c] = bg

                # Place at centered position
                for (r, c) in shape_cells:
                    nr, nc = r + offset_r, c + offset_c
                    if cell_r1 <= nr <= cell_r2 and cell_c1 <= nc <= cell_c2:
                        out[nr][nc] = 1

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 11. 2601afb7 — sort columns by height of non-bg content (tallest leftmost)
# ─────────────────────────────────────────────────────────────
def _solve_sort_cols_by_height(pairs, test_inp):
    """
    Sort columns by height of non-background content.
    Tallest column goes to leftmost position.
    Verified on task 2601afb7.
    """
    def _col_height(grid, c, bg):
        for r in range(len(grid)):
            if grid[r][c] != bg:
                return len(grid) - r
        return 0

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        heights = [(_col_height(grid, c, bg), c) for c in range(C)]
        sorted_cols = [c for _, c in sorted(heights, key=lambda x: -x[0])]

        out = [[grid[r][sorted_cols[c]] for c in range(C)] for r in range(R)]
        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 12. 13713586 — fill toward 5-wall (full row or col of 5s), closer wins
# ─────────────────────────────────────────────────────────────
def _solve_forward_fill_toward_5_wall(pairs, test_inp):
    """
    5-wall is a full row or full column of 5s.
    Each non-bg, non-5 cell fills toward the wall along its row (wall=col) or col (wall=row).
    In overlap zones, cell closer to wall takes precedence (paint farthest→closest).
    Verified on task 13713586.
    """
    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])
        out = _copy_grid(grid)

        wall_row = None
        wall_col = None
        for r in range(R):
            if all(grid[r][c] == 5 for c in range(C)):
                wall_row = r
                break
        for c in range(C):
            if all(grid[r][c] == 5 for r in range(R)):
                wall_col = c
                break

        if wall_row is None and wall_col is None:
            return None

        if wall_row is not None:
            for c in range(C):
                colored_cells = []
                for r in range(R):
                    v = grid[r][c]
                    if v != bg and v != 5:
                        colored_cells.append((r, v))
                colored_cells.sort(key=lambda x: -abs(x[0] - wall_row))
                for (r, v) in colored_cells:
                    if r < wall_row:
                        step_range = range(r, wall_row)
                    else:
                        step_range = range(r, wall_row, -1)
                    for rr in step_range:
                        out[rr][c] = v

        if wall_col is not None:
            for r in range(R):
                colored_cells = []
                for c in range(C):
                    v = grid[r][c]
                    if v != bg and v != 5:
                        colored_cells.append((c, v))
                colored_cells.sort(key=lambda x: -abs(x[0] - wall_col))
                for (c, v) in colored_cells:
                    if c < wall_col:
                        step_range = range(c, wall_col)
                    else:
                        step_range = range(c, wall_col, -1)
                    for cc in step_range:
                        out[r][cc] = v

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 13. 1caeab9d — all blocks slide to row range of color-1 block
# ─────────────────────────────────────────────────────────────
def _solve_gravity_color1_anchor(pairs, test_inp):
    """
    All rectangular blocks (non-bg) slide to share the row range of the block
    whose pixel has color value 1. Column positions are preserved.
    Verified on task 1caeab9d.
    """
    def _find_rect_blocks(grid, bg):
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        blocks = []
        for sr in range(R):
            for sc in range(C):
                v = grid[sr][sc]
                if v != bg and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= R or c < 0 or c >= C:
                            continue
                        if visited[r][c] or grid[r][c] != v:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((r + dr, c + dc))
                    rs2 = [r for r, c in comp]
                    cs2 = [c for r, c in comp]
                    blocks.append({
                        'color': v,
                        'cells': comp,
                        'r1': min(rs2), 'r2': max(rs2),
                        'c1': min(cs2), 'c2': max(cs2),
                    })
        return blocks

    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        blocks = _find_rect_blocks(grid, bg)
        if not blocks:
            return None

        anchor = None
        for b in blocks:
            if b['color'] == 1:
                anchor = b
                break

        if anchor is None:
            return None

        anchor_r1 = anchor['r1']

        out = [[bg] * C for _ in range(R)]

        for b in blocks:
            dr = anchor_r1 - b['r1']
            for (r, c) in b['cells']:
                nr = r + dr
                if 0 <= nr < R:
                    out[nr][c] = b['color']

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)


# ─────────────────────────────────────────────────────────────
# 14. 29700607 — header row maps colors → cols; dots extend partial row + col
#     FIXED: partial row fill (min–max), col fill to dot_row inclusive,
#            no-dot columns fill to R-1
# ─────────────────────────────────────────────────────────────
def _solve_header_row_col_expand(pairs, test_inp):
    """
    Header row (first row with ≥3 non-bg cells) maps each color to a column.
    For each lone dot (row with exactly 1 non-bg cell):
      - Fill that row from min(dot_c, mapped_c) to max(dot_c, mapped_c).
      - Fill mapped column from header+1 to dot_row (inclusive).
    For header colors with NO dot: fill mapped column from header+1 to R-1.
    Verified on task 29700607.
    """
    def _apply(grid):
        bg = _get_bg(grid)
        R, C = len(grid), len(grid[0])

        # Header row: first row with ≥3 non-bg cells
        header_row = None
        color_to_col = {}
        for r in range(R):
            non_bg = [(c, grid[r][c]) for c in range(C) if grid[r][c] != bg]
            if len(non_bg) >= 3:
                header_row = r
                color_to_col = {v: c for c, v in non_bg}
                break

        if header_row is None:
            return None

        out = _copy_grid(grid)
        dots_found = set()

        for r in range(R):
            if r == header_row:
                continue
            non_bg_in_row = [(c, grid[r][c]) for c in range(C) if grid[r][c] != bg]
            if len(non_bg_in_row) == 1:
                dot_c, dot_color = non_bg_in_row[0]
                if dot_color not in color_to_col:
                    continue
                mapped_c = color_to_col[dot_color]
                dots_found.add(dot_color)

                # Partial row fill: min to max of dot_c and mapped_c (inclusive)
                c_lo = min(dot_c, mapped_c)
                c_hi = max(dot_c, mapped_c)
                for c in range(c_lo, c_hi + 1):
                    out[r][c] = dot_color

                # Column fill: mapped_c from header+1 to dot_row inclusive
                for rr in range(header_row + 1, r + 1):
                    out[rr][mapped_c] = dot_color

        # Header colors with NO dot: fill mapped column from header+1 to R-1
        for color, mapped_c in color_to_col.items():
            if color not in dots_found:
                for rr in range(header_row + 1, R):
                    out[rr][mapped_c] = color

        return out

    for p in pairs:
        if _apply(p['input']) != p['output']:
            return None

    return _apply(test_inp)




_T1_SOLVERS = [
    # --- Exact / trivial ---
    _solve_identity,
    _solve_constant_output,

    # --- Geometric transforms ---
    _solve_direct_transform,

    # --- Precise signal-processing crops ---
    _solve_remove_border,
    _solve_add_border,
    _solve_select_half,
    _solve_select_quarter,
    _solve_crop_background,

    # --- Precise color operations ---
    _solve_color_remap,
    _solve_color_freq_map,
    _solve_ca_center_only,
    _solve_color_swap,
    _solve_color_invert,
    _solve_color_complement_bg,
    _solve_color_isolate,

    # --- Scale / tiling ---
    _solve_scale,
    _solve_tiling,
    _solve_stamp_upscale,
    _solve_row_col_repeat,

    # --- Stride / subgrid extraction ---
    _solve_stride_crop,
    _solve_extract_subgrid_offset,

    # --- Physics ---
    _solve_gravity,
    _solve_gravity_nonzero,

    # --- BFS / fill ---
    _solve_fill_enclosed,
    _solve_flood_fill,
    _solve_border_fill,

    # --- Computer vision ---
    _solve_largest_component,
    _solve_smallest_component,
    _solve_propagate_nonbg,
    _solve_dilation,
    _solve_erosion,
    _solve_majority_fill,

    # --- Cellular automata (LOCAL_CHANGE tasks) ---
    _solve_ca_4d,
    _solve_ca_4s,
    _solve_ca_8d,
    _solve_ca_8s,
    _solve_ca_4d_alt_bv,

    # --- Crystal growth / symmetry ---
    _solve_sym_h,
    _solve_sym_v,
    _solve_sym_rot180,
    _solve_sym_diagonal,

    # --- Pattern / structure ---
    _solve_color_parity,
    _solve_object_translate,

    # --- Chain / compound ---
    _solve_chain_geo_remap,
    _solve_chain_remap_geo,
    _solve_scale_then_remap,

    # ── NEW: diagnostically derived targeted solvers ─────────
    # Fractal / self-stamp
    _solve_fractal_stamp_self,

    # Separator-based overlay
    _solve_overlay_sep_col,
    _solve_overlay_sep_row,

    # Separator AND-halves
    _solve_and_halves_sep_col,

    # Row/col connect (from dots / endpoint pairs)
    _solve_fill_row_endpoints,
    _solve_connect_row,
    _solve_connect_col,
    _solve_connect_row_col,
    _solve_color_columns_full,

    # Fill between exact pair
    _solve_fill_pair_col,

    # Enclosed fill with new color
    _solve_enclosed_new_color,

    # Separator-divided grid → extract content sub-cell
    _solve_content_subgrid_sep,

    # Overlay scattered same-sized objects
    _solve_overlay_objects,

    # Output separator line colors
    _solve_sep_line_colors,

    # Crop to content bounding box
    _solve_crop_to_content,

    # Pattern sequence continuation across separator sections
    _solve_pattern_sequence_sep,
    # ── END NEW ──────────────────────────────────────────────

    # --- NEW-A/B/C: additional targeted fixes ----------------
    _solve_segment_fill_col,      # 17b80ad2: territory fill by column dots
    _solve_fill_2x2_blocks,       # 31adaf00: 2×2 bg block fill
    _solve_connect_endpoints,     # 253bf280: row+col endpoint pair fill

    # --- Batch-2 targeted solvers ---
    _solve_diagonal_stripe,          # 05269061: diagonal stripe
    _solve_fractal_stamp_complement, # 0692e18c: fractal stamp complement
    _solve_stamp_max_freq,           # 27f8ce4f: stamp at max-freq positions
    _solve_stamp_uniform_axis,       # 15696249: stamp along uniform axis
    _solve_mirror_mosaic_2x2,        # 0c786b71: 2×2 mirror mosaic

    # --- Batch-3 targeted solvers ---
    _solve_cross_draw_markers,       # 140c817e: cross draw + diag neighbors
    _solve_voronoi_row_frame,        # 0f63c0b9: row-Voronoi frame
    _solve_recolor_8_by_key_pattern, # 009d5c81: key-pattern→color, replace 8s
    _solve_grid_and_intersection,    # 0520fde7: split AND → 2
    _solve_grid_nor,                 # 1b2d62fb: split NOR → 8

    # --- Row-pattern lookup (last: high recall, moderate precision) ---
    _solve_recolor_small_components,     # 12eac192 + 1e5d6875
    _solve_most_frequent_below_sep,      # 27a77e38
    _solve_template_color_markers,       # 12997ef3
    _solve_replace_ones_by_matching_shape, # 2a5f8217
    _solve_cross_diagonal_markers,       # 0ca9ddb6
    _solve_rotate_column_colors,         # 2601afb7
    _solve_four_section_priority,        # 281123b4
    _solve_xor_halves,                  # 34b99a2b
    _solve_block_connectivity,          # 239be575
    _solve_or_halves_1,                 # 195ba7dc
    _solve_sparse_to_blocks,            # 33067df9
    _solve_diagonal_line_draw,          # 1f876c06
    _solve_color_swap_key,              # 0becf7df
    _solve_shape_lookup_3x3,            # 27a28665
    _solve_row_complement_noise,         # 2de01db2
    _solve_3section_stack,               # 25c199f5
    _solve_shape_to_color_3x3_sections,  # 17cae0c1
    _solve_uniform_row_encode,       # 25d8a9c8: uniform-row→5, mixed→0
    _solve_count_2x2_ones_blocks,    # 1fad071e: count 2×2-all-1 blocks→1x5
    _solve_trie_rows,

    # ── Tile/grow pattern solvers ────────────────────────────────────────
    _solve_tile_3x3_with_marker,     # 310f3251: tile 3x3 with diagonal marker
    _solve_tile_3x3_alt_hflip,       # 00576224: tile 3x3 alternating hflip
    _solve_extend_period_to_9rows,   # 017c7c7b: extend period rows to 9 with remap
    _solve_nearest_wall_color,         # 2204b7a8: replace dots with nearest wall color
    _solve_largest_blob_to_3x3,        # 3194b014: largest non-noise blob -> 3x3 uniform
    _solve_xor_halves,                 # 31d5ba1a, 3428a4f5: XOR top/bottom halves
    _solve_nor_halves,                 # 0c9aba6e: NOR top/bottom halves with divider
    _solve_tile_repeat_third,          # 2dee498d: output = first W//3 cols
    _solve_bbox_tile_horizontal,       # 28bf18c6: bbox of signal tiled 2x horizontally
    _solve_minority_bbox_fill,         # 23b5c85d: least-freq color bbox → uniform fill
    _solve_grid_divider_sections,      # 1190e5a7: divider lines → section count output
    _solve_staircase_blob_count,       # 2753e76c: blob count → staircase
    _solve_quadrant_blobs,             # 19bb5feb: dominant-color quadrant → 2x2 output
    _solve_minority_bbox_extract,      # 0b148d64: 2-color input, extract signal bbox crop
    _solve_checkerboard_grid_on_empty, # 332efdb3: all-zero input → grid pattern
    _solve_spiral_on_empty,            # 28e73c20: all-zero input → clockwise spiral
    # --- Batch-4 targeted solvers ---
    _solve_color_shift_signal,         # 342dd610: per-color fixed shift vector
    _solve_shift_left_fill5,           # 32e9702f: shift-left-1, 0->5 fill
    _solve_perimeter_tile,             # 30f42897: tile N-cell signal with period 2N on perimeter
    _solve_complement_blocks,          # 1c0d0a4b: divider rows/cols -> 0; else 0->2, 8->0
    _solve_row_col_from_single_dots,
    _solve_two_dot_cross_rect_fill,
    _solve_closed_rect_corner_marks,
    _solve_open_rect_fill_exterior,
    _solve_arith_seq_5_separator,
    _solve_select_most_interior_nulls_region,
    _solve_extract_rect_interior,
    _solve_overlay_rect_interiors,
    _solve_shape_move_to_dots,
    _solve_center_shapes_in_cells,
    _solve_sort_cols_by_height,
    _solve_forward_fill_toward_5_wall,
    _solve_gravity_color1_anchor,
    _solve_header_row_col_expand,
]


def _run_solvers_on_task(task_data):
    """
    Run all T1 solvers on every test input of a task.
    Returns list of (attempt_1, attempt_2) per test index,
    or None if no solver fired.
    """
    train_pairs = task_data.get("train", [])
    test_inputs = task_data.get("test", [])

    pairs = []
    for ex in train_pairs:
        inp = ex.get("input")
        out = ex.get("output")
        if inp is not None and out is not None:
            pairs.append({"input": inp, "output": out})

    if not pairs:
        return None

    results = []  # per test index: (attempt_1, attempt_2)
    for test_ex in test_inputs:
        test_inp = test_ex.get("input") if isinstance(test_ex, dict) \
                   else test_ex
        if test_inp is None:
            return None  # can't solve if test input missing

        preds = []
        for solver in _T1_SOLVERS:
            try:
                pred = solver(pairs, test_inp)
                if _valid_grid(pred) and pred not in preds:
                    preds.append(pred)
                if len(preds) == 2:
                    break
            except Exception:
                pass

        if not preds:
            return None  # at least one test unsolvable → skip whole task
        results.append(preds)

    return results


# ════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════
def run_dsa_prepass(test_path, out_path):
    """
    Load all tasks from test_path (JSON: {task_id: task_data}).
    Run T1 solvers on every task.

    Saves results to out_path in OFFICIAL submission format:
      {
        task_id: [
          {"attempt_1": grid, "attempt_2": grid},  # test[0]
          {"attempt_1": grid, "attempt_2": grid},  # test[1] (if present)
          ...
        ],
        ...
      }

    Rules:
    - ALL task_ids from test_path are written (no missing keys).
    - Unsolved test inputs fall back to a copy of the test input itself.
    - Both attempt_1 and attempt_2 are always present.
    - Multiple test inputs per task are handled in order.

    Returns set of task IDs where at least one solver fired on test[0].
    """
    import copy as _copy
    print(f"[DSA-prepass] Loading tasks from {test_path}")
    with open(test_path) as f:
        data = json.load(f)

    submission  = {}
    solved_ids  = set()
    total       = len(data)

    for task_id, task_data in data.items():
        tests = task_data.get("test", [])
        if not tests:
            submission[task_id] = []
            continue

        try:
            results = _run_solvers_on_task(task_data)  # list[list[grid]] or None
        except Exception as e:
            print(f"[DSA-prepass]   ERROR on {task_id}: {e}")
            results = None

        answers = []
        for ti, test_item in enumerate(tests):
            fallback = _copy.deepcopy(test_item["input"])
            if results is not None and ti < len(results):
                preds = results[ti]           # up to 2 grids
                a1 = preds[0] if preds else fallback
                a2 = preds[1] if len(preds) > 1 else a1
                answers.append({"attempt_1": a1, "attempt_2": a2})
                if ti == 0:
                    solved_ids.add(task_id)
            else:
                answers.append({"attempt_1": fallback, "attempt_2": fallback})

        submission[task_id] = answers
        if task_id in solved_ids:
            print(f"[DSA-prepass]   SOLVED {task_id}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f)

    n_solved = len(solved_ids)
    print(f"[DSA-prepass] Done: {n_solved}/{total} tasks solved → {out_path}")
    return solved_ids


if __name__ == "__main__":
    # Quick smoke-test on local training tasks
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
           "/home/sandbox/arc2_combined_train.json"
    solved = run_dsa_prepass(path, "/tmp/dsa_prepass_test.json")
    print(f"Solved: {len(solved)} tasks")
    if solved:
        sample = next(iter(solved))
        with open("/tmp/dsa_prepass_test.json") as f:
            out = json.load(f)
        print(f"Sample {sample}: attempt_1 shape={len(out[sample]['attempt_1'])}x{len(out[sample]['attempt_1'][0])}")


