"""
Wave 39 Solvers — 7 confirmed high-confidence patterns
Tasks: 4258a5f9, 3906de3d, 3bdb4ada, 3c9b0459, 3bd67248, 3ac3eb23, 3d6c6e23
"""

from collections import Counter

# ─── helpers ────────────────────────────────────────────────────────────────

def _grid(g):
    return [list(r) for r in g]

# ────────────────────────────────────────────────────────────────────────────
# Task 4258a5f9
# Pattern: for each 5 at (r,c), set all 8 neighbors to 1, keep 5 at center
# ────────────────────────────────────────────────────────────────────────────

def solve_4258a5f9(input_grid):
    g = _grid(input_grid)
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    for r in range(H):
        for c in range(W):
            if g[r][c] == 5:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not (dr == 0 and dc == 0):
                            out[nr][nc] = 1
    return out

# ────────────────────────────────────────────────────────────────────────────
# Task 3906de3d
# Pattern: column-wise 2s pack — count n=2s in column, find last row with 1
#          in that column (r_top), place n 2s at rows r_top+1..r_top+n,
#          zero original 2s
# ────────────────────────────────────────────────────────────────────────────

def solve_3906de3d(input_grid):
    g = _grid(input_grid)
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    for c in range(W):
        col = [g[r][c] for r in range(H)]
        n_twos = col.count(2)
        if n_twos == 0:
            continue
        r_top = -1
        for r in range(H - 1, -1, -1):
            if col[r] == 1:
                r_top = r
                break
        if r_top == -1:
            continue
        for r in range(H):
            if out[r][c] == 2:
                out[r][c] = 0
        for i in range(n_twos):
            nr = r_top + 1 + i
            if 0 <= nr < H:
                out[nr][c] = 2
    return out

# ────────────────────────────────────────────────────────────────────────────
# Task 3bdb4ada
# Pattern: 3-row thick solid block; middle row gets alternating color/0 from block start
# ────────────────────────────────────────────────────────────────────────────

def solve_3bdb4ada(input_grid):
    g = _grid(input_grid)
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    r = 0
    while r <= H - 3:
        cols_r = [c for c in range(W) if g[r][c] != 0]
        if not cols_r:
            r += 1
            continue
        color = g[r][cols_r[0]]
        c0, c1 = cols_r[0], cols_r[-1]
        valid = (all(g[r][c] == color for c in range(c0, c1 + 1)) and
                 all(g[r][c] == 0 for c in range(0, c0)) and
                 all(g[r][c] == 0 for c in range(c1 + 1, W)))
        if valid:
            for rr in [r + 1, r + 2]:
                row_cols = [c for c in range(W) if g[rr][c] != 0]
                if not row_cols or row_cols[0] != c0 or row_cols[-1] != c1:
                    valid = False; break
                if not all(g[rr][c] == color for c in range(c0, c1 + 1)):
                    valid = False; break
        if not valid:
            r += 1
            continue
        mid = r + 1
        for c in range(c0, c1 + 1):
            out[mid][c] = color if (c - c0) % 2 == 0 else 0
        r += 3
    return out

# ────────────────────────────────────────────────────────────────────────────
# Task 3c9b0459
# Pattern: rotate grid 180°
# ────────────────────────────────────────────────────────────────────────────

def solve_3c9b0459(input_grid):
    g = _grid(input_grid)
    return [row[::-1] for row in g[::-1]]

# ────────────────────────────────────────────────────────────────────────────
# Task 3bd67248
# Pattern: anti-diagonal of 2 at (r, W-1-r) for r in 0..H-2;
#          bottom row cols 1..W-1 all set to 4
# ────────────────────────────────────────────────────────────────────────────

def solve_3bd67248(input_grid):
    g = _grid(input_grid)
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    for r in range(H - 1):
        c = W - 1 - r
        if 0 <= c < W:
            out[r][c] = 2
    for c in range(1, W):
        out[H - 1][c] = 4
    return out

# ────────────────────────────────────────────────────────────────────────────
# Task 3ac3eb23
# Pattern: zigzag downward from each non-zero seed
# ────────────────────────────────────────────────────────────────────────────

def solve_3ac3eb23(input_grid):
    g = _grid(input_grid)
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    seeds = [(r, c, g[r][c]) for r in range(H) for c in range(W) if g[r][c] != 0]
    for (r0, c0, v) in seeds:
        for rr in range(r0 + 1, H):
            delta = rr - r0
            if delta % 2 == 0:
                out[rr][c0] = v
            else:
                if c0 - 1 >= 0: out[rr][c0 - 1] = v
                if c0 + 1 < W:  out[rr][c0 + 1] = v
    return out

# ────────────────────────────────────────────────────────────────────────────
# Task 3d6c6e23
# Pattern: pyramid at bottom, centered at c_center.
#   - Count cells per non-zero color.
#   - Single color (N cells): pyramid rows widths [1,3,5,...] summing to N, all same color.
#   - Multi-color: each color's count = its pyramid row width; sorted ascending top→bottom.
# ────────────────────────────────────────────────────────────────────────────

def _odd_widths_sum_to(N):
    """Return [1,3,5,...] whose sum == N."""
    widths, w, total = [], 1, 0
    while total + w <= N:
        widths.append(w)
        total += w
        w += 2
    return widths

def solve_3d6c6e23(input_grid):
    g = _grid(input_grid)
    H, W = len(g), len(g[0])
    out = [[0] * W for _ in range(H)]

    color_counts = Counter(g[r][c] for r in range(H) for c in range(W) if g[r][c] != 0)
    if not color_counts:
        return out

    # c_center: most common column of non-zero cells
    col_cnt = Counter(c for r in range(H) for c in range(W) if g[r][c] != 0)
    c_center = col_cnt.most_common(1)[0][0]

    colors_sorted = sorted(color_counts.items(), key=lambda x: x[1])  # (color, count)

    if len(colors_sorted) == 1:
        # Single color: expand total into pyramid rows [1,3,5,...]
        color = colors_sorted[0][0]
        N = colors_sorted[0][1]
        widths = _odd_widths_sum_to(N)
        layers = [(color, w) for w in widths]
    else:
        # Multi-color: each color's count = its pyramid row width
        layers = [(color, count) for color, count in colors_sorted]

    k = len(layers)
    for i, (color, width) in enumerate(layers):
        row_idx = H - k + i
        half = width // 2
        for dc in range(-half, half + 1):
            nc = c_center + dc
            if 0 <= nc < W:
                out[row_idx][nc] = color

    return out


# ────────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────────

WAVE39_SOLVERS = {
    "4258a5f9": solve_4258a5f9,
    "3906de3d": solve_3906de3d,
    "3bdb4ada": solve_3bdb4ada,
    "3c9b0459": solve_3c9b0459,
    "3bd67248": solve_3bd67248,
    "3ac3eb23": solve_3ac3eb23,
    "3d6c6e23": solve_3d6c6e23,
}
