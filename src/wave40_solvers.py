"""Wave 40 solvers — 4 tasks confirmed."""
import json, numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────
def _grid(g): return [list(r) for r in g]
def _eq(a, b): return a == b

# ═══════════════════════════════════════════════════════════════════════════
# Task 3f23242b  — bracket/arrow drawn around each pixel of color 3
# Pattern (for a 3 at row r, col c, grid H×W):
#   row r-2: 5s at cols c-2..c+2
#   row r-1: 2 at c-2, 5 at c, 2 at c+2
#   row r  : 2 at c-2, (original 3 stays), 2 at c+2
#   row r+1: 2 at c-2, 2 at c+2
#   row r+2: 8s at cols c-2..c+2; 2s at all other cols in that row
# ═══════════════════════════════════════════════════════════════════════════
def solve_3f23242b(inp):
    H, W = len(inp), len(inp[0])
    out = _grid(inp)
    for r in range(H):
        for c in range(W):
            if inp[r][c] == 3:
                # top bar of 5s
                if r-2 >= 0:
                    for dc in range(-2, 3):
                        if 0 <= c+dc < W:
                            out[r-2][c+dc] = 5
                # second row: 2-walls + center 5
                if r-1 >= 0:
                    if 0 <= c-2 < W: out[r-1][c-2] = 2
                    if 0 <= c+2 < W: out[r-1][c+2] = 2
                    if 0 <= c   < W: out[r-1][c]   = 5
                # middle row: 2-walls (keep 3)
                if 0 <= c-2 < W: out[r][c-2] = 2
                if 0 <= c+2 < W: out[r][c+2] = 2
                # row below: 2-walls only
                if r+1 < H:
                    if 0 <= c-2 < W: out[r+1][c-2] = 2
                    if 0 <= c+2 < W: out[r+1][c+2] = 2
                # bottom row: 8s in bracket, 2s outside
                if r+2 < H:
                    for dc in range(-2, 3):
                        if 0 <= c+dc < W:
                            out[r+2][c+dc] = 8
                    for cc in range(W):
                        if cc < c-2 or cc > c+2:
                            out[r+2][cc] = 2
    return out

# ═══════════════════════════════════════════════════════════════════════════
# Task 423a55dc  — diagonal left-shear: bottom row fixed, row r shifts left
#                 by (bottom_row - r). Out-of-bounds cells dropped.
# ═══════════════════════════════════════════════════════════════════════════
def solve_423a55dc(inp):
    H, W = len(inp), len(inp[0])
    out = [[0]*W for _ in range(H)]
    # find bounding bottom row of shape
    shape_rows = [r for r in range(H) if any(inp[r][c] != 0 for c in range(W))]
    if not shape_rows:
        return inp
    bottom_row = max(shape_rows)
    for r in range(H):
        shift = bottom_row - r
        for c in range(W):
            if inp[r][c] != 0:
                nc = c - shift
                if 0 <= nc < W:
                    out[r][nc] = inp[r][c]
    return out

# ═══════════════════════════════════════════════════════════════════════════
# Task 3de23699  — 4 corner markers define a box; extract interior shape,
#                 replace shape-color with corner-marker color.
# ═══════════════════════════════════════════════════════════════════════════
def solve_3de23699(inp):
    H, W = len(inp), len(inp[0])
    # find two distinct non-zero colors
    color_cells = {}
    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            if v: color_cells.setdefault(v, []).append((r,c))
    if len(color_cells) < 2:
        return inp
    # marker color = 4 cells; shape color = rest
    items = sorted(color_cells.items(), key=lambda x: len(x[1]))
    shape_color, shape_cells = items[-1]  # more cells = shape
    marker_color, marker_cells = items[0] # fewer (4) = corners
    if len(marker_cells) != 4:
        # try reverse
        shape_color, shape_cells = items[0]
        marker_color, marker_cells = items[-1]
    # bounding box of markers
    rs = [r for r,c in marker_cells]; cs = [c for r,c in marker_cells]
    r1,r2,c1,c2 = min(rs),max(rs),min(cs),max(cs)
    # interior
    ir1,ir2,ic1,ic2 = r1+1,r2-1,c1+1,c2-1
    oh,ow = ir2-ir1+1, ic2-ic1+1
    out = [[0]*ow for _ in range(oh)]
    for r in range(ir1, ir2+1):
        for c in range(ic1, ic2+1):
            if inp[r][c] == shape_color:
                out[r-ir1][c-ic1] = marker_color
    return out

# ═══════════════════════════════════════════════════════════════════════════
# Task 3ee1011a  — colored line segments in input; output = nested square
#                 rings. Largest segment → outermost NxN ring. Each inner
#                 ring decreases size by 2. Colors ranked by segment length.
# ═══════════════════════════════════════════════════════════════════════════
def solve_3ee1011a(inp):
    # collect segment colors + lengths
    color_len = {}
    for row in inp:
        for v in row:
            if v: color_len[v] = color_len.get(v, 0) + 1
    if not color_len:
        return inp
    # sort by length descending
    sorted_colors = sorted(color_len, key=lambda c: -color_len[c])
    N = color_len[sorted_colors[0]]  # outermost ring size
    out = [[0]*N for _ in range(N)]
    # fill rings from outside in
    for depth, color in enumerate(sorted_colors):
        lo, hi = depth, N-1-depth
        if lo > hi:
            break
        # top/bottom rows
        for c in range(lo, hi+1):
            out[lo][c] = color
            out[hi][c] = color
        # left/right cols
        for r in range(lo, hi+1):
            out[r][lo] = color
            out[r][hi] = color
    return out


# ─── verification ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    with open('/home/sandbox/arc-agi_test_challenges.json') as f:
        data = json.load(f)

    solvers = [
        ('3f23242b', solve_3f23242b),
        ('423a55dc', solve_423a55dc),
        ('3de23699', solve_3de23699),
        ('3ee1011a', solve_3ee1011a),
    ]

    total_pass = 0
    total_fail = 0
    for tid, fn in solvers:
        task = data[tid]
        passes = 0
        for i, ex in enumerate(task['train']):
            pred = fn(ex['input'])
            expected = ex['output']
            if pred == expected:
                passes += 1
            else:
                print(f"  FAIL {tid} ex{i}")
                # show diff
                for r in range(len(expected)):
                    for c in range(len(expected[r])):
                        ep = expected[r][c]
                        pp = pred[r][c] if r < len(pred) and c < len(pred[r]) else -1
                        if ep != pp:
                            print(f"    cell ({r},{c}): expected {ep}, got {pp}")
        n = len(task['train'])
        total_pass += passes; total_fail += n - passes
        print(f"{tid}: {passes}/{n} train pass")
    print(f"\nTOTAL: {total_pass}/{total_pass+total_fail}")
