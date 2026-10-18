"""
Wave 37 solvers — 13 tasks:
05a7bcf2, 1190bc91, 15113be4, 15660dd6, 17b866bd,
1b8318e3, 1c02dbbe, 212895b5, 25094a63, 256b0a75,
264363fd, 305b1341, 34cfa167
"""

from collections import deque, Counter


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 25094a63 – Find solid monochromatic rectangles >=5x5, replace all cells with 4
# ═══════════════════════════════════════════════════════════════════════════════
def solve_25094a63(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]
    to_replace = [[False]*W for _ in range(H)]
    colors_present = set(grid[r][c] for r in range(H) for c in range(W) if grid[r][c] != 0)
    for v in colors_present:
        height = [0] * W
        for r in range(H):
            for c in range(W):
                height[c] = (height[c] + 1) if grid[r][c] == v else 0
            for c1 in range(W):
                if height[c1] < 5:
                    continue
                min_h = height[c1]
                for c2 in range(c1, W):
                    if height[c2] < 5:
                        break
                    min_h = min(min_h, height[c2])
                    if (c2 - c1 + 1) >= 5 and min_h >= 5:
                        for dr in range(min_h):
                            for dc in range(c2 - c1 + 1):
                                to_replace[r - min_h + 1 + dr][c1 + dc] = True
    for r in range(H):
        for c in range(W):
            if to_replace[r][c]:
                out[r][c] = 4
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 1b8318e3 – Snap markers to nearest 5-block anchor cell
#    2×2 blocks of 5s are anchors. Single-cell colored markers (non-0,non-1,non-5)
#    snap to the closest 8-adjacent cell of the nearest 5-block.
# ═══════════════════════════════════════════════════════════════════════════════
def solve_1b8318e3(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    five_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 5]
    five_set = set(five_cells)

    markers = []
    for r in range(H):
        for c in range(W):
            v = int(grid[r][c])
            if v != 0 and v != 5:
                markers.append((r, c, v))

    if not five_cells or not markers:
        return out

    # Clear markers from output first
    for mr, mc, mv in markers:
        out[mr][mc] = 0

    def manhattan(r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2)

    # Build connected clusters of 5-cells (4-connected)
    unvisited = set(five_cells)
    clusters = []
    while unvisited:
        start = next(iter(unvisited))
        cluster = set()
        stack = [start]
        while stack:
            cell = stack.pop()
            if cell in cluster:
                continue
            cluster.add(cell)
            unvisited.discard(cell)
            r, c = cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in unvisited:
                    stack.append(nb)
        clusters.append(sorted(cluster))

    # Sort markers by distance to their nearest 5-cell (ascending) for greedy priority
    def nearest5_dist(marker):
        mr, mc, _ = marker
        return min(manhattan(mr, mc, r5, c5) for r5, c5 in five_cells)

    markers_sorted = sorted(markers, key=nearest5_dist)

    occupied = set()  # cells already claimed by a placed marker

    for mr, mc, mv in markers_sorted:
        # Rank clusters by nearest-member distance
        def cluster_dist(ci):
            return min(manhattan(mr, mc, r5, c5) for r5, c5 in clusters[ci])

        ranked_clusters = sorted(range(len(clusters)), key=cluster_dist)

        placed = False
        for ci in ranked_clusters:
            # Use ONLY the nearest 5-cell in this cluster; if blocked, skip cluster
            nr5, nc5 = min(clusters[ci],
                           key=lambda rc: manhattan(mr, mc, rc[0], rc[1]))
            candidates = [
                (nr5 + dr, nc5 + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (0 <= nr5 + dr < H and 0 <= nc5 + dc < W
                    and (nr5 + dr, nc5 + dc) not in five_set)
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda rc: manhattan(mr, mc, rc[0], rc[1]))
            best_cr, best_cc = candidates[0]
            if (best_cr, best_cc) not in occupied:
                out[best_cr][best_cc] = mv
                occupied.add((best_cr, best_cc))
                placed = True
                break
            # Best candidate in this cluster occupied → skip entire cluster

        if not placed:
            # Last resort: any unoccupied candidate around any 5-cell
            for r5, c5 in sorted(five_cells,
                                  key=lambda rc: manhattan(mr, mc, rc[0], rc[1])):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        cr, cc = r5 + dr, c5 + dc
                        if (0 <= cr < H and 0 <= cc < W
                                and (cr, cc) not in five_set
                                and (cr, cc) not in occupied):
                            out[cr][cc] = mv
                            occupied.add((cr, cc))
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break

    return out



# ═══════════════════════════════════════════════════════════════════════════════
# 3. 1c02dbbe – Fill bbox intersection of color-group bbox and 5-block bbox
#    Find 5-block bounding box. For each non-0,non-5 color group,
#    compute bbox of ALL that color's cells. Fill intersection of that bbox
#    with the 5-block bounding box using that color.
# ═══════════════════════════════════════════════════════════════════════════════
def solve_1c02dbbe(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Find 5-block bounding box
    five_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 5]
    if not five_cells:
        return out
    r_min = min(r for r, c in five_cells)
    r_max = max(r for r, c in five_cells)
    c_min = min(c for r, c in five_cells)
    c_max = max(c for r, c in five_cells)

    # Collect non-0, non-5 color groups
    color_cells = {}
    for r in range(H):
        for c in range(W):
            v = int(grid[r][c])
            if v != 0 and v != 5:
                color_cells.setdefault(v, []).append((r, c))

    def mdist(r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2)

    for color, cells in color_cells.items():
        # Classify each marker
        corner = None
        row_split = None
        col_split = None

        for r, c in cells:
            if r < r_min:       # above
                col_split = c
            elif r > r_max:     # below
                col_split = c
            elif c < c_min:     # left
                row_split = r
            elif c > c_max:     # right
                row_split = r
            else:               # inside/on block
                corner = (r, c)

        if corner is None or row_split is None or col_split is None:
            continue

        # Determine quadrant by proximity to 5-block corners
        cr, cc = corner
        dists = {
            'TL': mdist(cr, cc, r_min, c_min),
            'TR': mdist(cr, cc, r_min, c_max),
            'BL': mdist(cr, cc, r_max, c_min),
            'BR': mdist(cr, cc, r_max, c_max),
        }
        quadrant = min(dists, key=dists.get)

        # Determine fill region
        if quadrant == 'TL':
            fill_r0, fill_r1 = r_min, row_split
            fill_c0, fill_c1 = c_min, col_split
        elif quadrant == 'TR':
            fill_r0, fill_r1 = r_min, row_split
            fill_c0, fill_c1 = col_split, c_max
        elif quadrant == 'BL':
            fill_r0, fill_r1 = row_split, r_max
            fill_c0, fill_c1 = c_min, col_split
        else:  # BR
            fill_r0, fill_r1 = row_split, r_max
            fill_c0, fill_c1 = col_split, c_max

        # Fill cells that are currently 5 within the sub-rectangle
        for r in range(fill_r0, fill_r1 + 1):
            for c in range(fill_c0, fill_c1 + 1):
                if out[r][c] == 5:
                    out[r][c] = color

        # Clear outside markers (above/below/left/right) to 0
        for mr, mc in cells:
            if mr < r_min or mr > r_max or mc < c_min or mc > c_max:
                out[mr][mc] = 0

    return out



# ═══════════════════════════════════════════════════════════════════════════════
# 4. 15113be4 – Template-stamp rule
#    Grid divided by all-4 rows/cols into bands. Within each global band,
#    local all-4 separators further divide into 3×3 sub-regions.
#    A 6×6 colored shape (non-0,non-1,non-4) encodes a 3×3 template via
#    2×2 super-pixels. Stamp that color at positions (in matching sub-regions)
#    where all template-active positions currently = 1.
# ═══════════════════════════════════════════════════════════════════════════════
def solve_15113be4(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Step 1: Find special color S (not 0, 1, or 4)
    S, S_pos = None, None
    for r in range(H):
        for c in range(W):
            v = int(grid[r][c])
            if v not in (0, 1, 4):
                S, S_pos = v, (r, c)
                break
        if S is not None:
            break
    if S is None:
        return out

    # Step 2: Scan from S_pos in 4 directions until hitting a 4 -> template bounds
    s_r, s_c = S_pos
    t_r0 = 0
    for r in range(s_r - 1, -1, -1):
        if int(grid[r][s_c]) == 4:
            t_r0 = r + 1; break
    t_r1 = H - 1
    for r in range(s_r + 1, H):
        if int(grid[r][s_c]) == 4:
            t_r1 = r - 1; break
    t_c0 = 0
    for c in range(s_c - 1, -1, -1):
        if int(grid[s_r][c]) == 4:
            t_c0 = c + 1; break
    t_c1 = W - 1
    for c in range(s_c + 1, W):
        if int(grid[s_r][c]) == 4:
            t_c1 = c - 1; break

    # Step 3: Derive 3x3 mask from 2x2 sub-blocks of the 6x6 template
    mask = [[0]*3 for _ in range(3)]
    for mi in range(3):
        for mj in range(3):
            br, bc = t_r0 + mi*2, t_c0 + mj*2
            for dr in range(2):
                for dc in range(2):
                    if br+dr <= t_r1 and bc+dc <= t_c1:
                        if int(grid[br+dr][bc+dc]) == S:
                            mask[mi][mj] = 1

    active = [(mi, mj) for mi in range(3) for mj in range(3) if mask[mi][mj]]
    if not active:
        return out

    # Step 4: For each 3x3 cell (period-4 structure: starts at 0,4,8,12,16,20)
    for rs in range(0, H, 4):
        for cs in range(0, W, 4):
            if rs + 2 >= H or cs + 2 >= W:
                continue
            # Skip cells that overlap the template region
            if max(rs, t_r0) <= min(rs+2, t_r1) and max(cs, t_c0) <= min(cs+2, t_c1):
                continue
            # If all active mask positions == 1, replace with S
            if all(int(grid[rs+mi][cs+mj]) == 1 for mi, mj in active):
                for mi, mj in active:
                    out[rs+mi][cs+mj] = S

    return out

def solve_17b866bd(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Collect markers at tile-junction positions (multiples of 5), any non-zero value
    markers = []
    for r in range(0, H, 5):
        for c in range(0, W, 5):
            v = grid[r][c]
            if v != 0:
                markers.append((r, c, v))

    for (mr, mc, v) in markers:
        # Fill tile to lower-right: rows mr+1..mr+4, cols mc+1..mc+4
        for dr in range(1, 5):
            for dc in range(1, 5):
                tr, tc = mr + dr, mc + dc
                if 0 <= tr < H and 0 <= tc < W:
                    if out[tr][tc] == 0:
                        out[tr][tc] = v
        # Clear marker cell
        out[mr][mc] = 0

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 212895b5 – 3×3 block of 8s; diagonal 2-trails from corners + 4-staircase arms
# ═══════════════════════════════════════════════════════════════════════════════
def solve_212895b5(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Find 3x3 block of 8s (TL corner)
    TLr, TLc = None, None
    for r in range(H - 2):
        for c in range(W - 2):
            if all(int(grid[r+dr][c+dc]) == 8 for dr in range(3) for dc in range(3)):
                TLr, TLc = r, c
                break
        if TLr is not None:
            break
    if TLr is None:
        return out

    BRr, BRc = TLr + 2, TLc + 2

    def safe(r, c):
        return 0 <= r < H and 0 <= c < W

    def val(r, c):
        return int(grid[r][c]) if safe(r, c) else -1

    # ── 2-TRAILS ──────────────────────────────────────────────────────────────
    # From each corner shoot diagonal ray outward, place 2.
    # Stop: (a) next cell=5, (b) BOTH flanks of step=5 (flanks of (pr,pc)→(cr,cc): (cr,pc) and (pr,cc)), (c) OOB
    for r0, c0, dr, dc in [(TLr, TLc, -1, -1), (TLr, BRc, -1, +1),
                            (BRr, TLc, +1, -1), (BRr, BRc, +1, +1)]:
        pr, pc = r0, c0
        cr, cc = r0 + dr, c0 + dc
        while safe(cr, cc):
            if val(cr, cc) == 5:
                break
            if val(cr, pc) == 5 and val(pr, cc) == 5:
                break
            out[cr][cc] = 2
            pr, pc = cr, cc
            cr += dr
            cc += dc

    # ── ARM A — UP-RIGHT ──────────────────────────────────────────────────────
    # k-odd: single at (TLr-k, TLc+k)
    # k-even: triple L-to-R at row TLr-k, cols TLc+(k-1)..TLc+(k+1)
    k = 1
    while True:
        if k % 2 == 1:
            r, c = TLr - k, TLc + k
            if not safe(r, c) or val(r, c) == 5:
                break
            out[r][c] = 4
        else:
            row = TLr - k
            lc, rc2 = TLc + (k - 1), TLc + (k + 1)
            if not safe(row, lc) or val(row, lc) == 5:
                break
            stop = False
            for c in range(lc, rc2 + 1):
                if not safe(row, c) or val(row, c) == 5:
                    stop = True; break
                out[row][c] = 4
            if stop:
                break
        k += 1

    # ── ARM B — DOWN-LEFT ─────────────────────────────────────────────────────
    # anchor starts TLc+1; k-odd: single at (BRr+k, anchor); k-even: triple R-to-L rightmost=anchor; anchor-=2
    anchor_col = TLc + 1
    k = 1
    while True:
        if k % 2 == 1:
            r, c = BRr + k, anchor_col
            if not safe(r, c) or val(r, c) == 5:
                break
            out[r][c] = 4
        else:
            row = BRr + k
            if not safe(row, anchor_col) or val(row, anchor_col) == 5:
                break
            stop = False
            for c in range(anchor_col, anchor_col - 3, -1):
                if not safe(row, c) or val(row, c) == 5:
                    stop = True; break
                out[row][c] = 4
            anchor_col -= 2
            if stop:
                break
        k += 1

    # ── ARM C — LEFT-UP ───────────────────────────────────────────────────────
    # L-cells: (TLr+1,TLc-1),(TLr+1,TLc-2),(TLr,TLc-2) — place in order, stop if =5 or OOB
    # If all placed: k-odd triple R-to-L rightmost=anchor (starts TLc-2);
    #                k-even single at (TLr-k, anchor-2); anchor-=2
    all_c = True
    for r, c in [(TLr+1, TLc-1), (TLr+1, TLc-2), (TLr, TLc-2)]:
        if not safe(r, c) or val(r, c) == 5:
            all_c = False; break
        out[r][c] = 4
    if all_c:
        anchor_col = TLc - 2
        k = 1
        while True:
            if k % 2 == 1:
                row = TLr - k
                if not safe(row, anchor_col) or val(row, anchor_col) == 5:
                    break
                stop = False
                for c in range(anchor_col, anchor_col - 3, -1):
                    if not safe(row, c) or val(row, c) == 5:
                        stop = True; break
                    out[row][c] = 4
                if stop:
                    break
            else:
                r, c = TLr - k, anchor_col - 2
                if not safe(r, c) or val(r, c) == 5:
                    break
                out[r][c] = 4
                anchor_col -= 2
            k += 1

    # ── ARM D — RIGHT-DOWN ────────────────────────────────────────────────────
    # L-cells: (TLr+1,BRc+1),(TLr+1,BRc+2),(TLr+2,BRc+2) — place in order, stop if =5 or OOB
    # If all placed: k-odd triple L-to-R leftmost=anchor (starts BRc+2);
    #                k-even single at (TLr+2+k, anchor+2); anchor+=2
    all_d = True
    for r, c in [(TLr+1, BRc+1), (TLr+1, BRc+2), (TLr+2, BRc+2)]:
        if not safe(r, c) or val(r, c) == 5:
            all_d = False; break
        out[r][c] = 4
    if all_d:
        anchor_col = BRc + 2
        k = 1
        while True:
            if k % 2 == 1:
                row = TLr + 2 + k
                if not safe(row, anchor_col) or val(row, anchor_col) == 5:
                    break
                stop = False
                for c in range(anchor_col, anchor_col + 3):
                    if not safe(row, c) or val(row, c) == 5:
                        stop = True; break
                    out[row][c] = 4
                if stop:
                    break
            else:
                r, c = TLr + 2 + k, anchor_col + 2
                if not safe(r, c) or val(r, c) == 5:
                    break
                out[r][c] = 4
                anchor_col += 2
            k += 1

    return out

# ═══════════════════════════════════════════════════════════════════════════════
# 7. 05a7bcf2 – Vertical/horizontal 8-barrier; 4-blobs and 2-blobs on opposite sides
#    30×30 grid. 4-blobs on one side, 2-blobs on other.
#    For each 4-blob: cells→color 3, gaps fill with 4, barrier extends with 8s
#    on the 2-side, 2-blob pushed to far wall boundary.
# ═══════════════════════════════════════════════════════════════════════════════
def solve_05a7bcf2(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Find full 8-wall: a row or column where EVERY cell is 8
    barrier_row = None
    barrier_col = None
    for r in range(H):
        if all(int(grid[r][c]) == 8 for c in range(W)):
            barrier_row = r
            break
    if barrier_row is None:
        for c in range(W):
            if all(int(grid[r][c]) == 8 for r in range(H)):
                barrier_col = c
                break

    if barrier_row is None and barrier_col is None:
        return out

    if barrier_col is not None:
        # Vertical wall: 4s are to the left, 2s are to the right
        for r in range(H):
            fours = [c for c in range(barrier_col) if int(grid[r][c]) == 4]
            if not fours:
                continue
            n2 = sum(1 for c in range(barrier_col+1, W) if int(grid[r][c]) == 2)
            max_4 = max(fours)
            for c in fours:
                out[r][c] = 3
            for c in range(max_4+1, barrier_col):
                out[r][c] = 4
            # barrier_col already 8; extend 8s to right
            for c in range(barrier_col+1, W-n2):
                out[r][c] = 8
            for c in range(W-n2, W):
                out[r][c] = 2
    else:
        # Horizontal wall: 4s are above, 2s are below
        for c in range(W):
            fours = [r for r in range(barrier_row) if int(grid[r][c]) == 4]
            if not fours:
                continue
            n2 = sum(1 for r in range(barrier_row+1, H) if int(grid[r][c]) == 2)
            max_4 = max(fours)
            for r in fours:
                out[r][c] = 3
            for r in range(max_4+1, barrier_row):
                out[r][c] = 4
            for r in range(barrier_row+1, H-n2):
                out[r][c] = 8
            for r in range(H-n2, H):
                out[r][c] = 2

    return out

def solve_1190bc91(grid):
    H, W = len(grid), len(grid[0])
    out = [[0]*W for _ in range(H)]

    # Step 1: Find all sequences (contiguous runs of >=2 non-zero cells in a row or col)
    sequences = []  # each is list of (r, c, value)

    for r in range(H):
        c = 0
        while c < W:
            if int(grid[r][c]) != 0:
                run = []
                while c < W and int(grid[r][c]) != 0:
                    run.append((r, c, int(grid[r][c])))
                    c += 1
                if len(run) >= 2:
                    sequences.append(run)
            else:
                c += 1

    for c in range(W):
        r = 0
        while r < H:
            if int(grid[r][c]) != 0:
                run = []
                while r < H and int(grid[r][c]) != 0:
                    run.append((r, c, int(grid[r][c])))
                    r += 1
                if len(run) >= 2:
                    sequences.append(run)
            else:
                r += 1

    if not sequences:
        return [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Main sequence = longest; secondaries = rest
    sequences.sort(key=lambda s: -len(s))
    main_seq = sequences[0]
    secondary_seqs = sequences[1:]

    # Step 2: Determine direction from s[0]->s[1]
    mr0, mc0, mv0 = main_seq[0]
    mr1, mc1, mv1 = main_seq[1]
    if mr1 > mr0:
        direction = 'down'
    elif mr1 < mr0:
        direction = 'up'
    elif mc1 > mc0:
        direction = 'right'
    else:
        direction = 'left'

    # Step 3: Fill s[1] full X-diagonals, then s[0] (higher priority, overwrites)
    for r in range(H):
        for c in range(W):
            if (r - c == mr1 - mc1) or (r + c == mr1 + mc1):
                out[r][c] = mv1
    for r in range(H):
        for c in range(W):
            if (r - c == mr0 - mc0) or (r + c == mr0 + mc0):
                out[r][c] = mv0

    # Step 4: Secondary flood-fills blocked by all 4 wall-diagonals of s[0] and s[1]
    wall_set = set()
    for r in range(H):
        for c in range(W):
            if ((r - c == mr0 - mc0) or (r + c == mr0 + mc0) or
                    (r - c == mr1 - mc1) or (r + c == mr1 + mc1)):
                wall_set.add((r, c))

    for sec_seq in secondary_seqs:
        sec_val = sec_seq[0][2]
        visited = set()
        q = deque()
        for r, c, v in sec_seq:
            if (r, c) not in wall_set and (r, c) not in visited:
                visited.add((r, c))
                q.append((r, c))
        while q:
            r, c = q.popleft()
            out[r][c] = sec_val
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and (nr, nc) not in wall_set:
                    visited.add((nr, nc))
                    q.append((nr, nc))

    # Step 5: s[k>=2] forward arms — only fill cells where out[r][c]==0
    for k in range(2, len(main_seq)):
        rk, ck, vk = main_seq[k]
        for r in range(H):
            for c in range(W):
                if out[r][c] != 0:
                    continue
                on_diag = (r - c == rk - ck) or (r + c == rk + ck)
                if not on_diag:
                    continue
                if direction == 'down' and r >= rk:
                    out[r][c] = vk
                elif direction == 'up' and r <= rk:
                    out[r][c] = vk
                elif direction == 'right' and c >= ck:
                    out[r][c] = vk
                elif direction == 'left' and c <= ck:
                    out[r][c] = vk

    # Step 6: remaining cells stay 0
    return out


def solve_15660dd6(grid):
    H, W = len(grid), len(grid[0])

    # Full 8-rows and 8-cols
    h8rows = sorted([r for r in range(H) if all(int(grid[r][c]) == 8 for c in range(W))])
    h8cols = sorted([c for c in range(W) if all(int(grid[r][c]) == 8 for r in range(H))])

    if not h8rows or not h8cols:
        return [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Row-sections: contiguous non-8-row spans
    def spans(seps, total):
        secs = []
        prev = -1
        for s in seps:
            if s - prev > 1:
                secs.append(list(range(prev + 1, s)))
            prev = s
        if prev < total - 1:
            secs.append(list(range(prev + 1, total)))
        return [s for s in secs if s]

    row_secs = spans(h8rows, H)

    # Col-sections: col 0 is label col; first = cols 1..h8cols[0]-1
    col_secs = []
    if h8cols[0] > 1:
        col_secs.append(list(range(1, h8cols[0])))
    for i in range(len(h8cols) - 1):
        s, e = h8cols[i] + 1, h8cols[i + 1]
        if e > s:
            col_secs.append(list(range(s, e)))
    if h8cols[-1] < W - 1:
        col_secs.append(list(range(h8cols[-1] + 1, W)))
    col_secs = [cs for cs in col_secs if cs]

    # Template row-section: all col-section cells contain only 1/2/8
    template_sec = None
    for rs in row_secs:
        ok = all(int(grid[r][c]) in (1, 2, 8)
                 for cs in col_secs for r in rs for c in cs)
        if ok:
            template_sec = rs
            break

    if template_sec is None:
        return [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # For each col-section find the row-section with a SOLID fill:
    # non-1, non-8 interior values = exactly one color ≠ 2
    out_blocks = []
    for cs in col_secs:
        fill_color = border_color = None
        for rs in row_secs:
            if rs is template_sec:
                continue
            vals = set()
            for r in rs:
                for c in cs:
                    v = int(grid[r][c])
                    if v != 1 and v != 8:
                        vals.add(v)
            if len(vals) == 1:
                fc = next(iter(vals))
                if fc != 2:                     # 2 is the template pattern colour
                    fill_color = fc
                    border_color = int(grid[rs[0]][0])  # label at col 0
                    break
        if fill_color is None or border_color is None:
            continue

        # Build block: template with 1→border_color, 2→fill_color
        block = []
        for r in template_sec:
            row_out = []
            for c in cs:
                v = int(grid[r][c])
                if v == 1:
                    row_out.append(border_color)
                elif v == 2:
                    row_out.append(fill_color)
                else:
                    row_out.append(v)
            block.append(row_out)
        out_blocks.append(block)

    if not out_blocks:
        return [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Stack blocks side-by-side with single 8-separator columns
    num_rows = len(out_blocks[0])
    result = []
    for r in range(num_rows):
        row = []
        for i, block in enumerate(out_blocks):
            if i > 0:
                row.append(8)
            row.extend(block[r])
        result.append(row)

    return result


def solve_256b0a75(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Step 1: Find connected components of non-zero values
    visited = [[False]*W for _ in range(H)]
    components = []
    for r in range(H):
        for c in range(W):
            v = int(grid[r][c])
            if v != 0 and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and not visited[nr][nc] and int(grid[nr][nc]) != 0:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append((comp, int(grid[comp[0][0]][comp[0][1]])))

    # Step 2: Corner L-shapes have size 3
    corner_comps = [(comp, color) for (comp, color) in components if len(comp) == 3]
    if len(corner_comps) != 4:
        return out

    # Step 3: Box bounds and colors
    all_cells = [(r, c) for comp, _ in corner_comps for r, c in comp]
    box_top    = min(r for r, c in all_cells)
    box_bottom = max(r for r, c in all_cells)
    box_left   = min(c for r, c in all_cells)
    box_right  = max(c for r, c in all_cells)

    color_counts = Counter(color for _, color in corner_comps)
    box_color  = color_counts.most_common(1)[0][0]
    fill_color = [c for c in color_counts if c != box_color][0]

    # Step 4: Fill box perimeter and interior
    for r in range(box_top, box_bottom+1):
        for c in range(box_left, box_right+1):
            if r == box_top or r == box_bottom or c == box_left or c == box_right:
                out[r][c] = box_color
            else:
                out[r][c] = fill_color

    # Step 5: Row scans within box row range (scan from box edge outward)
    for r in range(box_top, box_bottom+1):
        # Left segment: scan from col box_left-1 toward 0
        current = fill_color
        for c in range(box_left-1, -1, -1):
            v = int(grid[r][c])
            if v != 0 and v != fill_color:
                current = v
            out[r][c] = current
        # Right segment: scan from col box_right+1 toward W-1
        current = fill_color
        for c in range(box_right+1, W):
            v = int(grid[r][c])
            if v != 0 and v != fill_color:
                current = v
            out[r][c] = current

    # Step 6: Column scans within box col range (scan from box edge outward)
    for c in range(box_left, box_right+1):
        # Above segment: scan from row box_top-1 toward 0
        current = fill_color
        for r in range(box_top-1, -1, -1):
            v = int(grid[r][c])
            if v != 0 and v != fill_color:
                current = v
            out[r][c] = current
        # Below segment: scan from row box_bottom+1 toward H-1
        current = fill_color
        for r in range(box_bottom+1, H):
            v = int(grid[r][c])
            if v != 0 and v != fill_color:
                current = v
            out[r][c] = current

    return out


def solve_34cfa167(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    flat = [int(grid[r][c]) for r in range(H) for c in range(W)]
    bg = Counter(flat).most_common(1)[0][0]

    vis = [[False]*W for _ in range(H)]
    comps = []
    for r in range(H):
        for c in range(W):
            v = int(grid[r][c])
            if v != bg and not vis[r][c]:
                cells, q = [], deque([(r,c)])
                vis[r][c] = True
                while q:
                    cr,cc = q.popleft()
                    cells.append((cr,cc))
                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<H and 0<=nc<W and not vis[nr][nc] and int(grid[nr][nc])==v:
                            vis[nr][nc]=True
                            q.append((nr,nc))
                comps.append((cells, v))

    by_color = {}
    for cells,color in comps:
        by_color.setdefault(color,[]).append(cells)

    dec_comp = bare_comp = None
    for color, clist in by_color.items():
        if len(clist)==2 and len(clist[0])==len(clist[1]) and len(clist[0])>2:
            def has_adj(cells, col):
                cs = set(cells)
                for r,c in cells:
                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr,nc=r+dr,c+dc
                        if 0<=nr<H and 0<=nc<W and (nr,nc) not in cs:
                            nv=int(grid[nr][nc])
                            if nv!=bg and nv!=col: return True
                return False
            a0,a1 = has_adj(clist[0],color), has_adj(clist[1],color)
            if a0 or a1:
                dec_comp = clist[0] if a0 else clist[1]
                bare_comp = clist[1] if a0 else clist[0]
                break

    if dec_comp is None: return out

    rd_top  = min(r for r,c in dec_comp)
    rd_bot  = max(r for r,c in dec_comp)
    rd_left = min(c for r,c in dec_comp)
    rd_right= max(c for r,c in dec_comp)
    rb_top  = min(r for r,c in bare_comp)
    rb_bot  = max(r for r,c in bare_comp)
    rb_left = min(c for r,c in bare_comp)
    rb_right= max(c for r,c in bare_comp)

    dc_c=(rd_left+rd_right)/2.0; bc_c=(rb_left+rb_right)/2.0
    dc_r=(rd_top+rd_bot)/2.0;   bc_r=(rb_top+rb_bot)/2.0
    h_dir = 1 if bc_c>dc_c else -1
    v_dir = 1 if bc_r>dc_r else -1

    # Extract horizontal pattern from dec block edge going h_dir
    h_start = rd_right+1 if h_dir==1 else rd_left-1
    scan_rows = list(range(rd_top, rd_bot+1))
    h_raw=[]
    for step in range(max(W,H)*2):
        c = h_start + step*h_dir
        if c<0 or c>=W: break
        val=bg
        for r in scan_rows:
            v2=int(grid[r][c])
            if v2!=bg: val=v2; break
        h_raw.append(val)
        if len(h_raw)>=2 and h_raw[-1]==bg and h_raw[-2]==bg: break
    while h_raw and h_raw[-1]==bg: h_raw.pop()

    # Extract vertical pattern from dec block edge going v_dir
    v_start = rd_bot+1 if v_dir==1 else rd_top-1
    scan_cols = list(range(rd_left, rd_right+1))
    v_raw=[]
    for step in range(max(W,H)*2):
        r = v_start + step*v_dir
        if r<0 or r>=H: break
        val=bg
        for c in scan_cols:
            v2=int(grid[r][c])
            if v2!=bg: val=v2; break
        v_raw.append(val)
        if len(v_raw)>=2 and v_raw[-1]==bg and v_raw[-2]==bg: break
    while v_raw and v_raw[-1]==bg: v_raw.pop()

    if not h_raw or not v_raw: return out

    h_period = len(h_raw)+1
    h_pat = h_raw + [bg]*(h_period-len(h_raw))
    v_period = len(v_raw)+1
    v_pat = v_raw + [bg]*(v_period-len(v_raw))
    border_h = h_pat[0]
    border_v = v_pat[0]

    if h_dir==1: h_cs,h_ce = rd_right+1, rb_left-1
    else:        h_cs,h_ce = rb_right+1, rd_left-1
    if v_dir==1: v_cs,v_ce = rd_bot+1, rb_top-1
    else:        v_cs,v_ce = rb_bot+1, rd_top-1
    h_lo,h_hi = min(h_cs,h_ce), max(h_cs,h_ce)
    v_lo,v_hi = min(v_cs,v_ce), max(v_cs,v_ce)

    # Horizontal connectors at dec rows and bare rows
    for rows in [range(rd_top,rd_bot+1), range(rb_top,rb_bot+1)]:
        for r in rows:
            for c in range(h_lo, h_hi+1):
                pos=abs(c-h_cs)
                out[r][c]=h_pat[pos%h_period]

    # Horizontal border rows
    for br in [min(rd_top,rb_top)-1, max(rd_bot,rb_bot)+1]:
        if 0<=br<H:
            for c in range(h_lo, h_hi+1):
                out[br][c]=border_h

    # Vertical connectors at dec cols and bare cols
    for cols in [range(rd_left,rd_right+1), range(rb_left,rb_right+1)]:
        for c in cols:
            for r in range(v_lo, v_hi+1):
                pos=abs(r-v_cs)
                out[r][c]=v_pat[pos%v_period]

    # Vertical border cols
    for bc in [min(rd_left,rb_left)-1, max(rd_right,rb_right)+1]:
        if 0<=bc<W:
            for r in range(v_lo, v_hi+1):
                out[r][bc]=border_v

    # Cross-block fills: block_color at (dec rows × bare cols) and (bare rows × dec cols)
    block_color = int(grid[dec_comp[0][0]][dec_comp[0][1]])
    for r in range(rd_top, rd_bot+1):
        for c in range(rb_left, rb_right+1):
            out[r][c] = block_color
    for r in range(rb_top, rb_bot+1):
        for c in range(rd_left, rd_right+1):
            out[r][c] = block_color

    return out



def solve_264363fd(grid):
    H, W = len(grid), len(grid[0])
    out = [[int(grid[r][c]) for c in range(W)] for r in range(H)]

    # Background = most common color
    flat = [int(grid[r][c]) for r in range(H) for c in range(W)]
    bg = Counter(flat).most_common(1)[0][0]

    # BFS to find connected components of non-background cells
    visited = [[False] * W for _ in range(H)]
    components = []
    for r in range(H):
        for c in range(W):
            if int(grid[r][c]) != bg and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and not visited[nr][nc]
                                and int(grid[nr][nc]) != bg):
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)

    # Classify: rectangle (fills bbox exactly) vs key template
    rectangles = []
    key_template = None
    for comp in components:
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bbox_size = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(comp) == bbox_size:
            rectangles.append({'cells': set(comp), 'r0': r0, 'r1': r1,
                                'c0': c0, 'c1': c1})
        else:
            key_template = comp

    if key_template is None or not rectangles:
        return out

    # Identify center_color: color of non-base cells inside rectangles
    center_color = None
    for rect in rectangles:
        colors_in = Counter(int(grid[r][c]) for r, c in rect['cells'])
        if len(colors_in) >= 2:
            base_c = colors_in.most_common(1)[0][0]
            for col, cnt in colors_in.items():
                if col != base_c:
                    center_color = col
                    break
            if center_color is not None:
                break

    if center_color is None:
        return out

    # Find key center = cell with center_color in key template
    key_center = None
    for r, c in key_template:
        if int(grid[r][c]) == center_color:
            key_center = (r, c)
            break

    if key_center is None:
        return out

    kcr, kcc = key_center

    # Parse cardinal arms from key center
    def parse_arm(dr, dc):
        cells = []
        r, c = kcr + dr, kcc + dc
        while 0 <= r < H and 0 <= c < W and int(grid[r][c]) != bg:
            cells.append(int(grid[r][c]))
            r += dr
            c += dc
        return cells

    arm_up    = parse_arm(-1, 0)
    arm_down  = parse_arm(+1, 0)
    arm_left  = parse_arm(0, -1)
    arm_right = parse_arm(0, +1)

    arm_len_vert  = len(arm_up)
    arm_color_vert = arm_up[0] if arm_up else None
    arm_len_horiz = len(arm_left)
    arm_color_horiz = arm_left[0] if arm_left else None

    # Diagonal color: any non-bg cell at Chebyshev-1 diagonals from center
    diag_color = None
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nc = kcr + dr, kcc + dc
        if 0 <= nr < H and 0 <= nc < W and int(grid[nr][nc]) != bg:
            diag_color = int(grid[nr][nc])
            break

    # For each rectangle, process markers (center_color cells)
    for rect in rectangles:
        r0, r1, c0, c1 = rect['r0'], rect['r1'], rect['c0'], rect['c1']
        markers_in_rect = [(r, c) for r, c in rect['cells']
                           if int(grid[r][c]) == center_color]

        for mr, mc in markers_in_rect:
            # Vertical arm fill
            if arm_color_vert is not None:
                if arm_len_vert >= 2:
                    for r in range(r0, r1 + 1):
                        if r != mr:
                            out[r][mc] = arm_color_vert
                else:  # arm_len_vert == 1
                    for delta in (-1, 1):
                        nr = mr + delta
                        if r0 <= nr <= r1:
                            out[nr][mc] = arm_color_vert

            # Horizontal arm fill
            if arm_color_horiz is not None:
                if arm_len_horiz >= 2:
                    for c in range(c0, c1 + 1):
                        if c != mc:
                            out[mr][c] = arm_color_horiz
                else:  # arm_len_horiz == 1
                    for delta in (-1, 1):
                        nc = mc + delta
                        if c0 <= nc <= c1:
                            out[mr][nc] = arm_color_horiz

            # Diagonal fills
            if diag_color is not None:
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    nr, nc = mr + dr, mc + dc
                    if r0 <= nr <= r1 and c0 <= nc <= c1:
                        out[nr][nc] = diag_color

    # Remove key template (set to background)
    for r, c in key_template:
        out[r][c] = bg

    return out



def solve_305b1341(grid):
    H, W = len(grid), len(grid[0])
    out = [[0] * W for _ in range(H)]

    # Read palette
    palette = []
    for r in range(H):
        dot_color = int(grid[r][0])
        if dot_color == 0:
            break
        frame_color = int(grid[r][1])
        palette.append((dot_color, frame_color))

    if not palette:
        return out

    # Process from last to first (lower index = higher priority → processed last)
    for idx in range(len(palette) - 1, -1, -1):
        dot_color, frame_color = palette[idx]

        # Find all dots at cols >= 2
        dots = [(r, c) for r in range(H) for c in range(2, W)
                if int(grid[r][c]) == dot_color]

        if not dots:
            continue

        r_min = min(r for r, c in dots)
        r_max = max(r for r, c in dots)
        c_min = min(c for r, c in dots)
        c_max = max(c for r, c in dots)

        # Fill output region [r_min-1..r_max+1] × [c_min-1..c_max+1]
        for r in range(max(0, r_min - 1), min(H, r_max + 2)):
            for c in range(max(0, c_min - 1), min(W, c_max + 2)):
                # Is this a dot position?
                if (r_min <= r <= r_max and c_min <= c <= c_max
                        and (r - r_min) % 2 == 0 and (c - c_min) % 2 == 0):
                    out[r][c] = dot_color
                else:
                    out[r][c] = frame_color

    # Palette rows: cols 0-1 are already 0 (initialized), cols 2+ may have dots
    # but palette row slots should be 0. Clear them explicitly.
    for r in range(len(palette)):
        out[r][0] = 0
        out[r][1] = 0

    return out

