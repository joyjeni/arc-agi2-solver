"""
Wave 38 Solvers — ARC-AGI-2
Tasks: 3aa6fb7a, 3af2c5a8, 3eda0437, 37ce87bb
All train-validated
"""

from collections import deque
from typing import List

Grid = List[List[int]]


# ── 3aa6fb7a: L-shape 8s (3 cells in 2×2 bbox) → fill missing corner with 1 ──
def solve_3aa6fb7a(grid: Grid) -> Grid:
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False] * W for _ in range(H)]

    def bfs(sr, sc):
        cells = []
        q = deque([(sr, sc)])
        visited[sr][sc] = True
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == 8:
                    visited[nr][nc] = True
                    q.append((nr, nc))
        return cells

    for r in range(H):
        for c in range(W):
            if grid[r][c] == 8 and not visited[r][c]:
                cells = bfs(r, c)
                if len(cells) == 3:
                    rows = [cr for cr, cc in cells]
                    cols = [cc for cr, cc in cells]
                    min_r, max_r = min(rows), max(rows)
                    min_c, max_c = min(cols), max(cols)
                    if max_r - min_r == 1 and max_c - min_c == 1:
                        # 2×2 bounding box, 3 occupied → find missing corner
                        occupied = set(cells)
                        for rr in [min_r, max_r]:
                            for cc in [min_c, max_c]:
                                if (rr, cc) not in occupied:
                                    out[rr][cc] = 1
    return out


# ── 3af2c5a8: 4-way mirror → 2H × 2W output ──────────────────────────────────
def solve_3af2c5a8(grid: Grid) -> Grid:
    H, W = len(grid), len(grid[0])
    out = [[0] * (2 * W) for _ in range(2 * H)]
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            out[r][c] = v                        # TL
            out[r][2 * W - 1 - c] = v            # TR (flip LR)
            out[2 * H - 1 - r][c] = v            # BL (flip UD)
            out[2 * H - 1 - r][2 * W - 1 - c] = v  # BR (flip both)
    return out


# ── 3eda0437: Largest all-zero rectangle → fill with 6 ───────────────────────
def solve_3eda0437(grid: Grid) -> Grid:
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    hist = [0] * W
    best_area = 0
    best_r1 = best_r2 = best_c1 = best_c2 = 0

    for r in range(H):
        # Update histogram: extend zero-streak downward
        for c in range(W):
            if grid[r][c] == 0:
                hist[c] += 1
            else:
                hist[c] = 0

        # Largest rectangle in histogram (stack-based O(W))
        # Both height and width must be >= 2 (single-row/col spans excluded)
        stack = []  # (start_col, height)
        for c in range(W + 1):
            h = hist[c] if c < W else 0
            start = c
            while stack and stack[-1][1] > h:
                sc, sh = stack.pop()
                width = c - sc
                if sh >= 2 and width >= 2:
                    area = sh * width
                    if area > best_area:
                        best_area = area
                        best_c1 = sc
                        best_c2 = c - 1
                        best_r1 = r - sh + 1
                        best_r2 = r
                start = sc
            stack.append((start, h))

    # Fill best rectangle with 6
    for r in range(best_r1, best_r2 + 1):
        for c in range(best_c1, best_c2 + 1):
            out[r][c] = 6
    return out


# ── 37ce87bb: Vertical line segments → running sum → place 5 column ──────────
def solve_37ce87bb(grid: Grid) -> Grid:
    H, W = len(grid), len(grid[0])
    BG = 7
    out = [row[:] for row in grid]

    # Find non-BG columns: measure contiguous height from bottom
    col_info = []  # (col_idx, dominant_color, height)
    for c in range(W):
        height = 0
        col_color = None
        for r in range(H - 1, -1, -1):
            v = grid[r][c]
            if v != BG:
                if col_color is None:
                    col_color = v
                height += 1
            else:
                break  # contiguous from bottom only
        if col_color is not None and height > 0:
            col_info.append((c, col_color, height))

    if len(col_info) < 2:
        return out

    # Spacing = gap between consecutive column positions
    spacing = col_info[1][0] - col_info[0][0]

    # Running sum: +h for color 8, -h for color 2
    acc = 0
    for (_, color, h) in col_info:
        if color == 8:
            acc += h
        elif color == 2:
            acc -= h

    # Place 5 column at last_col + spacing, height=acc from bottom
    last_col = col_info[-1][0]
    next_col = last_col + spacing

    if 0 <= next_col < W and acc > 0:
        for r in range(H - acc, H):
            out[r][next_col] = 5

    return out
