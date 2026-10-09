"""
grid_utils.py
Core grid manipulation primitives for ARC-AGI-2 solvers.
All operations are pure-function / side-effect-free unless noted.
"""
from collections import Counter, defaultdict, deque
import copy, math

# ────────────────────────────────────────────────────────────────
# Basic accessors
# ────────────────────────────────────────────────────────────────

def shape(grid):
    return len(grid), len(grid[0]) if grid else 0

def bg(grid):
    """Most-frequent cell value (background colour)."""
    c = Counter(v for row in grid for v in row)
    return max(c, key=c.get)

def fg_colors(grid):
    b = bg(grid)
    return sorted(set(v for row in grid for v in row) - {b})

def clone(grid):
    return [row[:] for row in grid]

def equal(a, b):
    if shape(a) != shape(b):
        return False
    return all(a[r][c] == b[r][c]
               for r in range(len(a)) for c in range(len(a[0])))

def make_grid(R, C, val=0):
    return [[val] * C for _ in range(R)]

# ────────────────────────────────────────────────────────────────
# Transformations
# ────────────────────────────────────────────────────────────────

def rotate_90cw(grid):
    R, C = shape(grid)
    return [[grid[R - 1 - c][r] for c in range(R)] for r in range(C)]

def rotate_180(grid):
    return rotate_90cw(rotate_90cw(grid))

def rotate_90ccw(grid):
    return rotate_90cw(rotate_90cw(rotate_90cw(grid)))

def flip_h(grid):
    return [row[::-1] for row in grid]

def flip_v(grid):
    return grid[::-1]

def transpose(grid):
    R, C = shape(grid)
    return [[grid[r][c] for r in range(R)] for c in range(C)]

def all_orientations(grid):
    """8 symmetry variants of a grid (4 rotations × 2 flips)."""
    variants = []
    g = clone(grid)
    for _ in range(4):
        variants.append(clone(g))
        variants.append(flip_h(g))
        g = rotate_90cw(g)
    return variants

# ────────────────────────────────────────────────────────────────
# Region / connectivity utilities
# ────────────────────────────────────────────────────────────────

def bfs_fill(grid, r0, c0, target, replacement):
    """In-place 4-connected flood fill."""
    R, C = shape(grid)
    if grid[r0][c0] != target or target == replacement:
        return
    q = deque([(r0, c0)])
    grid[r0][c0] = replacement
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == target:
                grid[nr][nc] = replacement
                q.append((nr, nc))

def connected_components(grid, color=None, include_bg=False):
    """Return list of cell-lists, one per 4-connected component."""
    R, C = shape(grid)
    bg_val = bg(grid) if not include_bg else None
    vis = [[False] * C for _ in range(R)]
    comps = []
    for r0 in range(R):
        for c0 in range(C):
            cv = grid[r0][c0]
            if vis[r0][c0]:
                continue
            if color is not None and cv != color:
                continue
            if not include_bg and cv == bg_val:
                continue
            q = deque([(r0, c0)])
            vis[r0][c0] = True
            comp = [(r0, c0)]
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C and not vis[nr][nc] and grid[nr][nc] == cv:
                        vis[nr][nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps

def bounding_box(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)

def extract_subgrid(grid, r1, c1, r2, c2):
    return [row[c1:c2+1] for row in grid[r1:r2+1]]

def paste_subgrid(canvas, sub, r_off, c_off):
    canvas = clone(canvas)
    for r, row in enumerate(sub):
        for c, v in enumerate(row):
            R, C = shape(canvas)
            if 0 <= r + r_off < R and 0 <= c + c_off < C:
                canvas[r + r_off][c + c_off] = v
    return canvas

# ────────────────────────────────────────────────────────────────
# Object utilities
# ────────────────────────────────────────────────────────────────

def find_objects(grid, ignore_bg=True):
    """Return list of dicts: {color, cells, bbox, subgrid}."""
    bg_val = bg(grid) if ignore_bg else None
    comps = connected_components(grid, include_bg=not ignore_bg)
    objects = []
    for comp in comps:
        col = grid[comp[0][0]][comp[0][1]]
        if ignore_bg and col == bg_val:
            continue
        r1, c1, r2, c2 = bounding_box(comp)
        sub = extract_subgrid(grid, r1, c1, r2, c2)
        objects.append({
            'color': col,
            'cells': comp,
            'bbox': (r1, c1, r2, c2),
            'subgrid': sub,
        })
    return objects

def object_mask(grid, cells):
    R, C = shape(grid)
    mask = make_grid(R, C, False)
    for r, c in cells:
        mask[r][c] = True
    return mask

# ────────────────────────────────────────────────────────────────
# Neighbourhood extraction
# ────────────────────────────────────────────────────────────────

def get_neighbourhood(grid, r, c, radius=1, pad_val=0):
    """Return flat tuple of (2r+1)x(2r+1) context around (r,c)."""
    R, C = shape(grid)
    window = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            window.append(grid[nr][nc] if 0 <= nr < R and 0 <= nc < C else pad_val)
    return tuple(window)

def get_neighbourhood_cross(grid, r, c, pad_val=0):
    """4-connected neighbours (N, S, W, E) + center."""
    R, C = shape(grid)
    def g(dr, dc):
        nr, nc = r+dr, c+dc
        return grid[nr][nc] if 0 <= nr < R and 0 <= nc < C else pad_val
    return (grid[r][c], g(-1,0), g(1,0), g(0,-1), g(0,1))

# ────────────────────────────────────────────────────────────────
# Text serialisation (for LLM interface)
# ────────────────────────────────────────────────────────────────

def grid_to_str(grid, sep=' ', row_sep='\n'):
    return row_sep.join(sep.join(str(v) for v in row) for row in grid)

def str_to_grid(text):
    return [[int(v) for v in row.split()] for row in text.strip().split('\n') if row.strip()]

def grid_to_compact(grid):
    """e.g. '012\\n345' — no spaces."""
    return '\n'.join(''.join(str(v) for v in row) for row in grid)

def compact_to_grid(text):
    return [[int(ch) for ch in row.strip()] for row in text.strip().split('\n') if row.strip()]

# ────────────────────────────────────────────────────────────────
# Colour utilities
# ────────────────────────────────────────────────────────────────

def remap_colors(grid, mapping):
    return [[mapping.get(v, v) for v in row] for row in grid]

def color_histogram(grid):
    return Counter(v for row in grid for v in row)

def normalize_colors(grid):
    """Remap so most-frequent = 0, next = 1, etc. Useful for comparison."""
    hist = color_histogram(grid)
    rank = {c: i for i, (c, _) in enumerate(hist.most_common())}
    return remap_colors(grid, rank)

# ────────────────────────────────────────────────────────────────
# Symmetry detection
# ────────────────────────────────────────────────────────────────

def is_horizontally_symmetric(grid):
    return grid == flip_h(grid)

def is_vertically_symmetric(grid):
    return grid == flip_v(grid)

def is_diagonally_symmetric(grid):
    R, C = shape(grid)
    if R != C:
        return False
    return grid == transpose(grid)

def is_rotationally_symmetric_180(grid):
    return grid == rotate_180(grid)

def count_unique_colors(grid):
    return len(set(v for row in grid for v in row))

# ────────────────────────────────────────────────────────────────
# Diff / delta utilities
# ────────────────────────────────────────────────────────────────

def grid_diff(a, b):
    """Return list of (r, c, val_a, val_b) where grids differ."""
    R, C = shape(a)
    return [(r, c, a[r][c], b[r][c])
            for r in range(R) for c in range(C)
            if a[r][c] != b[r][c]]

def changed_cells(inp, out):
    """Cells that changed from input to output."""
    return [(r, c, out[r][c]) for r, c, vi, vo in grid_diff(inp, out)]
