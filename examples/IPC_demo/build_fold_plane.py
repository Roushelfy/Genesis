"""
Generate a 1m x 1m plane mesh with fold lines for paper airplane.

7 radial fold lines emanate from the midpoint of the bottom edge (0.5, 0),
dividing the upper half-plane into 8 equal angular sectors (22.5 deg each).
The plane is also subdivided into a 20x20 grid.  All fold lines and grid
edges are embedded as mesh edges via constrained Delaunay triangulation.

Output: fold_plane.obj  (triangle mesh, Y-up)
"""

import numpy as np
import triangle as tr

# ---- Parameters ----
N = 20
SIZE = 1.0
ORIGIN = np.array([0.5, 0.0])
N_SECTORS = 8
OUTFILE = "fold_plane.obj"

h = SIZE / N

# ---- 1. Grid vertices (21 x 21) ----
xs = np.linspace(0, SIZE, N + 1)
ys = np.linspace(0, SIZE, N + 1)
grid_x, grid_y = np.meshgrid(xs, ys)
grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
n_grid = len(grid_pts)


def grid_idx(ix, iy):
    return iy * (N + 1) + ix


# ---- 2. Grid constraint edges (horizontal + vertical) ----
grid_edges = []
for iy in range(N + 1):
    for ix in range(N):
        grid_edges.append((grid_idx(ix, iy), grid_idx(ix + 1, iy)))
for ix in range(N + 1):
    for iy in range(N):
        grid_edges.append((grid_idx(ix, iy), grid_idx(ix, iy + 1)))

# ---- 3. Radial fold lines ----
# 7 lines at 22.5, 45, 67.5, 90, 112.5, 135, 157.5 degrees from +x
# Each line: from ORIGIN toward boundary, clipped to the unit square.
fold_angles_deg = [i * (180.0 / N_SECTORS) for i in range(1, N_SECTORS)]

extra_pts = []
extra_edges = []


def seg_seg_intersect(p1, p2, p3, p4):
    """Return parameter t for intersection of segment p1-p2 with line p3-p4.
    Returns None if parallel or t not in (epsilon, 1-epsilon)."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return None
    dp = p3 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
    s = (dp[0] * d1[1] - dp[1] * d1[0]) / cross
    if 0 - 1e-10 <= s <= 1 + 1e-10 and 1e-10 < t < 1 - 1e-10:
        return t
    return None


def snap_to_grid(pt, tol=1e-10):
    """Snap a point to the nearest grid vertex if within tol."""
    sx = round(pt[0] / h) * h
    sy = round(pt[1] / h) * h
    if abs(pt[0] - sx) < tol:
        pt[0] = sx
    if abs(pt[1] - sy) < tol:
        pt[1] = sy
    return pt


for angle_deg in fold_angles_deg:
    angle = np.deg2rad(angle_deg)
    direction = np.array([np.cos(angle), np.sin(angle)])

    # Find where the ray exits the unit square
    t_max = 1e9
    for boundary_val, axis, sign in [
        (SIZE, 1, +1),
        (SIZE, 0, +1),
        (0.0, 0, -1),
    ]:
        if abs(direction[axis]) > 1e-15:
            t_candidate = (boundary_val - ORIGIN[axis]) / direction[axis]
            if t_candidate > 1e-10:
                t_max = min(t_max, t_candidate)

    end_pt = ORIGIN + direction * t_max

    # Collect intersection points with all grid lines
    intersections = []

    # Intersect with vertical grid lines (x = k*h)
    for ix in range(N + 1):
        xval = ix * h
        if abs(direction[0]) > 1e-15:
            t = (xval - ORIGIN[0]) / direction[0]
            if 1e-10 < t < t_max - 1e-10:
                pt = ORIGIN + direction * t
                if -1e-10 <= pt[1] <= SIZE + 1e-10:
                    pt = snap_to_grid(pt.copy())
                    intersections.append((t, pt))

    # Intersect with horizontal grid lines (y = k*h)
    for iy in range(N + 1):
        yval = iy * h
        if abs(direction[1]) > 1e-15:
            t = (yval - ORIGIN[1]) / direction[1]
            if 1e-10 < t < t_max - 1e-10:
                pt = ORIGIN + direction * t
                if -1e-10 <= pt[0] <= SIZE + 1e-10:
                    pt = snap_to_grid(pt.copy())
                    intersections.append((t, pt))

    # Sort by parameter t, deduplicate
    intersections.sort(key=lambda x: x[0])
    filtered = []
    for t_val, pt in intersections:
        if not filtered or np.linalg.norm(pt - filtered[-1][1]) > 1e-10:
            filtered.append((t_val, pt))
    intersections = filtered

    # Build chain: origin -> intersection points -> end_pt
    chain_pts = [ORIGIN.copy()] + [pt for _, pt in intersections] + [snap_to_grid(end_pt.copy())]

    # Map each chain point to an existing grid vertex or a new extra vertex
    chain_indices = []
    for pt in chain_pts:
        # Check if it matches a grid vertex
        ix_round = round(pt[0] / h)
        iy_round = round(pt[1] / h)
        if (
            0 <= ix_round <= N
            and 0 <= iy_round <= N
            and abs(pt[0] - ix_round * h) < 1e-9
            and abs(pt[1] - iy_round * h) < 1e-9
        ):
            chain_indices.append(grid_idx(int(ix_round), int(iy_round)))
        else:
            # Check if it matches an already-added extra point
            found = False
            for ei, ep in enumerate(extra_pts):
                if np.linalg.norm(pt - ep) < 1e-9:
                    chain_indices.append(n_grid + ei)
                    found = True
                    break
            if not found:
                chain_indices.append(n_grid + len(extra_pts))
                extra_pts.append(pt.copy())

    # Add constraint edges along the chain
    for k in range(len(chain_indices) - 1):
        if chain_indices[k] != chain_indices[k + 1]:
            extra_edges.append((chain_indices[k], chain_indices[k + 1]))

# ---- 4. Also split grid edges that have extra points on them ----
# For each extra point, find which grid edge it lies on and split it.
split_edges = {}
for ei, ep in enumerate(extra_pts):
    global_idx = n_grid + ei
    # Check horizontal edges (y = iy*h, x between ix*h and (ix+1)*h)
    iy_round = round(ep[1] / h)
    if abs(ep[1] - iy_round * h) < 1e-9 and 0 <= iy_round <= N:
        ix_lo = int(np.floor(ep[0] / h - 1e-9))
        ix_lo = max(0, min(ix_lo, N - 1))
        edge_key = (grid_idx(ix_lo, int(iy_round)), grid_idx(ix_lo + 1, int(iy_round)))
        split_edges.setdefault(edge_key, []).append((ep[0], global_idx))

    # Check vertical edges (x = ix*h, y between iy*h and (iy+1)*h)
    ix_round = round(ep[0] / h)
    if abs(ep[0] - ix_round * h) < 1e-9 and 0 <= ix_round <= N:
        iy_lo = int(np.floor(ep[1] / h - 1e-9))
        iy_lo = max(0, min(iy_lo, N - 1))
        edge_key = (grid_idx(int(ix_round), iy_lo), grid_idx(int(ix_round), iy_lo + 1))
        split_edges.setdefault(edge_key, []).append((ep[1], global_idx))

# Rebuild grid edges, splitting where needed
final_grid_edges = []
for e in grid_edges:
    key = (min(e), max(e))
    e_sorted = (min(e), max(e))
    if e_sorted in split_edges:
        pts_on_edge = split_edges[e_sorted]
        pts_on_edge.sort(key=lambda x: x[0])
        chain = [e_sorted[0]] + [idx for _, idx in pts_on_edge] + [e_sorted[1]]
        for k in range(len(chain) - 1):
            if chain[k] != chain[k + 1]:
                final_grid_edges.append((chain[k], chain[k + 1]))
    else:
        final_grid_edges.append(e)

# ---- 5. Combine and triangulate ----
all_pts = np.vstack([grid_pts] + ([np.array(extra_pts)] if extra_pts else []))
all_edges = final_grid_edges + extra_edges

# Deduplicate edges
edge_set = set()
unique_edges = []
for e in all_edges:
    key = (min(e), max(e))
    if key not in edge_set:
        edge_set.add(key)
        unique_edges.append(list(key))

segments = np.array(unique_edges, dtype=np.int32)

# Run constrained Delaunay triangulation (no new Steiner points)
cdt_input = dict(vertices=all_pts, segments=segments)
cdt_result = tr.triangulate(cdt_input, "p")

out_verts = cdt_result["vertices"]
out_tris = cdt_result["triangles"]

print(f"Vertices: {len(out_verts)}")
print(f"Triangles: {len(out_tris)}")
print(f"Extra points (fold-line intersections): {len(extra_pts)}")

# ---- 6. Generate texture with fold lines ----
from PIL import Image, ImageDraw

TEX_SIZE = 2048
LINE_WIDTH = 24
PAPER_COLOR = (255, 230, 150)
FOLD_COLOR = (30, 30, 30)

img = Image.new("RGB", (TEX_SIZE, TEX_SIZE), PAPER_COLOR)
draw = ImageDraw.Draw(img)


# Draw fold lines on the texture.
# 2D coords are in [0, 1], texture pixel coords: u*TEX_SIZE, (1-v)*TEX_SIZE
def to_px(pt):
    return (pt[0] * TEX_SIZE, (1.0 - pt[1]) * TEX_SIZE)


for angle_deg in fold_angles_deg:
    angle = np.deg2rad(angle_deg)
    direction = np.array([np.cos(angle), np.sin(angle)])
    # Ray from ORIGIN to boundary
    t_max = 1e9
    for boundary_val, axis in [(SIZE, 1), (SIZE, 0), (0.0, 0)]:
        if abs(direction[axis]) > 1e-15:
            t_candidate = (boundary_val - ORIGIN[axis]) / direction[axis]
            if t_candidate > 1e-10:
                t_max = min(t_max, t_candidate)
    end_pt = ORIGIN + direction * t_max
    draw.line([to_px(ORIGIN), to_px(end_pt)], fill=FOLD_COLOR, width=LINE_WIDTH)

TEXFILE = OUTFILE.replace(".obj", "_tex.png")
img.save(TEXFILE)
print(f"Wrote {TEXFILE}")

# ---- 7. Export OBJ with UVs and material ----
# UVs are the 2D vertex coords (already in [0, 1])
# 3D coords: center at origin, XY plane (Z=0), range [-1, 1]
MTLFILE = OUTFILE.replace(".obj", ".mtl")
mtl_basename = MTLFILE.rsplit("/", 1)[-1]
tex_basename = TEXFILE.rsplit("/", 1)[-1]

with open(MTLFILE, "w") as f:
    f.write("newmtl paper\n")
    f.write("Ka 1.0 1.0 1.0\n")
    f.write("Kd 1.0 1.0 1.0\n")
    f.write("Ks 0.0 0.0 0.0\n")
    f.write(f"map_Kd {tex_basename}\n")
print(f"Wrote {MTLFILE}")

with open(OUTFILE, "w") as f:
    f.write(f"mtllib {mtl_basename}\n")
    f.write("usemtl paper\n")
    for v in out_verts:
        f.write(f"v {2 * v[0] - 1:.10f} {2 * v[1] - 1:.10f} 0.0\n")
    for v in out_verts:
        f.write(f"vt {v[0]:.10f} {v[1]:.10f}\n")
    for tri in out_tris:
        i, j, k = tri[0] + 1, tri[1] + 1, tri[2] + 1
        f.write(f"f {i}/{i} {j}/{j} {k}/{k}\n")

print(f"Wrote {OUTFILE}")
