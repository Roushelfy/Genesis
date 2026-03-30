"""
Generate a brick-with-handle mesh for pressing/folding paper.

Shape: rectangular brick base + door-handle-like arch on top.
The handle is a bridge (two posts + horizontal bar) that a gripper can grasp.

Output: brick_press.obj (centered at origin, Z-up)
"""

import numpy as np

OUTFILE = "brick_press.obj"

# ---- Dimensions (meters) ----
# Brick base
BRICK_W = 0.16  # X width (along paper fold line)
BRICK_D = 0.08  # Y depth
BRICK_H = 0.03  # Z height

# Handle posts
POST_W = 0.015  # X thickness of each post
POST_D = 0.02  # Y depth of each post
POST_H = 0.045  # Z height of posts above brick top
POST_INSET = 0.02  # X inset from brick edge

# Handle bar
BAR_W = BRICK_W - 2 * POST_INSET  # X span (connects the two posts)
BAR_D = POST_D  # Y depth (same as posts)
BAR_H = 0.015  # Z thickness of horizontal bar


def make_box(cx, cy, cz, sx, sy, sz):
    """Create a box centered at (cx, cy, cz) with half-sizes (sx, sy, sz).
    Returns (verts, faces) where faces are 0-indexed."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ]
    )
    # 12 triangles (2 per face)
    f = np.array(
        [
            # -Z face
            [0, 2, 1],
            [0, 3, 2],
            # +Z face
            [4, 5, 6],
            [4, 6, 7],
            # -Y face
            [0, 1, 5],
            [0, 5, 4],
            # +Y face
            [2, 3, 7],
            [2, 7, 6],
            # -X face
            [0, 4, 7],
            [0, 7, 3],
            # +X face
            [1, 2, 6],
            [1, 6, 5],
        ]
    )
    return v, f


def make_cylinder_arc(cx, cy, cz, radius, thickness, depth, n_seg=24):
    """Create a half-cylinder arc (180 deg) in the XZ plane centered at (cx, cy, cz).
    Arc goes from -X to +X over the top (+Z).
    Returns (verts, faces)."""
    angles = np.linspace(0, np.pi, n_seg + 1)
    r_outer = radius
    r_inner = radius - thickness
    hy = depth / 2

    verts = []
    # For each angle, 4 verts: outer-front, outer-back, inner-front, inner-back
    for a in angles:
        cos_a, sin_a = np.cos(a), np.sin(a)
        verts.append([cx + r_outer * cos_a, cy - hy, cz + r_outer * sin_a])
        verts.append([cx + r_outer * cos_a, cy + hy, cz + r_outer * sin_a])
        verts.append([cx + r_inner * cos_a, cy - hy, cz + r_inner * sin_a])
        verts.append([cx + r_inner * cos_a, cy + hy, cz + r_inner * sin_a])

    verts = np.array(verts)
    faces = []
    for i in range(n_seg):
        base = i * 4
        nxt = (i + 1) * 4
        # Outer face (normal pointing outward)
        faces.append([base + 0, nxt + 1, nxt + 0])
        faces.append([base + 0, base + 1, nxt + 1])
        # Inner face (normal pointing inward toward arc center)
        faces.append([base + 2, nxt + 2, nxt + 3])
        faces.append([base + 2, nxt + 3, base + 3])
        # Front face (-Y, normal pointing -Y)
        faces.append([base + 0, nxt + 0, nxt + 2])
        faces.append([base + 0, nxt + 2, base + 2])
        # Back face (+Y, normal pointing +Y)
        faces.append([base + 1, nxt + 3, nxt + 1])
        faces.append([base + 1, base + 3, nxt + 3])

    # Cap at angle=0 (right side, normal pointing +X)
    faces.append([0, 3, 1])
    faces.append([0, 2, 3])
    # Cap at angle=pi (left side, normal pointing -X)
    last = n_seg * 4
    faces.append([last + 0, last + 1, last + 3])
    faces.append([last + 0, last + 3, last + 2])

    return verts, np.array(faces)


def merge_meshes(parts):
    """Merge list of (verts, faces) into a single mesh."""
    all_verts = []
    all_faces = []
    offset = 0
    for v, f in parts:
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += len(v)
    return np.vstack(all_verts), np.vstack(all_faces)


# ---- Build geometry ----
parts = []

# 1. Brick base — centered at origin, bottom at Z=0
brick_cz = BRICK_H / 2
parts.append(make_box(0, 0, brick_cz, BRICK_W, BRICK_D, BRICK_H))

# 2. Handle — arch on top of brick
# Arc center at brick top surface, spanning half the brick width
arc_radius = BAR_W / 2
arc_cz = BRICK_H
parts.append(
    make_cylinder_arc(
        cx=0,
        cy=0,
        cz=arc_cz,
        radius=arc_radius + BAR_H,
        thickness=BAR_H,
        depth=BAR_D,
        n_seg=32,
    )
)

verts, faces = merge_meshes(parts)

# ---- Export OBJ ----
with open(OUTFILE, "w") as f:
    for v in verts:
        f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    for tri in faces:
        f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

print(f"Vertices: {len(verts)}")
print(f"Triangles: {len(faces)}")
print(f"Brick: {BRICK_W * 100:.0f}x{BRICK_D * 100:.0f}x{BRICK_H * 100:.0f} cm")
print(f"Handle arch radius: {arc_radius * 100:.1f} cm, bar thickness: {BAR_H * 100:.1f} cm")
print(f"Handle top at Z={arc_cz + arc_radius + BAR_H:.4f} m")
print(f"Wrote {OUTFILE}")
