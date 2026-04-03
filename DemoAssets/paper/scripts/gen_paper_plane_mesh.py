"""
Generate paper airplane mesh from crease line definitions.

Produces a flat rectangular mesh with crease edges embedded via constrained
Delaunay triangulation.  Mesh density is controlled by ``--max-area``.

The 14 crease lines are taken from the paper_plane_2 design.  The rectangle
spans X:[-1,1] Z:[-1.414,1.414] at Y=0.1.

Usage:
    python gen_paper_plane_mesh.py                    # default density
    python gen_paper_plane_mesh.py --max-area 0.005   # coarser
    python gen_paper_plane_mesh.py --max-area 0.0005  # finer
    python gen_paper_plane_mesh.py --output my_paper.obj
"""

import argparse
import json
import os
import struct

import numpy as np
import triangle as tr
from PIL import Image, ImageDraw

# ---- Rectangle bounds (2D: X, Z mapped to u, v) ----
X_MIN, X_MAX = -1.0, 1.0
Z_MIN, Z_MAX = -1.414, 1.414
Y_CONST = 0.1

# ---- 14 crease lines as (u0, v0) -> (u1, v1) ----
# Coordinates are in the XZ plane of the 3D mesh.
CREASE_LINES = [
    # Verticals (BT_V1, BT_V2, BT_V3)
    ((-0.24815, Z_MIN), (-0.24815, Z_MAX)),
    ((0.0, Z_MIN), (0.0, Z_MAX)),
    ((0.24815, Z_MIN), (0.24815, Z_MAX)),
    # Horizontals (LR_H1..H4)
    ((X_MIN, 1.16585), (X_MAX, 1.16585)),
    ((X_MIN, 0.414), (X_MAX, 0.414)),
    ((X_MIN, 0.16585), (X_MAX, 0.16585)),
    ((X_MIN, -0.0823), (X_MAX, -0.0823)),
    # Diagonals
    ((0.0, Z_MAX), (X_MAX, 0.414)),  # TR_DN1
    ((X_MIN, 1.16585), (X_MAX, -0.83415)),  # LR_DN1
    ((X_MIN, -0.83415), (-0.24815, Z_MIN)),  # LB_DN1
    ((X_MIN, 1.16585), (-0.24815, Z_MAX)),  # LT_DP1
    ((X_MIN, 0.414), (0.0, Z_MAX)),  # LT_DP2
    ((X_MIN, -0.83415), (X_MAX, 1.16585)),  # LR_DP1
    ((0.24815, Z_MIN), (X_MAX, -0.83415)),  # BR_DP1
]


def main():
    parser = argparse.ArgumentParser(description="Generate paper airplane mesh with crease lines")
    parser.add_argument(
        "--max-area", type=float, default=0.002, help="Max triangle area (controls density, default 0.002)"
    )
    parser.add_argument("--tex-size", type=int, default=2048, help="Texture resolution")
    parser.add_argument("--line-width", type=int, default=8, help="Crease line width in pixels")
    parser.add_argument("--paper-color", type=int, nargs=3, default=[245, 235, 220])
    parser.add_argument("--crease-color", type=int, nargs=3, default=[180, 60, 60])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Collect all vertices and constraint segments
    pts = []
    pt_index = {}
    segments = []

    def add_pt(x, z):
        key = (round(x, 8), round(z, 8))
        if key not in pt_index:
            pt_index[key] = len(pts)
            pts.append([x, z])
        return pt_index[key]

    # Rectangle boundary
    bl = add_pt(X_MIN, Z_MIN)
    br = add_pt(X_MAX, Z_MIN)
    tr_ = add_pt(X_MAX, Z_MAX)
    tl = add_pt(X_MIN, Z_MAX)
    segments += [(bl, br), (br, tr_), (tr_, tl), (tl, bl)]

    # Crease lines — clip to rectangle and add intersection points with
    # other creases and the boundary.
    for (x0, z0), (x1, z1) in CREASE_LINES:
        # Add endpoints
        i0 = add_pt(x0, z0)
        i1 = add_pt(x1, z1)

        # Find intersections with all other crease lines
        d = np.array([x1 - x0, z1 - z0])
        crease_pts = [(0.0, i0)]
        for (ox0, oz0), (ox1, oz1) in CREASE_LINES:
            if (ox0, oz0, ox1, oz1) == (x0, z0, x1, z1):
                continue
            od = np.array([ox1 - ox0, oz1 - oz0])
            cross = d[0] * od[1] - d[1] * od[0]
            if abs(cross) < 1e-12:
                continue
            dp = np.array([ox0 - x0, oz0 - z0])
            t = (dp[0] * od[1] - dp[1] * od[0]) / cross
            s = (dp[0] * d[1] - dp[1] * d[0]) / cross
            if 1e-10 < t < 1 - 1e-10 and -1e-10 < s < 1 + 1e-10:
                ix = x0 + t * d[0]
                iz = z0 + t * d[1]
                idx = add_pt(ix, iz)
                crease_pts.append((t, idx))

        crease_pts.append((1.0, i1))
        crease_pts.sort()

        # Deduplicate
        chain = []
        for _, idx in crease_pts:
            if not chain or chain[-1] != idx:
                chain.append(idx)

        for k in range(len(chain) - 1):
            seg = (min(chain[k], chain[k + 1]), max(chain[k], chain[k + 1]))
            segments.append(seg)

    # Deduplicate segments
    seg_set = set()
    unique_segs = []
    for s in segments:
        key = (min(s), max(s))
        if key not in seg_set:
            seg_set.add(key)
            unique_segs.append(list(key))

    vertices = np.array(pts, dtype=np.float64)
    seg_array = np.array(unique_segs, dtype=np.int32)

    print(f"Constraint vertices: {len(vertices)}")
    print(f"Constraint segments: {len(seg_array)}")

    # Constrained Delaunay triangulation with area constraint
    cdt_input = dict(vertices=vertices, segments=seg_array)
    cdt_result = tr.triangulate(cdt_input, f"pq30a{args.max_area}")

    out_verts_2d = cdt_result["vertices"]
    out_tris = cdt_result["triangles"]

    print(f"Output: {len(out_verts_2d)} vertices, {len(out_tris)} triangles")

    # Convert to 3D (X, Y=0.1, Z)
    out_verts_3d = np.zeros((len(out_verts_2d), 3), dtype=np.float64)
    out_verts_3d[:, 0] = out_verts_2d[:, 0]
    out_verts_3d[:, 1] = Y_CONST
    out_verts_3d[:, 2] = out_verts_2d[:, 1]

    # UVs: normalize 2D coords to [0, 1]
    u_range = X_MAX - X_MIN
    v_range = Z_MAX - Z_MIN
    uvs = np.zeros((len(out_verts_2d), 2), dtype=np.float32)
    uvs[:, 0] = (out_verts_2d[:, 0] - X_MIN) / u_range
    uvs[:, 1] = (out_verts_2d[:, 1] - Z_MIN) / v_range

    # Generate texture with crease lines
    TEX_SIZE = args.tex_size
    PAPER_COLOR = tuple(args.paper_color)
    CREASE_COLOR = tuple(args.crease_color)
    LINE_WIDTH = args.line_width

    img = Image.new("RGB", (TEX_SIZE, TEX_SIZE), PAPER_COLOR)
    draw = ImageDraw.Draw(img)

    def uv_to_px(u, v):
        return (u * TEX_SIZE, (1.0 - v) * TEX_SIZE)

    for (x0, z0), (x1, z1) in CREASE_LINES:
        u0 = (x0 - X_MIN) / u_range
        v0 = (z0 - Z_MIN) / v_range
        u1 = (x1 - X_MIN) / u_range
        v1 = (z1 - Z_MIN) / v_range
        draw.line([uv_to_px(u0, v0), uv_to_px(u1, v1)], fill=CREASE_COLOR, width=LINE_WIDTH)

    # Output paths
    if args.output is None:
        asset_dir = os.path.join(os.path.dirname(__file__), "..")
        args.output = os.path.join(asset_dir, "paper_plane_2_gen.glb")

    base = os.path.splitext(args.output)[0]

    # Save texture
    tex_path = base + "_tex.png"
    img.save(tex_path)
    print(f"Wrote: {tex_path}")

    # Save OBJ + MTL
    obj_path = base + ".obj"
    mtl_path = base + ".mtl"
    mtl_name = os.path.basename(mtl_path)
    tex_name = os.path.basename(tex_path)

    with open(mtl_path, "w") as f:
        f.write("newmtl paper\n")
        f.write("Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
        f.write(f"map_Kd {tex_name}\n")

    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_name}\nusemtl paper\n")
        for v in out_verts_3d:
            f.write(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.10f} {uv[1]:.10f}\n")
        for tri in out_tris:
            i, j, k = tri[0] + 1, tri[1] + 1, tri[2] + 1
            f.write(f"f {i}/{i} {j}/{j} {k}/{k}\n")
    print(f"Wrote: {obj_path}")

    # Convert to GLB with doubleSided material
    import trimesh

    m = trimesh.load(obj_path)
    glb_path = base + ".glb"
    from trimesh.exchange.gltf import export_glb

    glb_data = export_glb(m)

    # Patch doubleSided
    magic, version, length = struct.unpack("<III", glb_data[:12])
    json_len, json_type = struct.unpack("<II", glb_data[12:20])
    gltf = json.loads(glb_data[20 : 20 + json_len])
    for mat in gltf.get("materials", []):
        mat["doubleSided"] = True
    new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(new_json) % 4 != 0:
        new_json += b" "
    bin_chunk = glb_data[20 + json_len :]
    new_length = 12 + 8 + len(new_json) + len(bin_chunk)
    with open(glb_path, "wb") as f:
        f.write(struct.pack("<III", magic, version, new_length))
        f.write(struct.pack("<II", len(new_json), json_type))
        f.write(new_json)
        f.write(bin_chunk)
    print(f"Wrote: {glb_path}")


if __name__ == "__main__":
    main()
