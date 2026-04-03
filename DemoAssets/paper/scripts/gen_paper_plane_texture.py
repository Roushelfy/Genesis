"""
Generate texture for paper_plane_2_fine.obj (or coarse) with crease lines highlighted.

Uses the boundary crease detection from paper_plane_2_boundary_crease_debug_demo
to find crease edges, then generates a texture image with crease lines drawn in
a contrasting color and re-exports the OBJ with UVs + MTL.

Usage:
    python gen_paper_plane_texture.py [mesh.obj] [--tex-size 2048] [--line-width 6]

Output:
    <mesh>_textured.obj, <mesh>_textured.mtl, <mesh>_textured_tex.png
"""

import argparse
import os
import sys

import numpy as np

_FOLD_DIR = os.path.dirname(__file__)
if _FOLD_DIR not in sys.path:
    sys.path.insert(0, _FOLD_DIR)

from uipc import view
from uipc.geometry import SimplicialComplexIO, label_surface
import paper_plane_2_boundary_crease_debug_demo as boundary_debug

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description="Generate textured paper plane OBJ with crease lines")
    parser.add_argument("mesh", nargs="?", default=None, help="Path to paper plane OBJ")
    parser.add_argument("--tex-size", type=int, default=2048, help="Texture resolution (default 2048)")
    parser.add_argument("--line-width", type=int, default=6, help="Crease line width in pixels (default 6)")
    parser.add_argument("--paper-color", type=int, nargs=3, default=[245, 235, 220], help="Paper RGB color")
    parser.add_argument("--crease-color", type=int, nargs=3, default=[180, 60, 60], help="Crease line RGB color")
    args = parser.parse_args()

    # Resolve mesh path
    asset_filename = args.mesh or "paper_plane_2_fine.obj"
    mesh_path = boundary_debug.resolve_origami_asset_path(asset_filename)
    print(f"Loading mesh: {mesh_path}")

    # Load mesh and detect creases
    mesh = boundary_debug.load_paper_plane_mesh(asset_filename)
    diagnostics = boundary_debug.detect_boundary_creases(mesh, asset_filename)

    positions = np.array(view(mesh.positions()), copy=True).reshape(-1, 3)
    triangles = np.array(mesh.triangles().topo().view(), copy=True).reshape(-1, 3)
    n_verts = len(positions)
    n_tris = len(triangles)
    print(f"Mesh: {n_verts} verts, {n_tris} faces")

    # Detect planar axes — the mesh lies in a 2D plane embedded in 3D
    # Use the same plane detection as the crease detector
    topo = boundary_debug.build_topology(mesh)
    boundary_loop = boundary_debug.order_boundary_loop(
        boundary_debug.largest_boundary_component(topo["boundary_adj"]),
        topo["boundary_adj"],
    )
    plane_axes, normal_axis, bbox = boundary_debug.detect_planar_frame(positions, boundary_loop)

    # Extract 2D coordinates for UV mapping
    pos_2d = positions[:, plane_axes]
    u_min, u_max, v_min, v_max = bbox

    # Normalize to [0, 1] for UVs
    uvs = np.zeros((n_verts, 2), dtype=np.float64)
    u_range = u_max - u_min
    v_range = v_max - v_min
    if u_range < 1e-12:
        u_range = 1.0
    if v_range < 1e-12:
        v_range = 1.0
    uvs[:, 0] = (pos_2d[:, 0] - u_min) / u_range
    uvs[:, 1] = (pos_2d[:, 1] - v_min) / v_range

    # Collect crease edges from all chains
    crease_chains = diagnostics.get("chains", [])
    if not crease_chains:
        crease_lookup = diagnostics.get("crease_lookup", {})
        crease_chains = list({id(v): v for v in crease_lookup.values()}.values())

    print(f"Found {len(crease_chains)} crease chains")

    crease_edges = set()
    for chain in crease_chains:
        vids = chain.vertex_ids
        for i in range(len(vids) - 1):
            edge = tuple(sorted((vids[i], vids[i + 1])))
            crease_edges.add(edge)

    print(f"Total crease edges: {len(crease_edges)}")

    # Generate texture
    TEX_SIZE = args.tex_size
    PAPER_COLOR = tuple(args.paper_color)
    CREASE_COLOR = tuple(args.crease_color)
    LINE_WIDTH = args.line_width

    img = Image.new("RGB", (TEX_SIZE, TEX_SIZE), PAPER_COLOR)
    draw = ImageDraw.Draw(img)

    def uv_to_px(uv):
        return (uv[0] * TEX_SIZE, (1.0 - uv[1]) * TEX_SIZE)

    for a, b in crease_edges:
        pa = uv_to_px(uvs[a])
        pb = uv_to_px(uvs[b])
        draw.line([pa, pb], fill=CREASE_COLOR, width=LINE_WIDTH)

    # Output paths
    base = os.path.splitext(os.path.basename(mesh_path))[0]
    out_dir = os.path.dirname(mesh_path)
    obj_out = os.path.join(out_dir, f"{base}_textured.obj")
    mtl_out = os.path.join(out_dir, f"{base}_textured.mtl")
    tex_out = os.path.join(out_dir, f"{base}_textured_tex.png")

    img.save(tex_out)
    print(f"Wrote {tex_out}")

    # Write MTL
    mtl_basename = os.path.basename(mtl_out)
    tex_basename = os.path.basename(tex_out)
    with open(mtl_out, "w") as f:
        f.write("newmtl paper\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {tex_basename}\n")
    print(f"Wrote {mtl_out}")

    # Write OBJ with UVs
    with open(obj_out, "w") as f:
        f.write(f"mtllib {mtl_basename}\n")
        f.write("usemtl paper\n")
        for v in positions:
            f.write(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.10f} {uv[1]:.10f}\n")
        for tri in triangles:
            i, j, k = tri[0] + 1, tri[1] + 1, tri[2] + 1
            f.write(f"f {i}/{i} {j}/{j} {k}/{k}\n")
    print(f"Wrote {obj_out}")


if __name__ == "__main__":
    main()
