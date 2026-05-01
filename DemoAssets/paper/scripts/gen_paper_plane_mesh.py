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
from scipy.spatial import Delaunay

# ---- Rectangle bounds (2D: X, Z mapped to u, v) ----
X_MIN, X_MAX = -1.0, 1.0
Z_MIN, Z_MAX = -1.414, 1.414
Y_CONST = 0.1

# ---- 6 crease lines as (u0, v0) -> (u1, v1) ----
# Coordinates are in the XZ plane of the 3D mesh.
_CREASE_HX = 0.3  # half-width of inner verticals (creases 0 and 2)
_CREASE_HZ = Z_MAX - _CREASE_HX  # crease 3 z, geometrically tied: Z_MAX - _CREASE_HX

CREASE_LINES = [
    # Verticals (BT_V1, BT_V2, BT_V3)
    ((-_CREASE_HX, Z_MIN), (-_CREASE_HX, Z_MAX)),
    ((0.0, Z_MIN), (0.0, Z_MAX)),
    ((_CREASE_HX, Z_MIN), (_CREASE_HX, Z_MAX)),
    # Horizontal
    ((X_MIN, _CREASE_HZ), (X_MAX, _CREASE_HZ)),  # LR_H1
    # Diagonals (symmetric V at top)
    ((0.0, Z_MAX), (X_MAX, 0.414)),  # TR_DN1
    ((X_MIN, 0.414), (0.0, Z_MAX)),  # LT_DP2
]

# ---- Special region: between crease 0 (x=-_CREASE_HX) and crease 2 (x=+_CREASE_HX),
# below crease 3 (z=+_CREASE_HZ). Triangulated manually as a regular zigzag so
# every triangle vertex lies on crease 0/1/2 — no interior verts means no
# phantom collinear chains can form in these narrow strips.
SPECIAL_X = (-_CREASE_HX, 0.0, _CREASE_HX)
SPECIAL_Z_BOT = Z_MIN
SPECIAL_Z_TOP = _CREASE_HZ


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
    parser.add_argument(
        "--lloyd-iters",
        type=int,
        default=8,
        help="Lloyd relaxation iterations (centroidal Voronoi). "
        "Breaks the phantom diagonal lines in narrow strips while keeping "
        "uniform triangle areas. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for symmetry-breaking jitter.")
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

    # ---- Step 1: pre-place all "key" vertices --------------------------------
    # Rectangle corners
    add_pt(X_MIN, Z_MIN)
    add_pt(X_MAX, Z_MIN)
    add_pt(X_MAX, Z_MAX)
    add_pt(X_MIN, Z_MAX)
    # Crease endpoints (intersections of crease pairs all coincide with
    # endpoints/special-region pts in this design, so endpoints alone suffice
    # along with the pre-placed special-region grid below).
    for (x0, z0), (x1, z1) in CREASE_LINES:
        add_pt(x0, z0)
        add_pt(x1, z1)

    # ---- Step 2: pre-place the special-region zigzag grid --------------------
    # dz_special is chosen to match the OUTER mesher's natural edge length
    # (equilateral edge for area=max_area). This way the verts the zigzag puts
    # along x=±_CREASE_HX have the same spacing the outer CDT/Lloyd wants on
    # that boundary — no density mismatch, no sliver tris in the outer region.
    # Tradeoff: inner zigzag tris end up with area = strip_w * dz_special / 2,
    # which is larger than max_area (aspect ratio strip_w/dz_special). That's
    # acceptable: the inner region has fixed geometry (3 verts on creases) and
    # is excluded from the "uniform area" target.
    strip_w = SPECIAL_X[1] - SPECIAL_X[0]
    target_dz = float(np.sqrt(4.0 * args.max_area / np.sqrt(3.0)))
    span = SPECIAL_Z_TOP - SPECIAL_Z_BOT
    n_dz = max(1, int(round(span / target_dz)))
    dz_special = span / n_dz
    n_levels = n_dz + 1

    zigzag_idx: dict[tuple[int, int], int] = {}
    for xi, x in enumerate(SPECIAL_X):
        for zi in range(n_levels):
            z = SPECIAL_Z_BOT + zi * dz_special
            zigzag_idx[(xi, zi)] = add_pt(x, z)

    print(
        f"Special region: {n_dz} dz steps, dz={dz_special:.5f}, "
        f"per-tri area={strip_w * dz_special / 2:.5f} (target {args.max_area})"
    )

    # ---- Step 3: chain all collinear pts on each constraint line -------------
    # Target edge length for an equilateral tri of area = max_area.
    # All boundary subsegments get subdivided to roughly this length so
    # Triangle's `YY` flag (no Steiner pts on segments) doesn't starve the
    # CDT — it can fill the interior to satisfy q30 + max_area without
    # needing to split boundary segments.
    edge_len = float(np.sqrt(4.0 * args.max_area / np.sqrt(3.0)))

    def chain_collinear(p_start, p_end):
        x0_, z0_ = p_start
        x1_, z1_ = p_end
        dx = x1_ - x0_
        dz = z1_ - z0_
        d_len2 = dx * dx + dz * dz
        found = []
        for idx, (px, pz) in enumerate(pts):
            t = ((px - x0_) * dx + (pz - z0_) * dz) / d_len2
            proj_x = x0_ + t * dx
            proj_z = z0_ + t * dz
            perp2 = (px - proj_x) ** 2 + (pz - proj_z) ** 2
            if perp2 < 1e-14 and -1e-9 < t < 1 + 1e-9:
                found.append((max(0.0, min(1.0, t)), idx))
        found.sort()
        chain = []
        for _, idx in found:
            if not chain or chain[-1] != idx:
                chain.append(idx)
        return chain

    def _on_special_h_boundary(ax, az, bx, bz) -> bool:
        """Subsegment lies on the special region's top (crease 3) or bottom
        (rect bot) horizontal boundary, between SPECIAL_X[0] and SPECIAL_X[2].
        Subdivision here would T-junction with the manual zigzag triangles
        whose top/bottom edges span a full strip width (0.248) without
        intermediate verts."""
        eps = 1e-9
        on_top = abs(az - SPECIAL_Z_TOP) < eps and abs(bz - SPECIAL_Z_TOP) < eps
        on_bot = abs(az - SPECIAL_Z_BOT) < eps and abs(bz - SPECIAL_Z_BOT) < eps
        in_x = SPECIAL_X[0] - eps <= ax <= SPECIAL_X[2] + eps and SPECIAL_X[0] - eps <= bx <= SPECIAL_X[2] + eps
        return (on_top or on_bot) and in_x

    def add_chain(p_start, p_end):
        # Chain existing key pts on the line, then subdivide each subsegment
        # to target spacing (skip subdivision when subseg already short enough,
        # or when it bounds the special region — see _on_special_h_boundary).
        chain = chain_collinear(p_start, p_end)
        full_chain = [chain[0]]
        for k in range(len(chain) - 1):
            ax, az = pts[chain[k]]
            bx, bz = pts[chain[k + 1]]
            if _on_special_h_boundary(ax, az, bx, bz):
                full_chain.append(chain[k + 1])
                continue
            sub_len = float(np.hypot(bx - ax, bz - az))
            n = max(1, int(round(sub_len / edge_len)))
            for j in range(1, n):
                t = j / n
                full_chain.append(add_pt(ax + t * (bx - ax), az + t * (bz - az)))
            full_chain.append(chain[k + 1])
        for k in range(len(full_chain) - 1):
            segments.append((min(full_chain[k], full_chain[k + 1]), max(full_chain[k], full_chain[k + 1])))

    # Rectangle boundary (4 edges) — chained so pre-placed bottom verts split it
    add_chain((X_MIN, Z_MIN), (X_MAX, Z_MIN))
    add_chain((X_MAX, Z_MIN), (X_MAX, Z_MAX))
    add_chain((X_MAX, Z_MAX), (X_MIN, Z_MAX))
    add_chain((X_MIN, Z_MAX), (X_MIN, Z_MIN))

    # Crease lines — chained so pre-placed special-region verts split them
    for (x0, z0), (x1, z1) in CREASE_LINES:
        add_chain((x0, z0), (x1, z1))

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
    n_input_pts = len(pts)

    # Hole points: one inside each strip of the special region. Triangle's
    # CDT skips triangles whose centroid sits inside a hole region (bounded
    # by segments), so the special region won't be triangulated by CDT —
    # we'll fill it manually with the zigzag pattern after Lloyd.
    hole_z = 0.5 * (SPECIAL_Z_BOT + SPECIAL_Z_TOP)
    holes = np.array(
        [
            [0.5 * (SPECIAL_X[0] + SPECIAL_X[1]), hole_z],  # left strip
            [0.5 * (SPECIAL_X[1] + SPECIAL_X[2]), hole_z],  # right strip
        ],
        dtype=np.float64,
    )

    print(f"Constraint vertices: {len(vertices)}")
    print(f"Constraint segments: {len(seg_array)}")

    # Constrained Delaunay triangulation with area constraint, special region
    # marked as holes so it stays empty (we'll fill it manually).
    # `YY` forbids Steiner points on any segment, which keeps the special
    # region's pre-placed crease verts as the only verts on creases 0/1/2 in
    # that range — without it, CDT adds Steiner pts to satisfy quality on
    # the non-special side and creates T-junctions with the manual zigzag.
    cdt_input = dict(vertices=vertices, segments=seg_array, holes=holes)
    cdt_result = tr.triangulate(cdt_input, f"pq30a{args.max_area}YY")

    out_verts_2d = cdt_result["vertices"]
    out_tris = cdt_result["triangles"]

    print(f"Output: {len(out_verts_2d)} vertices, {len(out_tris)} triangles")

    # Lloyd relaxation: re-position interior verts to incident-triangle centroid.
    # Boundary and crease-line vertices stay pinned. Re-triangulate each iter
    # so connectivity adapts (this is what kills phantom diagonal lines).
    if args.lloyd_iters > 0:
        crease_segs_xy = [
            (np.array([x0, z0], dtype=np.float64), np.array([x1, z1], dtype=np.float64))
            for (x0, z0), (x1, z1) in CREASE_LINES
        ]
        # Min distance any interior vertex must keep to the rectangle boundary
        # and to any crease segment. Anything closer = "on the constraint" and
        # gets pinned. Loose enough that Lloyd moves don't dip below it.
        SAFE_D = 0.25 * float(np.sqrt(args.max_area))

        def dist_to_seg(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            ab = b - a
            ab2 = float(ab @ ab)
            if ab2 < 1e-18:
                return float(np.linalg.norm(pt - a))
            t = float((pt - a) @ ab) / ab2
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            return float(np.linalg.norm(pt - closest))

        def near_any_constraint(pt: np.ndarray, eps: float) -> bool:
            x, z = float(pt[0]), float(pt[1])
            if x - X_MIN < eps or X_MAX - x < eps:
                return True
            if z - Z_MIN < eps or Z_MAX - z < eps:
                return True
            for a, b in crease_segs_xy:
                if dist_to_seg(pt, a, b) < eps:
                    return True
            return False

        # Vertex indices on crease 1 strictly inside the special region.
        # Triangles touching any of them are dropped from the intermediate
        # scipy.Delaunay output (they would cross the special region).
        crease1_interior_indices = {zigzag_idx[(1, zi)] for zi in range(1, n_levels - 1)}

        def _tri_in_special(simp, verts_arr) -> bool:
            a, b, c = int(simp[0]), int(simp[1]), int(simp[2])
            if a in crease1_interior_indices or b in crease1_interior_indices or c in crease1_interior_indices:
                return True
            cx = (verts_arr[a, 0] + verts_arr[b, 0] + verts_arr[c, 0]) / 3.0
            cz = (verts_arr[a, 1] + verts_arr[b, 1] + verts_arr[c, 1]) / 3.0
            return SPECIAL_X[0] < cx < SPECIAL_X[2] and SPECIAL_Z_BOT < cz < SPECIAL_Z_TOP

        # Symmetry-breaking jitter on interior verts BEFORE Lloyd. Without this,
        # narrow strips between adjacent verticals sit at a symmetric 2-row
        # equilibrium that Lloyd can't escape — phantom collinear midline.
        # x-bias is heavier because the phantom runs vertically; we need to
        # displace verts across the strip, not along it.
        rng = np.random.default_rng(args.seed)
        edge_len = float(np.sqrt(4.0 * args.max_area / np.sqrt(3.0)))
        jitter_x = 0.45 * edge_len
        jitter_z = 0.15 * edge_len
        verts0 = np.asarray(out_verts_2d, dtype=np.float64).copy()
        for vi in range(len(verts0)):
            if near_any_constraint(verts0[vi], 1e-7):
                continue
            verts0[vi, 0] += rng.uniform(-jitter_x, jitter_x)
            verts0[vi, 1] += rng.uniform(-jitter_z, jitter_z)
        out_verts_2d = verts0

        for it in range(args.lloyd_iters):
            verts = np.asarray(out_verts_2d, dtype=np.float64)
            tris = np.asarray(out_tris, dtype=np.int32)
            n_verts = len(verts)
            # Free = not on any constraint
            pin_eps = 1e-7
            is_pinned = np.zeros(n_verts, dtype=bool)
            for vi in range(n_verts):
                if near_any_constraint(verts[vi], pin_eps):
                    is_pinned[vi] = True

            # Area-weighted incident-triangle centroid
            centroid_sum = np.zeros_like(verts)
            weight_sum = np.zeros(n_verts, dtype=np.float64)
            for ti in range(len(tris)):
                a, b, c = tris[ti]
                pa, pb, pc = verts[a], verts[b], verts[c]
                area = 0.5 * abs((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1]))
                centroid = (pa + pb + pc) / 3.0
                for v in (a, b, c):
                    centroid_sum[v] += area * centroid
                    weight_sum[v] += area

            new_verts = verts.copy()
            damping = 0.25
            moved = 0
            for vi in range(n_verts):
                if is_pinned[vi] or weight_sum[vi] <= 0.0:
                    continue
                target = centroid_sum[vi] / weight_sum[vi]
                proposed = (1.0 - damping) * verts[vi] + damping * target
                # Reject any move that crosses below SAFE_D from a constraint.
                if near_any_constraint(proposed, SAFE_D):
                    continue
                new_verts[vi] = proposed
                moved += 1

            # Plain Delaunay on the moved vertices (no constraints during
            # smoothing — final constrained CDT applied below). Drop any
            # triangle that falls inside the special region — the final
            # mesh fills that region with a manual zigzag pattern.
            dt = Delaunay(new_verts)
            simplices = np.asarray(dt.simplices)
            kept = [simplices[ti] for ti in range(len(simplices)) if not _tri_in_special(simplices[ti], new_verts)]
            out_verts_2d = new_verts
            out_tris = np.array(kept, dtype=np.int32) if kept else np.zeros((0, 3), dtype=np.int32)
            print(
                f"  Lloyd iter {it + 1}/{args.lloyd_iters}: moved {moved} verts, "
                f"now {len(out_verts_2d)} v / {len(out_tris)} f (special excluded)"
            )

        # Final constrained re-triangulation: re-snap crease/boundary segments
        # so the post-Lloyd vertex layout still respects the constraints.
        # Always keep input verts (indices < n_input_pts). Drop Steiner pts
        # that drifted within SAFE_D of any constraint — at the asset scale
        # used in IPC (0.125), even a paper-local distance of 1e-3 gives a
        # world distance below contact_d_hat, so the threshold has to match
        # SAFE_D rather than a tiny numerical eps.
        keep_mask = np.ones(len(out_verts_2d), dtype=bool)
        for vi in range(n_input_pts, len(out_verts_2d)):
            if near_any_constraint(np.asarray(out_verts_2d[vi]), SAFE_D):
                keep_mask[vi] = False
        kept_verts = np.asarray(out_verts_2d)[keep_mask]
        cdt_final = dict(vertices=kept_verts, segments=seg_array, holes=holes)
        cdt_result = tr.triangulate(cdt_final, f"pq30a{args.max_area}YY")
        out_verts_2d = cdt_result["vertices"]
        out_tris = cdt_result["triangles"]
        print(f"After Lloyd + final CDT: {len(out_verts_2d)} v / {len(out_tris)} f")

    # ---- Append manual zigzag triangles for the special region ---------------
    # Every triangle has all 3 verts on crease 0/1/2, with uniform area
    # ~max_area. Diagonals alternate per row so no long collinear chain forms.
    zigzag_tris = []
    for xi in range(2):  # 0 = left strip (creases 0-1), 1 = right strip (1-2)
        for zi in range(n_dz):
            v_bl = zigzag_idx[(xi, zi)]
            v_br = zigzag_idx[(xi + 1, zi)]
            v_tr = zigzag_idx[(xi + 1, zi + 1)]
            v_tl = zigzag_idx[(xi, zi + 1)]
            if zi % 2 == 0:
                zigzag_tris.append([v_bl, v_br, v_tr])
                zigzag_tris.append([v_bl, v_tr, v_tl])
            else:
                zigzag_tris.append([v_bl, v_br, v_tl])
                zigzag_tris.append([v_br, v_tr, v_tl])
    out_tris = np.vstack(
        [
            np.asarray(out_tris, dtype=np.int64),
            np.array(zigzag_tris, dtype=np.int64),
        ]
    )
    print(
        f"Special region zigzag: +{len(zigzag_tris)} triangles "
        f"({2 * n_dz} per strip × 2 strips). Total: {len(out_tris)} f"
    )

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
