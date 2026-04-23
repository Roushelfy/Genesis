"""
subdivide_trashbag_trajectory.py — Apply Loop subdivision to the trashbag mesh
and linear midpoint insertion to the ropes, updating the trajectory NPZ to match.

Three FEM entities are subdivided together and written into a single output NPZ:

  Entity       | Input mesh         | In NPZ       | Subdivision
  -------------|--------------------|--------------|-----------------------
  trashbag     | Trashbag_rope.glb  | fem_trashbag | Loop subdivision (triangle mesh)
  rope1        | rope1.obj          | fem_rope1    | Linear midpoints (polyline)
  rope2        | rope2.obj          | fem_rope2    | Linear midpoints (polyline)

The trashbag GLB contains two separate geometries which Genesis merges into a single
FEM entity.  We concatenate them here too so the unique-vertex ordering matches
exactly what Genesis sees (2920 unique positions after deduplication).

The rope OBJ files contain only vertex positions (no faces) — Genesis infers
sequential edges 0→1→2→…→N-1 and extrudes them into a tube for rendering.
We insert one midpoint between every consecutive pair of points:
  N_orig points → 2*N_orig - 1 points (e.g. 84 → 167)

Usage
-----
  python subdivide_trashbag_trajectory.py              # defaults
  python subdivide_trashbag_trajectory.py --glb-only  # skip NPZ, just visual test
  python subdivide_trashbag_trajectory.py \\
      --input-npz  DemoAssets/trashbag/trajectory_sharpa.npz \\
      --output-npz DemoAssets/trashbag/trajectory_sharpa_subdiv.npz

After running, add --subdiv flag to replay_trashbag_sharpa_traj.py (to be wired up).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

_REPO  = Path(__file__).resolve().parents[2]
_TRASH = _REPO / "DemoAssets" / "trashbag"

DEFAULT_INPUT_GLB  = str(_TRASH / "Trashbag_rope.glb")
DEFAULT_ROPE1_OBJ  = str(_TRASH / "rope1.obj")
DEFAULT_ROPE2_OBJ  = str(_TRASH / "rope2.obj")
DEFAULT_INPUT_NPZ  = str(_TRASH / "trajectory_sharpa.npz")
DEFAULT_OUTPUT_GLB = str(_TRASH / "Trashbag_rope_subdiv.glb")
DEFAULT_OUTPUT_R1  = str(_TRASH / "rope1_subdiv.obj")
DEFAULT_OUTPUT_R2  = str(_TRASH / "rope2_subdiv.obj")
DEFAULT_OUTPUT_NPZ = str(_TRASH / "trajectory_sharpa_subdiv.npz")


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_trashbag_glb(path: str):
    """Load Trashbag_rope.glb and return a single combined Trimesh + the scene.

    The GLB has two separate geometries.  Genesis merges them into one FEM entity
    by concatenating vertices and faces, then deduplicating by position.  We do
    the same here so the unique-vertex ordering matches the FEM particle ordering.
    """
    import trimesh
    scene = trimesh.load(path, process=False, force="scene")
    geoms = list(scene.geometry.values())
    if not geoms:
        raise ValueError(f"No geometry found in {path}")
    if len(geoms) == 1:
        return geoms[0], scene

    # Concatenate multiple geometries
    combined = trimesh.util.concatenate(geoms)
    return combined, scene


def load_rope_points(path: str) -> tuple[np.ndarray, bool]:
    """Load a rope OBJ and return (vertices (N, 3), is_closed).

    is_closed is True when the OBJ has a line segment from the last vertex back
    to the first vertex (e.g. ``l 84 1``), which Genesis renders as a closed ring.
    """
    pts = []
    line_pairs: list[tuple[int, int]] = []
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if raw.startswith("v "):
                pts.append([float(x) for x in raw.split()[1:4]])
            elif raw.startswith("l "):
                a, b = int(raw.split()[1]), int(raw.split()[2])
                line_pairs.append((a, b))
    n = len(pts)
    is_closed = bool(line_pairs) and (n, 1) in line_pairs
    return np.array(pts, dtype=np.float64), is_closed


# ---------------------------------------------------------------------------
# Genesis vertex deduplication (matches fem_entity._merge_elements exactly)
# ---------------------------------------------------------------------------

def unique_mesh(vertices: np.ndarray, faces: np.ndarray):
    """Return (unique_verts, unique_faces, inv, buf_indices) with Genesis ordering."""
    quantized = np.round(vertices * 1e8).astype(np.int64)
    _, unique_idx, remap = np.unique(quantized, axis=0,
                                     return_index=True, return_inverse=True)
    sorted_order = np.argsort(unique_idx)
    rank = np.empty_like(sorted_order)
    rank[sorted_order] = np.arange(len(sorted_order))
    inv = rank[remap]
    buffer_indices = np.sort(unique_idx)
    unique_verts = vertices[buffer_indices]
    unique_faces = inv[faces].astype(np.int32)
    return unique_verts.astype(np.float64), unique_faces, inv, buffer_indices


# ---------------------------------------------------------------------------
# Loop subdivision (triangle mesh — for trashbag)
# ---------------------------------------------------------------------------

def build_loop_subdivision(unique_verts: np.ndarray, unique_faces: np.ndarray):
    """Build Loop subdivision matrix S and new face connectivity.

    Returns (S: CSR (n_new, n_orig), new_faces: (4*F, 3), unique_edges: list)
    """
    n = len(unique_verts)

    edge_count: Counter = Counter()
    edge_opposite: dict[tuple, list] = defaultdict(list)
    for f in unique_faces:
        for i in range(3):
            a, b, c = int(f[i]), int(f[(i+1)%3]), int(f[(i+2)%3])
            key = (min(a, b), max(a, b))
            edge_count[key] += 1
            edge_opposite[key].append(c)

    boundary_edges = {e for e, cnt in edge_count.items() if cnt == 1}

    neighbors:      dict[int, list] = defaultdict(list)
    bdry_neighbors: dict[int, list] = defaultdict(list)
    for (a, b), cnt in edge_count.items():
        neighbors[a].append(b);  neighbors[b].append(a)
        if cnt == 1:
            bdry_neighbors[a].append(b); bdry_neighbors[b].append(a)

    is_boundary = np.zeros(n, dtype=bool)
    for v in bdry_neighbors:
        is_boundary[v] = True

    unique_edges = sorted(edge_count.keys())
    n_edges = len(unique_edges)
    edge_to_idx = {e: i for i, e in enumerate(unique_edges)}
    n_new = n + n_edges

    S = lil_matrix((n_new, n), dtype=np.float64)

    # Vertex points
    for i in range(n):
        nbs = neighbors[i]
        k = len(nbs)
        if k == 0:
            S[i, i] = 1.0
            continue
        if is_boundary[i]:
            bnbs = bdry_neighbors[i]
            if len(bnbs) >= 2:
                S[i, i] = 0.75; S[i, bnbs[0]] = 0.125; S[i, bnbs[1]] = 0.125
            else:
                S[i, i] = 1.0
        else:
            cos_val = np.cos(2.0 * np.pi / k)
            beta = (1.0 / k) * (5.0/8.0 - (3.0/8.0 + 0.25*cos_val)**2)
            S[i, i] = 1.0 - k * beta
            for nb in nbs:
                S[i, nb] = beta

    # Edge points
    for idx, (a, b) in enumerate(unique_edges):
        row = n + idx
        if (a, b) in boundary_edges:
            S[row, a] = 0.5; S[row, b] = 0.5
        else:
            opp = edge_opposite[(a, b)]
            S[row, a] = 3.0/8.0; S[row, b] = 3.0/8.0
            if len(opp) >= 2:
                S[row, opp[0]] = 1.0/8.0; S[row, opp[1]] = 1.0/8.0
            elif len(opp) == 1:
                S[row, opp[0]] = 1.0/4.0

    # New faces
    new_faces_list = []
    for f in unique_faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        eab = n + edge_to_idx[(min(a,b), max(a,b))]
        ebc = n + edge_to_idx[(min(b,c), max(b,c))]
        eca = n + edge_to_idx[(min(c,a), max(c,a))]
        new_faces_list.extend([[a,eab,eca],[eab,b,ebc],[eca,ebc,c],[eab,ebc,eca]])

    return csr_matrix(S), np.array(new_faces_list, dtype=np.int32), unique_edges


# ---------------------------------------------------------------------------
# Linear midpoint subdivision (polyline — for ropes)
# ---------------------------------------------------------------------------

def build_rope_subdivision(n_pts: int):
    """Build linear midpoint-insertion matrix for an ordered polyline of n_pts.

    Inserts one midpoint between every consecutive pair:
      n_pts → 2*n_pts - 1

    The new ordering interleaves originals and midpoints:
      [p0, mid(0,1), p1, mid(1,2), p2, ..., p_{n-1}]

    Genesis infers sequential edges from point order, so this ordering
    gives a smooth curve when extruded into a tube.

    Returns S: CSR matrix (n_new, n_orig)
    """
    n_new = 2 * n_pts - 1
    S = lil_matrix((n_new, n_pts), dtype=np.float64)
    for i in range(n_pts):
        S[2 * i, i] = 1.0                             # original point
    for i in range(n_pts - 1):
        S[2*i + 1, i]   = 0.5                         # midpoint
        S[2*i + 1, i+1] = 0.5
    return csr_matrix(S)


# ---------------------------------------------------------------------------
# GLB / OBJ export
# ---------------------------------------------------------------------------

def save_trashbag_glb(new_verts: np.ndarray, new_faces: np.ndarray,
                      original_scene, out_path: str) -> None:
    """Export subdivided trashbag as a single-geometry GLB with smooth normals.

    The replay script overrides the surface with gs.surfaces.Plastic() so the
    original texture is not used — we export geometry + smooth normals only.

    The first geometry's PBRMaterial is preserved verbatim so Genesis reads the
    correct baseColorFactor (including alpha) and alphaMode (BLEND).  Using
    ColorVisuals instead would drop these and make the bag appear fully opaque.
    """
    import trimesh
    from trimesh.visual.texture import TextureVisuals

    out_mesh = trimesh.Trimesh(vertices=new_verts.astype(np.float32),
                               faces=new_faces, process=False)
    # Bake smooth per-vertex normals
    out_mesh.vertex_normals  # trigger computation

    # Carry over the first geometry's PBRMaterial (no texture, no UVs needed).
    # This preserves baseColorFactor alpha=128 and alphaMode=BLEND which give
    # the bag its semi-transparent appearance in Genesis.
    orig_geoms = list(original_scene.geometry.values())
    try:
        orig_material = orig_geoms[0].visual.material
        out_mesh.visual = TextureVisuals(material=orig_material)
    except Exception:
        pass

    # Write as a fresh scene (single geometry)
    out_scene = trimesh.Scene(geometry={"Trashbag_subdiv": out_mesh})
    out_scene.export(out_path)
    print(f"  Saved GLB  → {out_path}")


def save_rope_obj(pts: np.ndarray, out_path: str, closed: bool = False) -> None:
    """Write ordered point positions as an OBJ file with explicit line connectivity.

    The ``l`` entries mirror what Genesis reads to build the tube topology.
    When *closed* is True a final ``l N 1`` entry is added to form a closed ring,
    matching the original rope OBJ which has ``l 84 1`` as its last line.
    """
    n = len(pts)
    lines = [f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n" for p in pts]
    for i in range(1, n):
        lines.append(f"l {i} {i + 1}\n")
    if closed:
        lines.append(f"l {n} 1\n")
    with open(out_path, "w") as f:
        f.writelines(lines)
    shape = "closed ring" if closed else "open polyline"
    print(f"  Saved OBJ  → {out_path}  ({n} pts, {shape})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-glb",  default=DEFAULT_INPUT_GLB)
    parser.add_argument("--rope1-obj",  default=DEFAULT_ROPE1_OBJ)
    parser.add_argument("--rope2-obj",  default=DEFAULT_ROPE2_OBJ)
    parser.add_argument("--input-npz",  default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--output-glb", default=DEFAULT_OUTPUT_GLB)
    parser.add_argument("--output-r1",  default=DEFAULT_OUTPUT_R1)
    parser.add_argument("--output-r2",  default=DEFAULT_OUTPUT_R2)
    parser.add_argument("--output-npz", default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--iterations", type=int, default=1,
                        help="Loop subdivision levels for trashbag (default: 1)")
    parser.add_argument("--glb-only", action="store_true",
                        help="Export meshes only, skip NPZ processing")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Trashbag GLB
    # ------------------------------------------------------------------
    print(f"\nLoading trashbag GLB  ← {args.input_glb}")
    bag_mesh, bag_scene = load_trashbag_glb(args.input_glb)
    verts_full = np.array(bag_mesh.vertices, dtype=np.float64)
    faces_full = np.array(bag_mesh.faces, dtype=np.int32)
    print(f"  Combined mesh: {len(verts_full)} buffer verts, {len(faces_full)} faces")

    uverts, ufaces, _, _ = unique_mesh(verts_full, faces_full)
    print(f"  Unique positions: {len(uverts)}")

    print(f"\nBuilding Loop subdivision matrix ({args.iterations} level(s))…")
    t0 = time.perf_counter()
    cur_v, cur_f = uverts.copy(), ufaces.copy()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    S_lvl, cur_f, _ = build_loop_subdivision(cur_v, cur_f)
    cur_v = np.array(S_lvl @ cur_v)
    S_bag = S_lvl
    for _ in range(args.iterations - 1):
        S_lvl, cur_f, _ = build_loop_subdivision(cur_v, cur_f)
        cur_v = np.array(S_lvl @ cur_v)
        S_bag = S_lvl @ S_bag
    print(f"  Done in {time.perf_counter()-t0:.1f}s  →  "
          f"{len(cur_v)} verts, {len(cur_f)} faces")

    print("\nExporting subdivided trashbag GLB…")
    save_trashbag_glb(cur_v, cur_f, bag_scene, args.output_glb)

    # ------------------------------------------------------------------
    # Ropes
    # ------------------------------------------------------------------
    print(f"\nLoading ropes…")
    r1_pts, r1_closed = load_rope_points(args.rope1_obj)
    r2_pts, r2_closed = load_rope_points(args.rope2_obj)
    n_rope = len(r1_pts)
    assert len(r2_pts) == n_rope, "rope1 and rope2 must have the same point count"
    print(f"  rope1: {n_rope} pts (closed={r1_closed}),  rope2: {len(r2_pts)} pts (closed={r2_closed})")

    S_rope = build_rope_subdivision(n_rope)
    S_rope_dense = np.array(S_rope.todense(), dtype=np.float32)

    r1_subdiv = (S_rope_dense @ r1_pts.astype(np.float32)).astype(np.float32)
    r2_subdiv = (S_rope_dense @ r2_pts.astype(np.float32)).astype(np.float32)
    print(f"  Subdivided: {n_rope} → {len(r1_subdiv)} pts per rope")

    save_rope_obj(r1_subdiv, args.output_r1, closed=r1_closed)
    save_rope_obj(r2_subdiv, args.output_r2, closed=r2_closed)

    if args.glb_only:
        print("\nDone (--glb-only, skipping NPZ).")
        return

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------
    print(f"\nLoading NPZ  ← {args.input_npz}")
    traj = np.load(args.input_npz)
    print(f"  Keys: {list(traj.files)}")

    fem_bag   = traj["fem_trashbag"]   # (n_frames, 2920, 3)
    fem_rope1 = traj["fem_rope1"]      # (n_frames, 84, 3)
    fem_rope2 = traj["fem_rope2"]      # (n_frames, 84, 3)
    n_frames  = len(fem_bag)

    # Verify counts match subdivision inputs
    n_bag_orig, n_rope_orig = fem_bag.shape[1], fem_rope1.shape[1]
    if n_bag_orig != len(uverts):
        raise ValueError(
            f"fem_trashbag has {n_bag_orig} verts/frame but unique mesh has "
            f"{len(uverts)} — mismatch. Check that the right GLB is used.")
    if n_rope_orig != n_rope:
        raise ValueError(
            f"fem_rope1 has {n_rope_orig} verts/frame but rope OBJ has {n_rope} pts.")

    n_bag_new  = len(cur_v)
    n_rope_new = len(r1_subdiv)
    print(f"  Trashbag: {n_frames}f × {n_bag_orig} → {n_frames}f × {n_bag_new}")
    print(f"  Ropes   : {n_frames}f × {n_rope_orig} → {n_frames}f × {n_rope_new}")

    t0 = time.perf_counter()
    S_bag_dense = np.array(S_bag.todense(), dtype=np.float32)

    print("  Subdividing trashbag frames…")
    fem_bag_sub = np.einsum("ij,fjk->fik", S_bag_dense, fem_bag.astype(np.float32))

    print("  Subdividing rope frames…")
    fem_r1_sub = np.einsum("ij,fjk->fik", S_rope_dense, fem_rope1.astype(np.float32))
    fem_r2_sub = np.einsum("ij,fjk->fik", S_rope_dense, fem_rope2.astype(np.float32))

    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # Copy everything else unchanged, replace the three FEM arrays
    out_arrays = {k: traj[k] for k in traj.files
                  if k not in ("fem_trashbag", "fem_rope1", "fem_rope2")}
    out_arrays["fem_trashbag"] = fem_bag_sub
    out_arrays["fem_rope1"]    = fem_r1_sub
    out_arrays["fem_rope2"]    = fem_r2_sub

    np.savez(args.output_npz, **out_arrays)
    print(f"  Saved NPZ  → {args.output_npz}")

    print("\nDone.")
    print(f"  Trashbag GLB : {args.output_glb}")
    print(f"  Rope 1 OBJ   : {args.output_r1}")
    print(f"  Rope 2 OBJ   : {args.output_r2}")
    print(f"  NPZ          : {args.output_npz}")
    print("\nWire up replay_trashbag_sharpa_traj.py with a --subdiv flag pointing to these assets.")


if __name__ == "__main__":
    main()
