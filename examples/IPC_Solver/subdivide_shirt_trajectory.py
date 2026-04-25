"""
subdivide_shirt_trajectory.py — Apply Loop subdivision to the shirt mesh and
trajectory together, producing a matched GLB + NPZ pair for higher-resolution replay.

replay_hanger_sharpa_traj.py drives cloth positions via set_position() each frame.
For that to work the GLB unique-vertex count must equal the NPZ positions-per-frame
count.  This script ensures they match by building the subdivision matrix S from the
mesh topology and applying it to BOTH assets:

  new GLB : genesis_shirt_subdiv.glb  (~8 500 unique verts, ~16 600 faces @ level 1)
  new NPZ : trajectory_sharpa_subdiv.npz  (same shape as original but denser)

The matrix S is built from the unique-position mesh (UV seam duplicates collapsed to
single vertices so they move together, matching the FEM solver's view of the mesh).

Usage
-----
  # 1 level of Loop subdivision (default)
  python subdivide_shirt_trajectory.py

  # 2 levels (~33 000 verts — heavier but very smooth)
  python subdivide_shirt_trajectory.py --iterations 2

  # Test the GLB only (no NPZ processing)
  python subdivide_shirt_trajectory.py --glb-only

  # Custom paths
  python subdivide_shirt_trajectory.py \\
      --input-glb ../../DemoAssets/coat_hanger/genesis_shirt.glb \\
      --input-npz ../../DemoAssets/coat_hanger/trajectory_sharpa.npz \\
      --output-glb ../../DemoAssets/coat_hanger/genesis_shirt_subdiv.glb \\
      --output-npz ../../DemoAssets/coat_hanger/trajectory_sharpa_subdiv.npz
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

_REPO  = Path(__file__).resolve().parents[2]
_HANGER = _REPO / "DemoAssets" / "coat_hanger"

DEFAULT_INPUT_GLB  = str(_HANGER / "genesis_shirt.glb")
DEFAULT_INPUT_NPZ  = str(_HANGER / "trajectory_sharpa.npz")
DEFAULT_OUTPUT_GLB = str(_HANGER / "genesis_shirt_subdiv.glb")
DEFAULT_OUTPUT_NPZ = str(_HANGER / "trajectory_sharpa_subdiv.npz")


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def load_glb(path: str):
    """Return (trimesh.Trimesh, scene) — scene carries name/transform/material."""
    import trimesh
    loaded = trimesh.load(path, process=False, force="scene")
    geoms = list(loaded.geometry.values())
    if len(geoms) != 1:
        raise ValueError(f"Expected exactly 1 mesh in GLB, found {len(geoms)}")
    return geoms[0], loaded


def unique_mesh(vertices: np.ndarray, faces: np.ndarray):
    """Collapse UV-seam duplicate positions using Genesis's exact deduplication.

    Replicates fem_entity.py _merge_elements() so the unique vertex ordering
    here matches the FEM particle ordering Genesis assigns at load time:
      - quantize with round(pos * 1e8)
      - np.unique for first-occurrence indices
      - sort first occurrences by buffer position → first-occurrence order

    Returns (unique_verts, unique_faces, inv) where inv[i] maps original buffer
    vertex index i to its Genesis particle index.
    """
    quantized = np.round(vertices * 1e8).astype(np.int64)
    _, unique_idx, remap = np.unique(quantized, axis=0,
                                     return_index=True, return_inverse=True)
    sorted_order = np.argsort(unique_idx)
    rank = np.empty_like(sorted_order)
    rank[sorted_order] = np.arange(len(sorted_order))
    inv = rank[remap]                                # particle index per buffer vertex
    buffer_indices = np.sort(unique_idx)             # original buffer index per unique vertex
    unique_verts = vertices[buffer_indices]          # verts in first-occurrence order
    unique_faces = inv[faces].astype(np.int32)
    return unique_verts.astype(np.float64), unique_faces, inv, buffer_indices


# ---------------------------------------------------------------------------
# Loop subdivision matrix
# ---------------------------------------------------------------------------

def build_loop_subdivision(unique_verts: np.ndarray, unique_faces: np.ndarray,
                           identity: bool = False):
    """Build Loop subdivision matrix S and new face connectivity.

    Returns
    -------
    S : scipy CSR matrix, shape (n_new, n_orig)
        new_verts = S @ old_verts  (works for any per-vertex data: positions, etc.)
    new_faces : np.ndarray, shape (4 * n_faces, 3)
        Face connectivity after one level of Loop subdivision.
    """
    n = len(unique_verts)
    faces = unique_faces

    # ------------------------------------------------------------------
    # 1. Build edge table: edge → (face count, list of opposite vertices)
    # ------------------------------------------------------------------
    edge_count: Counter = Counter()
    edge_opposite: dict[tuple, list[int]] = defaultdict(list)

    for f in faces:
        for i in range(3):
            a, b, c = int(f[i]), int(f[(i + 1) % 3]), int(f[(i + 2) % 3])
            key = (min(a, b), max(a, b))
            edge_count[key] += 1
            edge_opposite[key].append(c)

    boundary_edge_set: set[tuple] = {e for e, cnt in edge_count.items() if cnt == 1}

    # ------------------------------------------------------------------
    # 2. Build vertex adjacency and boundary vertex mask
    # ------------------------------------------------------------------
    neighbors: dict[int, list[int]] = defaultdict(list)
    bdry_neighbors: dict[int, list[int]] = defaultdict(list)

    for (a, b), cnt in edge_count.items():
        neighbors[a].append(b)
        neighbors[b].append(a)
        if cnt == 1:
            bdry_neighbors[a].append(b)
            bdry_neighbors[b].append(a)

    is_boundary = np.zeros(n, dtype=bool)
    for v in bdry_neighbors:
        is_boundary[v] = True

    # ------------------------------------------------------------------
    # 3. Ordered edge list for stable indexing
    # ------------------------------------------------------------------
    unique_edges = sorted(edge_count.keys())
    n_edges = len(unique_edges)
    edge_to_idx = {e: i for i, e in enumerate(unique_edges)}
    n_new = n + n_edges

    # ------------------------------------------------------------------
    # 4. Fill subdivision matrix
    # ------------------------------------------------------------------
    S = lil_matrix((n_new, n), dtype=np.float64)

    # --- Vertex points (rows 0 … n-1) ---
    for i in range(n):
        nbs = neighbors[i]
        k = len(nbs)
        if k == 0 or identity:
            # Identity mode: original vertices stay exactly in place
            S[i, i] = 1.0
            continue

        if is_boundary[i]:
            bdry_nbs = bdry_neighbors[i]
            if len(bdry_nbs) >= 2:
                S[i, i]           = 0.75
                S[i, bdry_nbs[0]] = 0.125
                S[i, bdry_nbs[1]] = 0.125
            else:
                # Corner / isolated boundary vertex — keep in place
                S[i, i] = 1.0
        else:
            # Loop interior formula
            cos_val = np.cos(2.0 * np.pi / k)
            beta = (1.0 / k) * (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * cos_val) ** 2)
            S[i, i] = 1.0 - k * beta
            for nb in nbs:
                S[i, nb] = beta

    # --- Edge points (rows n … n+n_edges-1) ---
    for idx, (a, b) in enumerate(unique_edges):
        row = n + idx
        if (a, b) in boundary_edge_set:
            S[row, a] = 0.5
            S[row, b] = 0.5
        else:
            opp = edge_opposite[(a, b)]
            S[row, a] = 3.0 / 8.0
            S[row, b] = 3.0 / 8.0
            if len(opp) >= 2:
                S[row, opp[0]] = 1.0 / 8.0
                S[row, opp[1]] = 1.0 / 8.0
            elif len(opp) == 1:
                # Degenerate: only one adjacent face — treat as boundary midpoint
                S[row, opp[0]] = 1.0 / 4.0

    # ------------------------------------------------------------------
    # 5. New face connectivity
    # ------------------------------------------------------------------
    new_faces_list: list[list[int]] = []
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        eab = n + edge_to_idx[(min(a, b), max(a, b))]
        ebc = n + edge_to_idx[(min(b, c), max(b, c))]
        eca = n + edge_to_idx[(min(c, a), max(c, a))]
        new_faces_list.extend([
            [a,   eab, eca],
            [eab, b,   ebc],
            [eca, ebc, c  ],
            [eab, ebc, eca],   # centre triangle
        ])

    new_faces = np.array(new_faces_list, dtype=np.int32)
    return csr_matrix(S), new_faces, unique_edges


# ---------------------------------------------------------------------------
# GLB export
# ---------------------------------------------------------------------------

def save_glb(new_verts: np.ndarray, new_faces: np.ndarray,
             original_mesh, scene, out_path: str,
             buf_indices: np.ndarray | None = None,
             unique_edges: list | None = None) -> None:
    """Export the subdivided mesh as a GLB, preserving texture via UV subdivision.

    UV coordinates are subdivided linearly (midpoint for edge points, passthrough
    for original vertices) using the original buffer's per-vertex UVs.  This keeps
    the texture intact at the cost of slight inaccuracy at UV seam edges.

    buf_indices : shape (n_orig,) — original buffer index of each unique vertex
    unique_edges : list of (a, b) — edge list in S-row order (rows n_orig onward)
    """
    import trimesh
    from trimesh.visual import ColorVisuals
    from trimesh.visual.texture import TextureVisuals

    # Compute smooth per-vertex normals
    tm = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    normals = np.array(tm.vertex_normals, dtype=np.float32)

    out_mesh = trimesh.Trimesh(
        vertices=new_verts.astype(np.float32),
        faces=new_faces,
        vertex_normals=normals,
        process=False,
    )

    # ------------------------------------------------------------------
    # Try to subdivide UV coordinates and carry over the texture
    # ------------------------------------------------------------------
    uv_ok = False
    if buf_indices is not None and unique_edges is not None:
        try:
            orig_uvs = np.array(original_mesh.visual.uv, dtype=np.float32)  # (n_buf, 2)
            n_orig = len(buf_indices)
            n_new  = n_orig + len(unique_edges)
            new_uvs = np.zeros((n_new, 2), dtype=np.float32)
            # Original vertices: take UV from their first-occurrence buffer position
            new_uvs[:n_orig] = orig_uvs[buf_indices]
            # Edge midpoints: linear interpolation of endpoint UVs
            for idx, (a, b) in enumerate(unique_edges):
                new_uvs[n_orig + idx] = 0.5 * (new_uvs[a] + new_uvs[b])
            # Carry over the texture material (image + PBR params)
            orig_material = original_mesh.visual.material
            out_mesh.visual = TextureVisuals(uv=new_uvs, material=orig_material)
            uv_ok = True
        except Exception as e:
            print(f"  Warning: UV subdivision failed ({e}), falling back to vertex colour")

    if not uv_ok:
        try:
            color = original_mesh.visual.material.main_color
            out_mesh.visual = ColorVisuals(
                mesh=out_mesh,
                vertex_colors=np.tile(color, (len(new_verts), 1)),
            )
        except Exception:
            pass

    # Re-use the scene graph so the GLB preserves the original node name / transform
    name = list(scene.geometry.keys())[0]
    scene.geometry[name] = out_mesh
    scene.export(out_path)
    print(f"  Saved GLB  → {out_path}  (UVs: {'subdivided' if uv_ok else 'fallback colour'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-glb",  default=DEFAULT_INPUT_GLB)
    parser.add_argument("--input-npz",  default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--output-glb", default=DEFAULT_OUTPUT_GLB)
    parser.add_argument("--output-npz", default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--iterations", type=int, default=1,
                        help="Levels of Loop subdivision (default: 1)")
    parser.add_argument("--glb-only", action="store_true",
                        help="Skip NPZ processing (useful for visual testing)")
    parser.add_argument("--npz-only", action="store_true",
                        help="Skip GLB export (reuse an existing subdivided GLB)")
    parser.add_argument("--identity", action="store_true",
                        help="Pipeline test: original vertices stay in place, only edge "
                             "midpoints added (simple 50/50 average). Verifies that the "
                             "GLB+NPZ pipeline is correct before trusting the Loop stencil.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load original GLB
    # ------------------------------------------------------------------
    print(f"\nLoading GLB  ← {args.input_glb}")
    mesh, scene = load_glb(args.input_glb)
    verts_full = np.array(mesh.vertices, dtype=np.float64)
    faces_full = np.array(mesh.faces,    dtype=np.int32)

    n_full   = len(verts_full)
    n_faces  = len(faces_full)
    unique_v, unique_f, _inv, buf_indices = unique_mesh(verts_full, faces_full)
    n_unique = len(unique_v)

    print(f"  Full mesh : {n_full} verts, {n_faces} faces")
    print(f"  Unique pos: {n_unique} (UV seam duplicates collapsed)")

    # ------------------------------------------------------------------
    # Build subdivision matrix (iteratively for multiple levels)
    # ------------------------------------------------------------------
    mode_label = "identity (pipeline test)" if args.identity else f"Loop ({args.iterations} level(s))"
    print(f"\nBuilding subdivision matrix — {mode_label}…")
    t0 = time.perf_counter()

    cur_verts = unique_v.copy()
    cur_faces = unique_f.copy()
    S_combined = None  # accumulated S: new_verts = S_combined @ original_verts

    n_iters = 1 if args.identity else args.iterations
    last_unique_edges = None
    for _ in range(n_iters):
        S_level, cur_faces, last_unique_edges = build_loop_subdivision(
            cur_verts, cur_faces, identity=args.identity)
        cur_verts = np.array(S_level @ cur_verts)
        if S_combined is None:
            S_combined = S_level
        else:
            S_combined = S_level @ S_combined  # chain: new = S_level @ S_prev @ orig

    if S_combined is None:
        raise ValueError("--iterations must be >= 1")

    n_subdiv = len(cur_verts)
    n_faces_subdiv = len(cur_faces)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    print(f"  Subdivided: {n_subdiv} unique verts, {n_faces_subdiv} faces")

    # ------------------------------------------------------------------
    # Export GLB
    # ------------------------------------------------------------------
    if not args.npz_only:
        print(f"\nExporting GLB…")
        save_glb(cur_verts, cur_faces, mesh, scene, args.output_glb,
                 buf_indices=buf_indices, unique_edges=last_unique_edges)

    if args.glb_only:
        print("\nDone (--glb-only, skipping NPZ).")
        return

    # ------------------------------------------------------------------
    # Process NPZ
    # ------------------------------------------------------------------
    print(f"\nLoading NPZ  ← {args.input_npz}")
    traj = np.load(args.input_npz)
    fem_shirt = traj["fem_shirt"]          # (n_frames, 2177, 3)
    n_frames, n_verts_orig, _ = fem_shirt.shape

    if n_verts_orig != n_unique:
        raise ValueError(
            f"NPZ fem_shirt has {n_verts_orig} verts/frame but the unique mesh has "
            f"{n_unique} unique positions — they must match."
        )

    print(f"  {n_frames} frames × {n_verts_orig} verts → {n_frames} frames × {n_subdiv} verts")

    t0 = time.perf_counter()
    S_dense = np.array(S_combined.todense(), dtype=np.float32)  # (n_subdiv, n_orig)
    fem_shirt_subdiv = np.einsum("ij,fjk->fik", S_dense, fem_shirt)
    # shape: (n_frames, n_subdiv, 3)

    elapsed = time.perf_counter() - t0
    print(f"  Subdivision applied in {elapsed:.1f}s")
    print(f"  Output shape: {fem_shirt_subdiv.shape}")

    # Copy all other arrays verbatim, replace fem_shirt
    out_arrays = {k: traj[k] for k in traj.files if k != "fem_shirt"}
    out_arrays["fem_shirt"] = fem_shirt_subdiv.astype(np.float32)

    np.savez(args.output_npz, **out_arrays)
    print(f"  Saved NPZ  → {args.output_npz}")

    print("\nDone.")
    print(f"  GLB : {args.output_glb}")
    print(f"  NPZ : {args.output_npz}")
    print(f"\nTo use in replay:")
    print(f"  Edit SHIRT_GLB and DEFAULT_TRAJ in replay_hanger_sharpa_traj.py, or")
    print(f"  pass  --traj {args.output_npz}")
    print(f"  and set SHIRT_GLB = '{args.output_glb}' in the script.")


if __name__ == "__main__":
    main()
