"""
Decimate paper_plane mesh while preserving crease edges.

Strategy: split the mesh along crease edges (turning them into boundary edges),
decimate with preserve_border=True, then stitch the boundary back.

Usage:
    python decimate_paper_plane.py [input.obj] [--target N] [--output out.obj]
"""

import argparse
import os
import sys

import numpy as np

_FOLD_DIR = os.path.dirname(__file__)
if _FOLD_DIR not in sys.path:
    sys.path.insert(0, _FOLD_DIR)

import paper_plane_2_boundary_crease_debug_demo as boundary_debug
from uipc import view
import pyfqmr
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", nargs="?", default="paper_plane_2_coarse.obj")
    parser.add_argument("--target", type=int, default=3000, help="Target face count")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    mesh_path = boundary_debug.resolve_origami_asset_path(args.mesh)
    print(f"Loading: {mesh_path}")

    # Load and detect creases
    uipc_mesh = boundary_debug.load_paper_plane_mesh(args.mesh)
    diag = boundary_debug.detect_boundary_creases(uipc_mesh, args.mesh)

    positions = np.array(view(uipc_mesh.positions()), copy=True).reshape(-1, 3)
    triangles = np.array(uipc_mesh.triangles().topo().view(), copy=True).reshape(-1, 3)
    n_verts = len(positions)
    n_faces = len(triangles)

    # Collect crease vertex IDs
    chains = list({id(v): v for v in diag["crease_lookup"].values()}.values())
    crease_verts = set()
    crease_edges = set()
    for chain in chains:
        vids = chain.vertex_ids
        crease_verts.update(vids)
        for i in range(len(vids) - 1):
            crease_edges.add(tuple(sorted((vids[i], vids[i + 1]))))

    print(f"Original: {n_verts} verts, {n_faces} faces")
    print(f"Crease: {len(crease_verts)} verts, {len(crease_edges)} edges")

    # Thin crease chains: keep only every Nth interior vertex.
    # Since crease vertices are perfectly collinear, removing interior points
    # doesn't change the crease geometry.
    CREASE_KEEP_RATIO = 0.5
    collapse_map = {}
    for chain in chains:
        vids = list(chain.vertex_ids)
        if len(vids) <= 3:
            continue
        interior = vids[1:-1]
        n_keep = max(1, int(len(interior) * CREASE_KEEP_RATIO))
        keep_idx = set(np.linspace(0, len(interior) - 1, n_keep, dtype=int))
        for i, vid in enumerate(interior):
            if i not in keep_idx:
                nearest = min(keep_idx, key=lambda k: abs(k - i))
                collapse_map[vid] = interior[nearest]

    if collapse_map:
        for fi in range(len(triangles)):
            for k in range(3):
                v = int(triangles[fi, k])
                if v in collapse_map:
                    triangles[fi, k] = collapse_map[v]
        valid = np.array([len(set(t)) == 3 for t in triangles])
        triangles = triangles[valid]
        used = np.unique(triangles)
        remap = np.full(n_verts, -1, dtype=np.int32)
        remap[used] = np.arange(len(used))
        positions = positions[used]
        triangles = remap[triangles]
        n_verts, n_faces = len(positions), len(triangles)
        crease_verts = {int(remap[v]) for v in crease_verts if v < len(remap) and remap[v] >= 0}
        crease_edges = set()
        for chain in chains:
            remapped = [int(remap[v]) for v in chain.vertex_ids if v < len(remap) and remap[v] >= 0]
            remapped = list(dict.fromkeys(remapped))
            for i in range(len(remapped) - 1):
                crease_edges.add(tuple(sorted((remapped[i], remapped[i + 1]))))
        print(f"After crease thinning: {n_verts} verts, {n_faces} faces, {len(crease_verts)} crease verts")

    # Split mesh along crease edges by duplicating crease vertices
    # For each crease vertex, create a duplicate. Faces on one side of the
    # crease use the original, faces on the other side use the duplicate.
    # This turns crease edges into boundary edges.

    # Build edge-to-face adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, tri in enumerate(triangles):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            edge = tuple(sorted((a, b)))
            edge_faces.setdefault(edge, []).append(fi)

    # For each crease edge, split: one adjacent face keeps original verts,
    # the other gets duplicated verts.
    # Strategy: flood-fill from one side of each crease to determine which
    # faces are on which side.

    # Build face adjacency (faces sharing non-crease edges)
    face_adj: dict[int, list[int]] = {fi: [] for fi in range(n_faces)}
    for edge, faces in edge_faces.items():
        if edge in crease_edges:
            continue
        if len(faces) == 2:
            face_adj[faces[0]].append(faces[1])
            face_adj[faces[1]].append(faces[0])

    # Connected components (faces separated by crease edges)
    visited = np.full(n_faces, -1, dtype=np.int32)
    component = 0
    for start in range(n_faces):
        if visited[start] >= 0:
            continue
        stack = [start]
        visited[start] = component
        while stack:
            f = stack.pop()
            for nb in face_adj[f]:
                if visited[nb] < 0:
                    visited[nb] = component
                    stack.append(nb)
        component += 1
    print(f"Face components (separated by creases): {component}")

    # For each crease vertex, determine which component is "primary"
    # (keeps original vertex ID) and which gets a duplicate.
    new_positions = list(positions)
    new_triangles = triangles.copy()
    vert_remap: dict[tuple[int, int], int] = {}

    for vid in crease_verts:
        # Find all faces using this vertex
        face_indices = [fi for fi, tri in enumerate(triangles) if vid in tri]
        # Group by component
        comp_faces: dict[int, list[int]] = {}
        for fi in face_indices:
            comp_faces.setdefault(visited[fi], []).append(fi)

        if len(comp_faces) <= 1:
            continue

        # First component keeps original vertex
        comps = sorted(comp_faces.keys())
        for comp_id in comps[1:]:
            # Create duplicate vertex
            new_vid = len(new_positions)
            new_positions.append(positions[vid].copy())
            vert_remap[(vid, comp_id)] = new_vid
            # Remap faces in this component
            for fi in comp_faces[comp_id]:
                tri = new_triangles[fi]
                for k in range(3):
                    if tri[k] == vid:
                        new_triangles[fi, k] = new_vid

    new_positions = np.array(new_positions, dtype=np.float64)
    print(f"After split: {len(new_positions)} verts, {len(new_triangles)} faces")

    # Decimate with preserve_border=True
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(new_positions, new_triangles)
    simplifier.simplify_mesh(
        target_count=args.target,
        preserve_border=True,
        aggressiveness=5.0,
        verbose=False,
    )
    dec_verts, dec_faces, _ = simplifier.getMesh()
    print(f"After decimate: {len(dec_verts)} verts, {len(dec_faces)} faces")

    # Stitch back: merge duplicate vertices that are at the same position
    # (crease vertices that were split)
    unique_map = {}
    merged_ids = np.arange(len(dec_verts))
    for i in range(len(dec_verts)):
        key = tuple(np.round(dec_verts[i], decimals=10))
        if key in unique_map:
            merged_ids[i] = unique_map[key]
        else:
            unique_map[key] = i

    # Remap faces
    dec_faces_merged = np.array([[merged_ids[v] for v in f] for f in dec_faces])

    # Remove unused vertices
    used = np.unique(dec_faces_merged)
    new_id = np.full(len(dec_verts), -1, dtype=np.int32)
    new_id[used] = np.arange(len(used))
    final_verts = dec_verts[used]
    final_faces = new_id[dec_faces_merged]

    # Remove degenerate faces
    valid = np.array([len(set(f)) == 3 for f in final_faces])
    final_faces = final_faces[valid]

    print(f"After merge: {len(final_verts)} verts, {len(final_faces)} faces")

    # Save
    if args.output is None:
        base = os.path.splitext(os.path.basename(mesh_path))[0]
        out_dir = os.path.dirname(mesh_path)
        args.output = os.path.join(out_dir, f"{base}_decimated.obj")

    m = trimesh.Trimesh(vertices=final_verts, faces=final_faces, process=False)
    m.export(args.output)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
