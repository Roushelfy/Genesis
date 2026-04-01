"""
Segment each G1 STL mesh by sharp-crease boundaries into sub-meshes.

For each STL file, this script:
1. Finds sharp edges (dihedral angle > threshold)
2. Removes sharp edges from the face adjacency graph
3. Flood-fills connected face regions → each region becomes a sub-mesh
4. Exports sub-meshes as separate OBJ files
5. Generates a visualization HTML report showing all regions per link

Usage:
    python segment_stl.py [--angle 80] [--mesh head_link.STL] [--all]
    python segment_stl.py --mesh head_link.STL          # segment one mesh
    python segment_stl.py --all                          # segment all 35 meshes
    python segment_stl.py --mesh head_link.STL --vis     # open polyscope viewer
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


MESH_DIR = Path(__file__).parent / "meshes"
OUTPUT_DIR = Path(__file__).parent / "meshes_segmented"

# The 35 meshes used by g1_29dof_rev_1_0.urdf
G1_MESHES = [
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "logo_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]


def segment_mesh(mesh: trimesh.Trimesh, angle_threshold_deg: float = 80.0):
    """Segment a mesh into regions separated by sharp creases.

    Returns list of face index arrays, one per region, sorted by size (largest first).
    """
    n_faces = len(mesh.faces)

    # Build face adjacency, excluding sharp edges
    sharp_mask = mesh.face_adjacency_angles > np.radians(angle_threshold_deg)
    adj = defaultdict(set)
    for (f1, f2), is_sharp in zip(mesh.face_adjacency, sharp_mask):
        if not is_sharp:
            adj[f1].add(f2)
            adj[f2].add(f1)

    # Flood-fill to find connected regions
    visited = np.zeros(n_faces, dtype=bool)
    regions = []

    for seed in range(n_faces):
        if visited[seed]:
            continue
        # BFS
        region = []
        queue = [seed]
        visited[seed] = True
        while queue:
            f = queue.pop()
            region.append(f)
            for nb in adj[f]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        regions.append(np.array(region, dtype=int))

    # Sort by size, largest first
    regions.sort(key=len, reverse=True)
    return regions


def extract_submesh(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> trimesh.Trimesh:
    """Extract a submesh from face indices, reindexing vertices."""
    faces = mesh.faces[face_indices]
    unique_verts, new_faces = np.unique(faces, return_inverse=True)
    new_faces = new_faces.reshape(-1, 3)
    new_vertices = mesh.vertices[unique_verts]
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)


def random_colors(n: int, seed: int = 42) -> np.ndarray:
    """Generate N visually distinct colors."""
    rng = np.random.RandomState(seed)
    colors = []
    for i in range(n):
        # Use HSV with evenly spaced hues
        hue = (i * 137.508) % 360  # golden angle spacing
        sat = 0.6 + rng.random() * 0.3
        val = 0.7 + rng.random() * 0.25
        # HSV to RGB
        c = hue / 60.0
        x = val * sat * (1 - abs(c % 2 - 1))
        m = val - val * sat
        if c < 1:
            r, g, b = val * sat + m, x + m, m
        elif c < 2:
            r, g, b = x + m, val * sat + m, m
        elif c < 3:
            r, g, b = m, val * sat + m, x + m
        elif c < 4:
            r, g, b = m, x + m, val * sat + m
        elif c < 5:
            r, g, b = x + m, m, val * sat + m
        else:
            r, g, b = val * sat + m, m, x + m
        colors.append([r, g, b])
    return np.array(colors)


def process_one_mesh(
    name: str,
    angle_deg: float,
    min_faces: int = 10,
    visualize: bool = False,
) -> dict:
    """Segment one STL and save sub-meshes. Returns summary dict."""
    stl_path = MESH_DIR / f"{name}.STL"
    if not stl_path.exists():
        print(f"  SKIP {name}: {stl_path} not found")
        return {}

    mesh = trimesh.load(stl_path)
    regions = segment_mesh(mesh, angle_deg)

    # Merge tiny regions (< min_faces) into nearest large region
    large_regions = [r for r in regions if len(r) >= min_faces]
    small_regions = [r for r in regions if len(r) < min_faces]

    if small_regions:
        # For each small region, find which large region shares the most adjacency
        # Simple approach: merge into the large region whose faces are adjacent
        face_to_region = np.full(len(mesh.faces), -1, dtype=int)
        for i, r in enumerate(large_regions):
            face_to_region[r] = i

        for small_r in small_regions:
            # Find which large region is most adjacent
            neighbor_counts = defaultdict(int)
            for f in small_r:
                for nb_pair in mesh.face_adjacency:
                    if nb_pair[0] == f and face_to_region[nb_pair[1]] >= 0:
                        neighbor_counts[face_to_region[nb_pair[1]]] += 1
                    elif nb_pair[1] == f and face_to_region[nb_pair[0]] >= 0:
                        neighbor_counts[face_to_region[nb_pair[0]]] += 1
            if neighbor_counts:
                best = max(neighbor_counts, key=neighbor_counts.get)
                large_regions[best] = np.concatenate([large_regions[best], small_r])
                for f in small_r:
                    face_to_region[f] = best

        regions = large_regions

    # Re-sort by size
    regions.sort(key=len, reverse=True)

    # Save sub-meshes
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = random_colors(len(regions))
    summary = {
        "name": name,
        "total_faces": len(mesh.faces),
        "total_verts": len(mesh.vertices),
        "n_regions": len(regions),
        "regions": [],
    }

    for i, region_faces in enumerate(regions):
        submesh = extract_submesh(mesh, region_faces)
        out_path = out_dir / f"{name}_part{i:02d}.obj"
        submesh.export(str(out_path))
        summary["regions"].append(
            {
                "part": i,
                "faces": len(region_faces),
                "verts": len(submesh.vertices),
                "file": str(out_path.relative_to(OUTPUT_DIR.parent)),
                "color_rgb": colors[i].tolist(),
            }
        )

    # Also save a colored version for quick visual check
    face_colors = np.ones((len(mesh.faces), 4)) * 0.5
    for i, region_faces in enumerate(regions):
        face_colors[region_faces, :3] = colors[i]
        face_colors[region_faces, 3] = 1.0
    mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)
    colored_path = out_dir / f"{name}_colored.ply"
    mesh.export(str(colored_path))

    if visualize:
        try:
            import polyscope as ps

            ps.init()
            ps_mesh = ps.register_surface_mesh(name, mesh.vertices, mesh.faces)
            # Color per face
            region_ids = np.zeros(len(mesh.faces), dtype=float)
            for i, region_faces in enumerate(regions):
                region_ids[region_faces] = i
            ps_mesh.add_scalar_quantity("region", region_ids, defined_on="faces", cmap="spectral", enabled=True)
            ps.show()
        except ImportError:
            print("  polyscope not available, skipping visualization")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Segment G1 STL meshes by sharp creases")
    parser.add_argument("--angle", type=float, default=80.0, help="Dihedral angle threshold in degrees (default: 80)")
    parser.add_argument("--mesh", type=str, default=None, help="Process single mesh by name (e.g. head_link)")
    parser.add_argument("--all", action="store_true", help="Process all 35 G1 meshes")
    parser.add_argument(
        "--min-faces", type=int, default=10, help="Minimum faces per region; smaller regions get merged (default: 10)"
    )
    parser.add_argument("--vis", action="store_true", help="Open polyscope viewer (only with --mesh)")
    args = parser.parse_args()

    if not args.mesh and not args.all:
        parser.error("Specify --mesh NAME or --all")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mesh:
        meshes = [args.mesh.replace(".STL", "").replace(".stl", "")]
    else:
        meshes = G1_MESHES

    all_summaries = []
    for name in meshes:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        summary = process_one_mesh(
            name,
            angle_deg=args.angle,
            min_faces=args.min_faces,
            visualize=args.vis and len(meshes) == 1,
        )
        if summary:
            all_summaries.append(summary)
            print(f"  {summary['total_faces']} faces → {summary['n_regions']} regions")
            for r in summary["regions"]:
                pct = r["faces"] / summary["total_faces"] * 100
                print(f"    part{r['part']:02d}: {r['faces']:6d} faces ({pct:5.1f}%)  {r['file']}")

    # Print grand summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {len(all_summaries)} meshes segmented")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    for s in all_summaries:
        print(f"  {s['name']:35s}  {s['n_regions']:3d} regions  ({s['total_faces']} faces)")


if __name__ == "__main__":
    main()
