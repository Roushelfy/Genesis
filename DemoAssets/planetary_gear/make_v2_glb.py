"""Create v2 GLB files by transferring PBR materials from v1 GLBs onto v2 OBJ geometry.

UV coordinates are transferred via nearest-vertex lookup from v1 → v2.
This works because v2 gears have the same overall shape, just slightly smaller teeth.

Output:
    assets/sun_gear_handle_v2.glb
    assets/planet_gear_v2.glb
"""

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ASSETS = Path(__file__).resolve().parent / "assets"

PAIRS = [
    ("sun_gear_handle", "sun_gear_handle_v2"),
    ("planet_gear", "planet_gear_v2"),
]


def transfer_material(v1_glb_path: str, v2_obj_path: str, out_glb_path: str) -> None:
    # Load v1 GLB (has PBR material + UVs, Y-up convention)
    v1_scene = trimesh.load(v1_glb_path)
    v1_geom = list(v1_scene.geometry.values())[0]
    v1_mat = v1_geom.visual.material
    v1_uv = v1_geom.visual.uv  # (N1, 2)
    v1_verts = v1_geom.vertices  # (N1, 3) — Y-up

    # Load v2 OBJ (Z-up from OpenSCAD)
    v2_mesh = trimesh.load(v2_obj_path, force="mesh")

    # Rotate v2 from Z-up to Y-up (GLB convention): X stays, Y←Z, Z←-Y
    rot_z_to_y = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    v2_mesh.apply_transform(rot_z_to_y)
    v2_verts = v2_mesh.vertices  # (N2, 3) — now Y-up

    print(f"  v1: {len(v1_verts)} verts, v2: {len(v2_verts)} verts")
    print(f"  v1 Y range: [{v1_verts[:, 1].min():.2f}, {v1_verts[:, 1].max():.2f}]")
    print(f"  v2 Y range: [{v2_verts[:, 1].min():.2f}, {v2_verts[:, 1].max():.2f}]")

    # Transfer UVs: for each v2 vertex, find nearest v1 vertex and copy its UV
    tree = cKDTree(v1_verts)
    dists, indices = tree.query(v2_verts)
    v2_uv = v1_uv[indices]

    print(f"  UV transfer: max dist = {dists.max():.4f}, mean dist = {dists.mean():.4f}")

    # Apply material to v2 mesh
    v2_mesh.visual = trimesh.visual.TextureVisuals(uv=v2_uv, material=v1_mat)

    # Export as GLB (Y-up, matching v1 convention)
    v2_mesh.export(out_glb_path, file_type="glb")
    print(f"  Saved: {out_glb_path}")


def main() -> None:
    for v1_name, v2_name in PAIRS:
        v1_glb = str(ASSETS / f"{v1_name}.glb")
        v2_obj = str(ASSETS / f"{v2_name}.obj")
        out_glb = str(ASSETS / f"{v2_name}.glb")

        print(f"\n{v1_name}.glb → {v2_name}.glb")

        if not Path(v2_obj).exists():
            print(f"  SKIP: {v2_obj} not found")
            continue

        transfer_material(v1_glb, v2_obj, out_glb)


if __name__ == "__main__":
    main()
