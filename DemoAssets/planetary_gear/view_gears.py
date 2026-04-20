"""
Polyscope viewer for the planetary gear set.

Reads ``assets/transforms.json`` for per-part placement and loads the
STL meshes (mm units -> metres).  The single ``planet_gear.stl`` is
instanced three times at the positions given by planet_0 / 1 / 2.

Usage:  python view_gears.py
"""

from pathlib import Path
import json
import numpy as np
import trimesh
import polyscope as ps
from polyscope import imgui

ASSETS_DIR = Path(__file__).parent / "assets"
MM_TO_M = 0.001

PART_COLORS = {
    "sun_gear":  (0.95, 0.85, 0.20),   # yellow
    "planet_0":  (0.55, 0.70, 0.90),   # blue
    "planet_1":  (0.55, 0.70, 0.90),
    "planet_2":  (0.55, 0.70, 0.90),
    "ring_gear": (0.85, 0.55, 0.60),   # pink
    "carrier":   (0.75, 0.75, 0.75),   # grey
}

MESH_FILE_MAP = {
    "sun_gear":  "sun_gear_handle.stl",
    "planet_0":  "planet_gear.stl",
    "planet_1":  "planet_gear.stl",
    "planet_2":  "planet_gear.stl",
    "ring_gear": "ring_gear.stl",
    "carrier":   "carrier.stl",
}


def load_mesh(path: Path) -> trimesh.Trimesh:
    return trimesh.load(str(path), force="mesh")


def apply_transform(verts: np.ndarray, tf_4x4: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to Nx3 vertices."""
    ones = np.ones((len(verts), 1), dtype=verts.dtype)
    homo = np.hstack([verts, ones])
    transformed = (tf_4x4 @ homo.T).T
    return transformed[:, :3]


def main():
    tf_path = ASSETS_DIR / "transforms.json"
    if not tf_path.exists():
        print(f"transforms.json not found in {ASSETS_DIR}")
        return

    with open(tf_path, encoding="utf-8") as f:
        transforms = json.load(f)

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1600, 1000)
    ps.set_display_message_popups(False)

    mesh_cache: dict[str, trimesh.Trimesh] = {}
    registered: dict[str, ps.SurfaceMesh] = {}

    for part_name, mesh_file in MESH_FILE_MAP.items():
        mesh_path = ASSETS_DIR / mesh_file
        if not mesh_path.exists():
            print(f"  [SKIP] {mesh_file} not found")
            continue

        if mesh_file not in mesh_cache:
            mesh_cache[mesh_file] = load_mesh(mesh_path)

        mesh = mesh_cache[mesh_file]
        verts = mesh.vertices.copy()

        if part_name in transforms:
            tf = np.array(transforms[part_name]["transform_4x4"], dtype=np.float64)
            verts = apply_transform(verts, tf)

        verts *= MM_TO_M

        color = PART_COLORS.get(part_name, (0.6, 0.6, 0.6))
        sm = ps.register_surface_mesh(part_name, verts, mesh.faces)
        sm.set_color(color)
        sm.set_smooth_shade(True)
        sm.set_edge_width(0.5)
        registered[part_name] = sm

        bbox = verts.max(axis=0) - verts.min(axis=0)
        print(f"  {part_name:12s}  {len(verts):>6,} verts  {len(mesh.faces):>6,} faces  "
              f"bbox {bbox[0]:.4f} x {bbox[1]:.4f} x {bbox[2]:.4f} m")

    print(f"\nLoaded {len(registered)} parts (mm -> m, transforms applied)")

    wireframe = [False]
    transparency = [1.0]

    def on_update():
        changed, wireframe[0] = imgui.Checkbox("Wireframe", wireframe[0])
        if changed:
            for sm in registered.values():
                sm.set_edge_width(2.0 if wireframe[0] else 0.5)

        changed, transparency[0] = imgui.SliderFloat(
            "Transparency", transparency[0], 0.0, 1.0)
        if changed:
            for sm in registered.values():
                sm.set_transparency(transparency[0])

        imgui.Separator()
        imgui.TextUnformatted(f"Parts: {len(registered)}")
        for name in registered:
            imgui.TextUnformatted(f"  {name}")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    main()
