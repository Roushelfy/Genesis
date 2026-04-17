"""
Programmatic multi-angle screenshots of the planetary gear set.

Loads meshes + transforms.json, applies mm->m scaling, takes screenshots
from several camera angles, and saves them next to this script.

Usage:  python screenshot_gears.py
"""

from pathlib import Path
import json
import numpy as np
import trimesh
import polyscope as ps

ASSETS_DIR = Path(__file__).parent / "assets"
OUT_DIR = Path(__file__).parent
MM_TO_M = 0.001

PART_COLORS = {
    "sun_gear":  (0.95, 0.85, 0.20),
    "planet_0":  (0.55, 0.70, 0.90),
    "planet_1":  (0.55, 0.70, 0.90),
    "planet_2":  (0.55, 0.70, 0.90),
    "ring_gear": (0.85, 0.55, 0.60),
    "carrier":   (0.75, 0.75, 0.75),
}

MESH_FILE_MAP = {
    "sun_gear":  "sun_gear.stl",
    "planet_0":  "planet_gear.stl",
    "planet_1":  "planet_gear.stl",
    "planet_2":  "planet_gear.stl",
    "ring_gear": "ring_gear.stl",
    "carrier":   "carrier.stl",
}


def load_mesh(path: Path) -> trimesh.Trimesh:
    return trimesh.load(str(path), force="mesh")


def apply_transform(verts: np.ndarray, tf_4x4: np.ndarray) -> np.ndarray:
    ones = np.ones((len(verts), 1), dtype=verts.dtype)
    homo = np.hstack([verts, ones])
    return (tf_4x4 @ homo.T).T[:, :3]


def main():
    tf_path = ASSETS_DIR / "transforms.json"
    with open(tf_path, encoding="utf-8") as f:
        transforms = json.load(f)

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1920, 1080)
    ps.set_SSAA_factor(2)
    ps.set_display_message_popups(False)

    mesh_cache: dict[str, trimesh.Trimesh] = {}
    all_verts_world = []

    for part_name, mesh_file in MESH_FILE_MAP.items():
        mesh_path = ASSETS_DIR / mesh_file
        if not mesh_path.exists():
            continue
        if mesh_file not in mesh_cache:
            mesh_cache[mesh_file] = load_mesh(mesh_path)

        mesh = mesh_cache[mesh_file]
        verts = mesh.vertices.copy()

        if part_name in transforms:
            tf = np.array(transforms[part_name]["transform_4x4"], dtype=np.float64)
            verts = apply_transform(verts, tf)

        verts *= MM_TO_M
        all_verts_world.append(verts)

        sm = ps.register_surface_mesh(part_name, verts, mesh.faces)
        sm.set_color(PART_COLORS.get(part_name, (0.6, 0.6, 0.6)))
        sm.set_smooth_shade(True)
        sm.set_edge_width(0.5)

        print(f"  {part_name:12s}  {len(verts):>6,} verts  {len(mesh.faces):>6,} faces")

    all_pts = np.vstack(all_verts_world)
    center = all_pts.mean(axis=0)
    bbox = all_pts.max(axis=0) - all_pts.min(axis=0)
    r = np.linalg.norm(bbox) / 2.0
    eps = r * 0.01
    c = tuple(center.tolist())
    print(f"\n  Center: {center}   BBox: {bbox}   radius: {r:.4f} m")

    views = {
        "top":        (center[0] + eps,      center[1] - eps,       center[2] + 3*r),
        "iso":        (center[0] + 2*r,      center[1] - 2*r,       center[2] + 2*r),
        "front":      (center[0],            center[1] - 3*r,       center[2]),
        "side":       (center[0] + 3*r,      center[1],             center[2]),
        "close_iso":  (center[0] + 1.2*r,    center[1] - 1.2*r,     center[2] + 0.6*r),
    }

    saved = []
    for vname, eye in views.items():
        ps.look_at(eye, c)
        ps.frame_tick()
        out = str(OUT_DIR / f"screenshot_{vname}.png")
        ps.screenshot(out, transparent_bg=False)
        saved.append(out)
        print(f"  Saved: {out}")

    # wireframe + transparency for penetration analysis
    for part_name in MESH_FILE_MAP:
        s = ps.get_surface_mesh(part_name)
        if s is not None:
            s.set_edge_width(2.0)
            s.set_transparency(0.6)

    for vname in ("top", "iso", "close_iso"):
        ps.look_at(views[vname], c)
        ps.frame_tick()
        out = str(OUT_DIR / f"screenshot_wire_{vname}.png")
        ps.screenshot(out, transparent_bg=False)
        saved.append(out)
        print(f"  Saved: {out}")

    print(f"\nDone -- {len(saved)} screenshots in {OUT_DIR}")


if __name__ == "__main__":
    main()
