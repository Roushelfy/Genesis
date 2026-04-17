"""
Load planetary gear STL files, convert to OBJ, assemble, and visualize
with Polyscope.  Also prints mesh quality checks for IPC readiness.

Usage:  python view_planetary.py
"""

from pathlib import Path
import json
import numpy as np
import trimesh
import polyscope as ps

ASSETS = Path(__file__).parent / "assets"

MODUL        = 3
SUN_TEETH    = 12
PLANET_TEETH = 9
NUM_PLANETS  = 3
GEAR_WIDTH   = 12

RING_TEETH = SUN_TEETH + 2 * PLANET_TEETH
SUN_PR     = MODUL * SUN_TEETH / 2      # 12 mm
PLANET_PR  = MODUL * PLANET_TEETH / 2   #  9 mm
RING_PR    = MODUL * RING_TEETH / 2     # 30 mm
ORBIT_R    = SUN_PR + PLANET_PR          # 21 mm  (== RING_PR - PLANET_PR)
CARRIER_THICK = 4.5

PARTS = {
    "sun_gear":    ASSETS / "sun_gear.stl",
    "planet_gear": ASSETS / "planet_gear.stl",
    "ring_gear":   ASSETS / "ring_gear.stl",
    "carrier":     ASSETS / "carrier.stl",
}

COLORS = {
    "sun_gear":    (1.0, 0.84, 0.0),
    "planet_gear": (0.27, 0.51, 0.71),
    "ring_gear":   (0.70, 0.13, 0.13),
    "carrier":     (0.75, 0.75, 0.75),
}


def load_and_check(name: str, path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")

    # Repair: merge close vertices, remove degenerate faces, fill holes
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)
    unique = mesh.unique_faces()
    mesh.update_faces(unique)
    mesh.fix_normals()
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fill_holes(mesh)

    wt = mesh.is_watertight
    vol = mesh.is_volume
    si = mesh.body_count
    print(f"  {name:14s}  verts={len(mesh.vertices):5d}  "
          f"faces={len(mesh.faces):5d}  "
          f"watertight={wt}  is_volume={vol}  bodies={si}")
    return mesh


def export_obj(mesh: trimesh.Trimesh, path: Path):
    mesh.export(str(path), file_type="obj")


def assemble_transforms() -> dict[str, np.ndarray]:
    """Return 4x4 transforms for each part in the assembled configuration."""
    tfs = {}

    # Sun: centered at origin (STL is already centered in Z)
    tfs["sun_gear"] = np.eye(4)

    # Planets: placed at orbit_r, rotated for mesh alignment
    # Matches library: rotate([0,0, n*360*d_sun/d_planet])
    for i in range(NUM_PLANETS):
        angle_deg = i * 360 / NUM_PLANETS
        angle_rad = np.radians(angle_deg)
        rot_z = i * 360 * SUN_TEETH / PLANET_TEETH
        rot_z_rad = np.radians(rot_z)

        tf = np.eye(4)
        c, s = np.cos(rot_z_rad), np.sin(rot_z_rad)
        tf[:2, :2] = [[c, -s], [s, c]]
        tf[0, 3] = ORBIT_R * np.cos(angle_rad)
        tf[1, 3] = ORBIT_R * np.sin(angle_rad)
        tfs[f"planet_{i}"] = tf

    # Ring: centered at origin
    tfs["ring_gear"] = np.eye(4)

    # Carrier: shifted below the gears
    tf = np.eye(4)
    tf[2, 3] = -GEAR_WIDTH / 2 - CARRIER_THICK
    tfs["carrier"] = tf

    return tfs


def main():
    print("Loading and checking meshes...")
    meshes = {}
    for name, path in PARTS.items():
        meshes[name] = load_and_check(name, path)

    # Export OBJ versions
    print("\nExporting OBJ files...")
    obj_dir = ASSETS
    for name, mesh in meshes.items():
        obj_path = obj_dir / f"{name}.obj"
        export_obj(mesh, obj_path)
        print(f"  {obj_path.name}")

    # Assembly transforms
    tfs = assemble_transforms()

    # Save transforms to JSON (for sim_gears.py compatibility)
    tf_data = {}
    for key, tf in tfs.items():
        tf_data[key] = {
            "transform_4x4": tf.tolist(),
        }
    tf_path = ASSETS / "transforms.json"
    with open(tf_path, "w", encoding="utf-8") as f:
        json.dump(tf_data, f, indent=2)
    print(f"\nSaved transforms -> {tf_path.name}")

    # Polyscope visualization
    print("\nLaunching Polyscope...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Register sun
    sun = meshes["sun_gear"]
    s = ps.register_surface_mesh("sun_gear", sun.vertices, sun.faces)
    s.set_color(COLORS["sun_gear"])

    # Register planets
    planet_base = meshes["planet_gear"]
    for i in range(NUM_PLANETS):
        tf = tfs[f"planet_{i}"]
        verts_h = np.hstack([planet_base.vertices,
                             np.ones((len(planet_base.vertices), 1))])
        verts_w = (tf @ verts_h.T).T[:, :3]
        sm = ps.register_surface_mesh(f"planet_{i}", verts_w, planet_base.faces)
        sm.set_color(COLORS["planet_gear"])

    # Register ring
    ring = meshes["ring_gear"]
    s = ps.register_surface_mesh("ring_gear", ring.vertices, ring.faces)
    s.set_color(COLORS["ring_gear"])
    s.set_transparency(0.5)

    # Register carrier
    carrier = meshes["carrier"]
    tf = tfs["carrier"]
    verts_h = np.hstack([carrier.vertices,
                         np.ones((len(carrier.vertices), 1))])
    verts_w = (tf @ verts_h.T).T[:, :3]
    s = ps.register_surface_mesh("carrier", verts_w, carrier.faces)
    s.set_color(COLORS["carrier"])

    # Auto-screenshot for verification, then interactive
    ps.set_view_projection_mode("orthographic")
    ps.look_at((5, -8, 60), (0, 0, 0))
    screenshot_path = str(Path(__file__).parent / "assets" / "preview.png")
    ps.screenshot(screenshot_path, transparent_bg=False)
    print(f"Screenshot saved: {screenshot_path}")

    ps.show()


if __name__ == "__main__":
    main()
