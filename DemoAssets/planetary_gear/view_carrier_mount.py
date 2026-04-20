"""
view_carrier_mount.py
=====================
Visualise the Marvin robot arm mounted on the carrier output flange.

Assembly concept
----------------
  sun_in_carrier_out mode:
    - Sun gear  -> input  (crank handle)
    - Ring gear -> fixed  (housing)
    - Carrier   -> output (rotates with robot attached)

  The carrier's bottom-face flange acts as the robot base-plate.
  Drag "Output Angle" to rotate the carrier + robot together, showing
  how the planetary-gear output drives the robot's first rotation axis.

Usage
-----
  python view_carrier_mount.py
"""

from pathlib import Path
import sys
import json
import numpy as np
import trimesh
import polyscope as ps
from polyscope import imgui

# ── Paths ──────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
ASSETS    = HERE / "assets"
REPO_ROOT = Path(__file__).resolve().parents[2]   # Genesis_IPC_demo
URDF_PATH = REPO_ROOT / "DemoAssets" / "marvin_robot" / "urdf" / "marvin_pika.urdf"
SCRIPTS   = REPO_ROOT / "DemoAssets" / "yoyo" / "scripts"
JOINTS_FILE = HERE / "robot_joints.json"

# ── Gear constants (mm, matching planetary_gear.scad) ─────────────────────
MODUL        = 3
SUN_TEETH    = 12
PLANET_TEETH = 9
NUM_PLANETS  = 3
GEAR_WIDTH   = 12.0   # mm
FLANGE_THICK = 6.0    # mm
ORBIT_R_MM   = MODUL * (SUN_TEETH + PLANET_TEETH) / 2   # 31.5 mm
MM_TO_M      = 0.001

# Carrier local-origin placed at world z = -GEAR_WIDTH/2.
# Flange bottom (robot mounting face) is FLANGE_THICK below that origin.
CARRIER_Z_MM = -GEAR_WIDTH / 2          # -6 mm
FLANGE_BOT_M = (CARRIER_Z_MM - FLANGE_THICK) * MM_TO_M   # -0.012 m

# ── Gear part colours ─────────────────────────────────────────────────────
PART_COLORS = {
    "sun_gear":  (0.95, 0.85, 0.20),
    "planet_0":  (0.35, 0.60, 0.90),
    "planet_1":  (0.35, 0.85, 0.50),
    "planet_2":  (0.70, 0.45, 0.85),
    "ring_gear": (0.85, 0.35, 0.35),
    "carrier":   (0.75, 0.75, 0.75),
}

MESH_FILE_MAP = {
    "sun_gear":  "sun_gear_handle.stl",
    "planet_0":  "planet_gear.stl",
    "planet_1":  "planet_gear.stl",
    "planet_2":  "planet_gear.stl",
    "ring_gear": "ring_gear.stl",
    "carrier":   "carrier.stl",
}

# Parts that rigidly follow the carrier output rotation
CARRIER_GROUP = {"carrier", "planet_0", "planet_1", "planet_2"}


# ── Math helpers ──────────────────────────────────────────────────────────
def rot_z_4x4(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s,  c
    return R


def apply_tf(verts, tf):
    ones = np.ones((len(verts), 1), dtype=verts.dtype)
    return (tf @ np.hstack([verts, ones]).T).T[:, :3]


def planet_base_tf_mm(idx):
    orbit_angle = np.radians(idx * 360.0 / NUM_PLANETS)
    self_rot    = np.radians(idx * 360.0 * SUN_TEETH / PLANET_TEETH)
    c, s = np.cos(self_rot), np.sin(self_rot)
    tf = np.eye(4)
    tf[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    tf[0, 3] = ORBIT_R_MM * np.cos(orbit_angle)
    tf[1, 3] = ORBIT_R_MM * np.sin(orbit_angle)
    return tf


def carrier_base_tf_mm():
    tf = np.eye(4)
    tf[2, 3] = CARRIER_Z_MM
    return tf


def build_root_tf(pos, rot_deg):
    rx, ry, rz = np.radians(rot_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    tf = np.eye(4)
    tf[:3, :3] = Rz @ Ry @ Rx
    tf[:3, 3]  = pos
    return tf


# ── URDF controller loader ────────────────────────────────────────────────
def load_urdf_controller():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from urdf_controller import URDFController
    return URDFController(str(URDF_PATH), mesh_source="visual")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # ── Load gear meshes ──────────────────────────────────────────────────
    verts0 = {}    # vertices in metres, base-assembled
    faces  = {}
    mesh_cache = {}

    print("\n-- Loading gear meshes --")
    for part, stl_name in MESH_FILE_MAP.items():
        path = ASSETS / stl_name
        if not path.exists():
            print(f"  [SKIP] {stl_name}")
            continue
        if stl_name not in mesh_cache:
            mesh_cache[stl_name] = trimesh.load(str(path), force="mesh")
        mesh = mesh_cache[stl_name]
        v_mm = mesh.vertices.copy().astype(np.float64)

        if part.startswith("planet_"):
            tf_mm = planet_base_tf_mm(int(part[-1]))
        elif part == "carrier":
            tf_mm = carrier_base_tf_mm()
        else:
            tf_mm = np.eye(4)

        v_mm = apply_tf(v_mm, tf_mm)
        verts0[part] = v_mm * MM_TO_M
        faces[part]  = mesh.faces.copy()
        print(f"  {part:12s}  {len(faces[part]):,} faces")

    # ── Load robot arm ────────────────────────────────────────────────────
    ctrl = None
    flange_pos = np.array([0.0, 0.0, FLANGE_BOT_M])

    print("\n-- Loading robot --")
    try:
        ctrl = load_urdf_controller()
        if JOINTS_FILE.exists():
            with open(JOINTS_FILE, encoding="utf-8") as f:
                jdata = json.load(f)
            jdata.pop("_root_pos", None)
            jdata.pop("_root_rot", None)
            ctrl.set_joint_positions(jdata)
        ctrl.set_root_transform(build_root_tf(flange_pos.tolist(), [0.0, 0.0, 0.0]))
        print(f"  Loaded {len(ctrl.mesh_nodes)} links from {URDF_PATH.name}")
        print(f"  Base at carrier flange  z = {FLANGE_BOT_M*1000:.1f} mm")
    except Exception as e:
        print(f"  [WARN] Robot load failed: {e}")

    # ── Polyscope ─────────────────────────────────────────────────────────
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_automatically_compute_scene_extents(False)
    ps.set_window_size(1600, 960)
    ps.set_display_message_popups(False)

    gear_ps = {}
    for part, v in verts0.items():
        sm = ps.register_surface_mesh(part, v, faces[part])
        sm.set_color(PART_COLORS.get(part, (0.6, 0.6, 0.6)))
        sm.set_smooth_shade(True)
        sm.set_edge_width(0.5)
        gear_ps[part] = sm
    if "ring_gear" in gear_ps:
        gear_ps["ring_gear"].set_transparency(0.35)

    # Flange indicator disk (orange ring at mounting face)
    theta_arr = np.linspace(0, 2 * np.pi, 64)
    flange_r_m = 0.053
    dz = FLANGE_BOT_M - 0.0002
    disk_rim = np.column_stack([
        flange_r_m * np.cos(theta_arr),
        flange_r_m * np.sin(theta_arr),
        np.full(64, dz),
    ])
    disk_v = np.vstack([[0.0, 0.0, dz], disk_rim])
    disk_f = np.array([[0, i+1, i+2] for i in range(63)] + [[0, 64, 1]])
    flange_sm = ps.register_surface_mesh("flange_indicator", disk_v, disk_f)
    flange_sm.set_color((0.9, 0.6, 0.1))
    flange_sm.set_transparency(0.45)

    def update_robot():
        if ctrl is None:
            return
        tfms = ctrl.get_mesh_transforms()
        for node in ctrl.mesh_nodes:
            tf  = tfms.get(node.node_name, np.eye(4))
            wv  = apply_tf(node.local_vertices.copy().astype(np.float64), tf)
            lbl = "robot/" + node.node_name
            m   = ps.register_surface_mesh(lbl, wv, node.faces)
            m.set_color((0.35, 0.48, 0.62))
            m.set_transparency(0.50)
            m.set_smooth_shade(True)

    update_robot()

    # ── Mutable state ─────────────────────────────────────────────────────
    carrier_angle  = [0.0]
    show_ring      = [True]
    show_flange    = [True]
    show_robot     = [True]
    robot_opacity  = [0.50]

    def apply_rotation(deg):
        Rz = rot_z_4x4(np.radians(deg))
        for part in CARRIER_GROUP:
            if part in verts0:
                gear_ps[part].update_vertex_positions(apply_tf(verts0[part], Rz))
        flange_sm.update_vertex_positions(apply_tf(disk_v, Rz))
        if ctrl is not None:
            T_f = np.eye(4)
            T_f[:3, 3] = flange_pos
            ctrl.set_root_transform(Rz @ T_f)
            update_robot()

    # ── GUI ───────────────────────────────────────────────────────────────
    def gui_callback():
        imgui.TextUnformatted("[ Carrier Output Rotation ]")
        changed, carrier_angle[0] = imgui.SliderFloat(
            "Output Angle (deg)", carrier_angle[0], -360.0, 360.0)
        if changed:
            apply_rotation(carrier_angle[0])

        if imgui.Button("Reset 0"):
            carrier_angle[0] = 0.0
            apply_rotation(0.0)
        imgui.SameLine()
        if imgui.Button("+90"):
            carrier_angle[0] = (carrier_angle[0] + 90.0) % 360.0
            apply_rotation(carrier_angle[0])
        imgui.SameLine()
        if imgui.Button("-90"):
            carrier_angle[0] = (carrier_angle[0] - 90.0) % 360.0
            apply_rotation(carrier_angle[0])

        imgui.Separator()
        imgui.TextUnformatted("[ Visibility ]")

        c, show_ring[0] = imgui.Checkbox("Ring gear (semi-transparent)", show_ring[0])
        if c and "ring_gear" in gear_ps:
            gear_ps["ring_gear"].set_transparency(0.35 if show_ring[0] else 0.0)

        c, show_flange[0] = imgui.Checkbox("Flange indicator", show_flange[0])
        if c:
            flange_sm.set_enabled(show_flange[0])

        c, show_robot[0] = imgui.Checkbox("Robot arm", show_robot[0])
        if c and ctrl is not None:
            for node in ctrl.mesh_nodes:
                try:
                    ps.get_surface_mesh("robot/" + node.node_name).set_enabled(show_robot[0])
                except Exception:
                    pass

        c, robot_opacity[0] = imgui.SliderFloat("Robot opacity", robot_opacity[0], 0.1, 1.0)
        if c and ctrl is not None:
            for node in ctrl.mesh_nodes:
                try:
                    ps.get_surface_mesh(
                        "robot/" + node.node_name).set_transparency(1.0 - robot_opacity[0])
                except Exception:
                    pass

        imgui.Separator()
        if imgui.TreeNode("Robot Joint Angles"):
            if ctrl is None:
                imgui.TextUnformatted("  (robot not loaded)")
            else:
                if imgui.Button("Snap to Flange"):
                    carrier_angle[0] = 0.0
                    apply_rotation(0.0)
                imgui.SameLine()
                if imgui.Button("Zero Joints"):
                    ctrl.set_joint_positions({n: 0.0 for n in ctrl.joint_names})
                    update_robot()

                positions = ctrl.get_joint_positions()
                limits    = ctrl.joint_limits
                any_changed = False
                for jname in ctrl.joint_names:
                    lo, hi = limits.get(jname, (-np.pi, np.pi))
                    c, val = imgui.SliderFloat(jname, positions[jname], lo, hi)
                    if c:
                        positions[jname] = val
                        any_changed = True
                if any_changed:
                    ctrl.set_joint_positions(positions)
                    update_robot()
            imgui.TreePop()

        imgui.Separator()
        imgui.TextUnformatted(
            f"Carrier: {carrier_angle[0]:+.1f} deg  "
            f"Flange z: {FLANGE_BOT_M*1000:.0f} mm  "
            f"Mode: sun_in -> carrier_out")

    ps.set_user_callback(gui_callback)

    print("\n========================================")
    print(" Carrier Mount Visualization")
    print("========================================")
    print(f" Flange z = {FLANGE_BOT_M*1000:.0f} mm  ({FLANGE_BOT_M:.4f} m)")
    print(f" Robot:   {URDF_PATH.name}")
    print(" Drag 'Output Angle' to rotate carrier + robot.")
    print("========================================\n")
    ps.show()


if __name__ == "__main__":
    main()
