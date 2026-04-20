"""
view_joint_reducer.py
=====================
Demonstrates a planetary gear reducer installed at a robot arm shoulder joint.

Concept
-------
  The shoulder joint between Link1_R (proximal, fixed housing) and Link2_R
  (distal, rotating output) is actuated by a planetary gear reducer:

    Motor side  -> Sun gear  (input, rotates fast)
    Housing     -> Ring gear (fixed to Link1_R, stationary)
    Output      -> Carrier   (rotates Link2_R at reduced speed)

  Reduction ratio  i = 1 + ring_teeth/sun_teeth  = 1 + 30/12 = 3.5
  So: omega_sun = 3.5 * omega_carrier = 3.5 * joint_angle_rate

GUI controls
------------
  - "Joint2_R angle" slider: rotates Link2_R (and all downstream links)
    while simultaneously spinning the carrier/sun to show the gear motion.
  - Visibility checkboxes for arm links and gear parts.

Usage
-----
  <venv>/python view_joint_reducer.py
"""

from pathlib import Path
import sys
import json
import numpy as np
import trimesh
import polyscope as ps
from polyscope import imgui

# ── Paths ─────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
ASSETS    = HERE / "assets"
REPO_ROOT = Path(__file__).resolve().parents[2]   # Genesis_IPC_demo
URDF_PATH = REPO_ROOT / "DemoAssets" / "marvin_robot" / "urdf" / "marvin_pika.urdf"
SCRIPTS   = REPO_ROOT / "DemoAssets" / "yoyo" / "scripts"
JOINTS_FILE = HERE / "robot_joints.json"

# ── Gear constants (mm, matching planetary_gear.scad) ─────────────────────
MODUL        = 3
SUN_TEETH    = 12
RING_TEETH   = 30        # sun + 2*planet = 12+18 = 30
PLANET_TEETH = 9
NUM_PLANETS  = 3
GEAR_WIDTH   = 12.0
ORBIT_R_MM   = MODUL * (SUN_TEETH + PLANET_TEETH) / 2   # 31.5 mm
MM_TO_M      = 0.001

# Reduction ratio (sun_in / carrier_out, ring fixed)
REDUCTION    = 1.0 + RING_TEETH / SUN_TEETH    # 3.5 x

# ── Which robot links to show ─────────────────────────────────────────────
# Link1_R = proximal (ring gear housing), Link2_R = distal (carrier output)
# Link3_R onward for visual context
SHOW_LINKS  = {"Base_R", "Link1_R", "Link2_R", "Link3_R", "Link4_R"}
TARGET_JOINT = "Joint2_R"   # shoulder pitch -- planetary reducer lives here

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


# ── Helpers ───────────────────────────────────────────────────────────────
def rot_z_3x3(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def apply_tf4(verts, tf4):
    ones = np.ones((len(verts), 1), dtype=verts.dtype)
    return (tf4 @ np.hstack([verts, ones]).T).T[:, :3]


def planet_base_tf_mm(idx):
    """Place planet at its orbit position, pointing in orbit tangent direction."""
    orbit_angle = np.radians(idx * 360.0 / NUM_PLANETS)
    c, s = np.cos(orbit_angle), np.sin(orbit_angle)
    tf = np.eye(4)
    # Rotate so planet tooth mesh aligns with sun mesh facing direction
    tf[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    tf[0, 3]   = ORBIT_R_MM * np.cos(orbit_angle)
    tf[1, 3]   = ORBIT_R_MM * np.sin(orbit_angle)
    return tf


def load_urdf_controller():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from urdf_controller import URDFController
    return URDFController(str(URDF_PATH), mesh_source="visual")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # ── Robot ─────────────────────────────────────────────────────────────
    print("-- Loading robot --")
    ctrl = load_urdf_controller()

    # Natural viewing pose: arm slightly raised, gear visible
    ctrl.set_root_transform(np.eye(4))
    ctrl.set_joint_positions({n: 0.0 for n in ctrl.joint_names})

    # Get joint-frame world transform (= Link2_R frame = Joint2_R frame)
    T_joint = ctrl.get_link_transform("Link2_R").astype(np.float64)
    R_joint = T_joint[:3, :3]   # gear-space -> world rotation
    p_joint = T_joint[:3, 3]    # gear centre in world

    print(f"  Joint {TARGET_JOINT} world position: {np.round(p_joint*1000,1)} mm")
    print(f"  Joint axis (world): {np.round(R_joint @ [0,0,1], 3)}")

    # ── Gear meshes ────────────────────────────────────────────────────────
    print("-- Loading gear meshes --")
    gear_verts_mm: dict[str, np.ndarray] = {}    # in gear local frame (mm)
    gear_faces:    dict[str, np.ndarray] = {}
    mesh_cache: dict[str, trimesh.Trimesh] = {}

    for part, stl_name in MESH_FILE_MAP.items():
        path = ASSETS / stl_name
        if not path.exists():
            print(f"  [SKIP] {stl_name}")
            continue
        if stl_name not in mesh_cache:
            mesh_cache[stl_name] = trimesh.load(str(path), force="mesh")
        mesh = mesh_cache[stl_name]
        v_mm = mesh.vertices.copy().astype(np.float64)

        # Assembly transform in gear local frame
        if part.startswith("planet_"):
            v_mm = apply_tf4(v_mm, planet_base_tf_mm(int(part[-1])))

        gear_verts_mm[part] = v_mm
        gear_faces[part]    = mesh.faces.copy()
        print(f"  {part:12s}  {len(gear_faces[part]):,} faces")

    def gear_to_world(v_mm_local):
        """Transform gear-space mm vertices to world-space metres."""
        v_m = v_mm_local * MM_TO_M
        return (R_joint @ v_m.T).T + p_joint

    def gear_to_world_rotated(v_mm_local, theta_gear):
        """Rotate in gear space by theta, then transform to world metres."""
        Rz = rot_z_3x3(theta_gear)
        v_rot = (Rz @ v_mm_local.T).T
        return gear_to_world(v_rot)

    # ── Polyscope ─────────────────────────────────────────────────────────
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_automatically_compute_scene_extents(False)
    ps.set_window_size(1600, 960)
    ps.set_display_message_popups(False)

    # Register gear meshes (initially at theta=0)
    gear_ps: dict[str, ps.SurfaceMesh] = {}
    for part, v_mm in gear_verts_mm.items():
        world_v = gear_to_world(v_mm)
        sm = ps.register_surface_mesh(part, world_v, gear_faces[part])
        sm.set_color(PART_COLORS.get(part, (0.6, 0.6, 0.6)))
        sm.set_smooth_shade(True)
        sm.set_edge_width(0.5)
        gear_ps[part] = sm
    if "ring_gear" in gear_ps:
        gear_ps["ring_gear"].set_transparency(0.4)   # see-through housing

    # Register robot arm meshes (only SHOW_LINKS)
    robot_ps: dict[str, ps.SurfaceMesh] = {}
    all_transforms = ctrl.get_mesh_transforms()

    for node in ctrl.mesh_nodes:
        # node.parent_link is the URDF link name; node.node_name is the scene-graph node
        link_name = node.parent_link if node.parent_link in SHOW_LINKS else None
        if link_name is None:
            continue

        tf  = all_transforms.get(node.node_name, np.eye(4))
        wv  = apply_tf4(node.local_vertices.copy().astype(np.float64), tf)
        lbl = "arm/" + node.node_name
        m   = ps.register_surface_mesh(lbl, wv, node.faces)

        # Colour: proximal (ring side) vs distal (carrier side)
        if node.parent_link in {"Base_R", "Link1_R"}:
            m.set_color((0.45, 0.55, 0.65))   # bluish – housing side
        else:
            m.set_color((0.65, 0.45, 0.35))   # reddish – output side
        m.set_transparency(0.35)
        m.set_smooth_shade(True)
        robot_ps[node.node_name] = m

    print(f"  Showing {len(robot_ps)} arm mesh nodes from {len(SHOW_LINKS)} links")

    # Draw a small axis indicator at the joint centre
    jax_len = 0.04   # 40 mm in metres
    joint_axis_world = R_joint @ np.array([0.0, 0.0, 1.0])
    axis_pts = np.array([p_joint - joint_axis_world * jax_len,
                         p_joint + joint_axis_world * jax_len])
    axis_curve = ps.register_curve_network(
        "joint_axis", axis_pts, np.array([[0, 1]]))
    axis_curve.set_color((1.0, 0.8, 0.0))
    axis_curve.set_radius(0.0025)

    # ── Mutable state ──────────────────────────────────────────────────────
    joint2_angle   = [0.0]    # radians
    show_ring      = [True]
    show_arm       = [True]
    show_axis      = [True]
    carrier_opacity= [1.0]

    def apply_joint_rotation(theta):
        """Update arm links and gear animation for joint angle theta (rad)."""
        # 1. Update robot FK
        ctrl.set_joint_positions({"Joint2_R": theta})
        new_tfs = ctrl.get_mesh_transforms()
        for nname, sm in robot_ps.items():
            node = next((n for n in ctrl.mesh_nodes if n.node_name == nname), None)
            if node is None:
                continue
            if node.parent_link not in {"Link2_R", "Link3_R", "Link4_R"}:
                continue   # only update downstream links
            tf = new_tfs.get(nname, np.eye(4))
            wv = apply_tf4(node.local_vertices.copy().astype(np.float64), tf)
            sm.update_vertex_positions(wv)

        # 2. Carrier rotates by theta (1:1 with joint output)
        for part in ("carrier", "planet_0", "planet_1", "planet_2"):
            if part in gear_ps:
                gear_ps[part].update_vertex_positions(
                    gear_to_world_rotated(gear_verts_mm[part], theta))

        # 3. Sun gear rotates REDUCTION times faster (motor input side)
        if "sun_gear" in gear_ps:
            gear_ps["sun_gear"].update_vertex_positions(
                gear_to_world_rotated(gear_verts_mm["sun_gear"], theta * REDUCTION))

    # ── GUI ───────────────────────────────────────────────────────────────
    def gui_callback():
        imgui.TextUnformatted("[ Shoulder Joint + Planetary Reducer ]")

        changed, joint2_angle[0] = imgui.SliderFloat(
            "Joint2_R angle (rad)", joint2_angle[0],
            -np.pi * 0.8, np.pi * 0.8)
        if changed:
            apply_joint_rotation(joint2_angle[0])

        imgui.SameLine()
        if imgui.Button("Zero"):
            joint2_angle[0] = 0.0
            apply_joint_rotation(0.0)

        deg = np.degrees(joint2_angle[0])
        sun_deg = deg * REDUCTION
        imgui.TextUnformatted(
            f"  Joint out: {deg:+.1f} deg   "
            f"| Carrier: {deg:+.1f} deg   "
            f"| Sun (motor): {sun_deg:+.1f} deg   "
            f"| Ratio: 1:{REDUCTION:.1f}")

        imgui.Separator()
        imgui.TextUnformatted("[ Visibility ]")

        c, show_ring[0] = imgui.Checkbox("Ring gear (housing)", show_ring[0])
        if c and "ring_gear" in gear_ps:
            gear_ps["ring_gear"].set_transparency(0.4 if show_ring[0] else 0.0)

        c, show_arm[0] = imgui.Checkbox("Arm links", show_arm[0])
        if c:
            for sm in robot_ps.values():
                sm.set_enabled(show_arm[0])

        c, show_axis[0] = imgui.Checkbox("Joint axis indicator", show_axis[0])
        if c:
            axis_curve.set_enabled(show_axis[0])

        c, carrier_opacity[0] = imgui.SliderFloat(
            "Carrier opacity", carrier_opacity[0], 0.1, 1.0)
        if c:
            for part in ("carrier", "planet_0", "planet_1", "planet_2"):
                if part in gear_ps:
                    gear_ps[part].set_transparency(1.0 - carrier_opacity[0])

        imgui.Separator()
        imgui.TextUnformatted(
            f"Ring gear = Link1_R housing (FIXED)   "
            f"Carrier = Link2_R output (rotating)")
        imgui.TextUnformatted(
            f"Joint pos: {np.round(p_joint*1000,1)} mm (world)   "
            f"Axis: {np.round(R_joint @ [0,0,1], 2)}")

    ps.set_user_callback(gui_callback)

    print()
    print("========================================")
    print(" Joint Reducer Visualization")
    print("========================================")
    print(f" Joint: {TARGET_JOINT}  Reduction: 1:{REDUCTION}")
    print(f" Gear Z-axis = joint rotation axis in world")
    print(f" Drag 'Joint2_R angle' to see gear + arm motion.")
    print("========================================")
    print()
    ps.show()

    # Clean up temp file
    tmp = REPO_ROOT / "DemoAssets" / "marvin_robot" / "_parse_joints.py"
    if tmp.exists():
        tmp.unlink()


if __name__ == "__main__":
    main()
