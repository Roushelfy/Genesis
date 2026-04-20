"""
IPC simulation of the planetary gear assembly.

- Length unit: metre  (STL/OBJ are in mm, scaled ×0.001 at load time)
- d_hat = 0.001 m
- Backend: cuda
- Carrier driven by SoftTransformConstraint to rotate around Z axis.
  Planets are free bodies pushed by contact.

Usage:  python sim_planetary.py
"""

from pathlib import Path
import json
import numpy as np

import polyscope as ps
from polyscope import imgui

from uipc import view, Logger, Animation
from uipc import Vector3, Transform, AngleAxis
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import (
    SimplicialComplex, SimplicialComplexIO,
    label_surface, ground,
)
from uipc.constitution import AffineBodyConstitution, SoftTransformConstraint
from uipc.gui import SceneGUI
from uipc.unit import MPa

ASSETS = Path(__file__).parent / "assets"
WORKSPACE = Path(__file__).parent / "sim_output"
REPO_ROOT = Path(__file__).resolve().parents[2]  # Genesis_IPC_demo
URDF_PATH = REPO_ROOT / "DemoAssets" / "marvin_robot" / "urdf" / "marvin_pika.urdf"
JOINT_ANGLES_PATH = Path(__file__).parent / "robot_joints.json"

# ── Gear parameters (mm, matching planetary_gear.scad) ──
MODUL        = 3
SUN_TEETH    = 12
PLANET_TEETH = 9
NUM_PLANETS  = 3
GEAR_WIDTH   = 12
CARRIER_THICK = 4.5
FLANGE_THICK_MM = 6.0     # must match flange_thick in planetary_gear.scad

D_SUN      = MODUL * SUN_TEETH
D_PLANET   = MODUL * PLANET_TEETH
ORBIT_R_MM = MODUL * (SUN_TEETH + PLANET_TEETH) / 2

MM_TO_M = 0.001

# Robot arm default mount: base centre at the carrier flange bottom face.
# carrier is shifted by carrier_tz = -GEAR_WIDTH/2 in sim, then flange adds -FLANGE_THICK_MM below.
_FLANGE_MOUNT_POS = [0.0, 0.0,
                     round(-(GEAR_WIDTH / 2 + FLANGE_THICK_MM) * MM_TO_M, 6)]   # metres

# ── IPC / simulation parameters ──
D_HAT           = 2e-4
DT              = 0.01
ABD_KAPPA       = 100.0    # MPa
# STC strength_rate = [translation, rotation]
STC_STRENGTH_DRIVEN  = np.array([1000, 1000], dtype=np.float64)  # input: lock both axes
STC_STRENGTH_PASSIVE = np.array([1000, 0],    dtype=np.float64)  # output: lock translation, free rotation
STC_STRENGTH_PLANET  = np.array([100,  100],  dtype=np.float64)  # planet_0: pick-and-place

# ── Drive-mode configuration ──────────────────────────────────────────────────
# Selects which part is the powered input, which is the housing, and which is
# the free output driven by contact.
#
#   "driven"  – powered input  (STC locks translation + rotation; has rotation animation)
#   "passive" – free output    (STC locks translation only; rotation follows contact)
#   "fixed"   – housing/ground (is_fixed=True; not simulated as a free body)
#
# Available modes:
#   "sun_in_carrier_out"  – Sun driven,     ring fixed,    carrier passive output  ← default
#   "sun_in_ring_out"     – Sun driven,     carrier fixed, ring passive output
#   "carrier_in_sun_out"  – Carrier driven, ring fixed,    sun passive output
#   "carrier_in_ring_out" – Carrier driven, sun fixed,     ring passive output
DRIVE_MODE = "sun_in_carrier_out"

_MODE_CONFIG: dict[str, dict[str, str]] = {
    "sun_in_carrier_out":  dict(driven="sun_gear",  fixed="ring_gear", passive="carrier"),
    "sun_in_ring_out":     dict(driven="sun_gear",  fixed="carrier",   passive="ring_gear"),
    "carrier_in_sun_out":  dict(driven="carrier",   fixed="ring_gear", passive="sun_gear"),
    "carrier_in_ring_out": dict(driven="carrier",   fixed="sun_gear",  passive="ring_gear"),
}

DRIVEN_RPM       = -15.0
DRIVEN_RAD_PER_S = DRIVEN_RPM * 2 * np.pi / 60.0

# ── Animation phasing ──
LIFT_HEIGHT_MM   = 30.0       # how high to lift the planet (mm)
LOWER_HEIGHT_MM  = 15.0       # lower only halfway, then release for gravity drop
LIFT_SPEED_MM_S  = 25.0       # vertical speed (mm/s)
HOLD_TIME        = 0.5        # seconds to hold at top before lowering
SETTLE_TIME      = 0.3        # seconds to wait after release before driven part starts
DRIVEN_RAMP_TIME = 2.0        # seconds to ramp up driven speed from 0 to full

Logger.set_level(Logger.Level.Warn)


def _try_key_pressed(key_char: str) -> bool:
    for attr in (f"ImGuiKey_{key_char.upper()}", f"Key_{key_char.upper()}"):
        key = getattr(imgui, attr, None)
        if key is not None:
            try:
                return bool(imgui.IsKeyPressed(key))
            except (TypeError, AttributeError):
                continue
    try:
        return bool(imgui.IsKeyPressed(ord(key_char.upper())))
    except (TypeError, AttributeError):
        pass
    return False


def _get_urdf_controller():
    import sys
    scripts_dir = REPO_ROOT / "DemoAssets" / "yoyo" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from urdf_controller import URDFController
    return URDFController(str(URDF_PATH), mesh_source="visual")


def _build_root_transform(pos, rot_deg):
    rx, ry, rz = np.radians(rot_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = Rz @ Ry @ Rx
    tf[:3, 3] = pos
    return tf


def _update_robot_meshes(ctrl):
    transforms = ctrl.get_mesh_transforms()
    for node in ctrl.mesh_nodes:
        tf = transforms.get(node.node_name, np.eye(4))
        verts_h = np.hstack([node.local_vertices,
                             np.ones((len(node.local_vertices), 1))])
        world_verts = (tf @ verts_h.T).T[:, :3]
        label = f"robot/{node.node_name}"
        surf = ps.register_surface_mesh(label, world_verts, node.faces)
        surf.set_color((0.4, 0.5, 0.6))
        surf.set_transparency(0.5)


def _load_joint_angles():
    joints: dict[str, float] = {}
    root_pos = [0.0, 0.0, 0.0]
    root_rot = [0.0, 0.0, 0.0]
    if JOINT_ANGLES_PATH.exists():
        try:
            with open(JOINT_ANGLES_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "_root_pos" in data:
                root_pos = data.pop("_root_pos")
                root_rot = data.pop("_root_rot", root_rot)
                joints = data
            elif isinstance(data, dict):
                joints = data
            print(f"  [JOINTS] Loaded saved state from {JOINT_ANGLES_PATH.name}")
        except Exception:
            pass
    return joints, root_pos, root_rot


def _save_joint_angles(ctrl, root_pos, root_rot_deg):
    data = ctrl.get_joint_positions()
    data["_root_pos"] = list(root_pos)
    data["_root_rot"] = list(root_rot_deg)
    with open(JOINT_ANGLES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [JOINTS] Saved state to {JOINT_ANGLES_PATH.name}")


def load_robot_reference(ctrl, root_pos, root_rot_deg):
    saved_joints, saved_pos, saved_rot = _load_joint_angles()
    if saved_joints:
        ctrl.set_joint_positions(saved_joints)
    root_pos[:] = saved_pos
    root_rot_deg[:] = saved_rot
    ctrl.set_root_transform(_build_root_transform(root_pos, root_rot_deg))
    _update_robot_meshes(ctrl)
    print(f"  [REF] Loaded {len(ctrl.mesh_nodes)} robot meshes from {URDF_PATH.name}")


def _planet_transform(index: int):
    orbit_angle = np.radians(index * 360.0 / NUM_PLANETS)
    self_rot    = np.radians(index * 360.0 * SUN_TEETH / PLANET_TEETH)
    c, s = np.cos(self_rot), np.sin(self_rot)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float64)
    tx = ORBIT_R_MM * MM_TO_M * np.cos(orbit_angle)
    ty = ORBIT_R_MM * MM_TO_M * np.sin(orbit_angle)
    return R, np.array([tx, ty, 0.0])


def _apply_rigid_transform(mesh: SimplicialComplex,
                           R: np.ndarray, t: np.ndarray):
    pos = view(mesh.positions())
    verts = np.array(pos, copy=True).reshape(-1, 3)
    verts = (R @ verts.T).T + t
    flat = pos.reshape(-1)
    flat[:] = verts.reshape(-1)


def _rotation_z_4x4(angle_rad: float, tz: float = 0.0) -> np.ndarray:
    """4×4 rotation around Z with optional Z-translation."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    tf = np.eye(4, dtype=np.float64)
    tf[0, 0] = c;  tf[0, 1] = -s
    tf[1, 0] = s;  tf[1, 1] = c
    tf[2, 3] = tz
    return tf


def main():
    # ── Engine / World / Scene ──
    engine = Engine("cuda", str(WORKSPACE))
    world  = World(engine)

    config = Scene.default_config()
    config["dt"] = DT
    config["contact"]["d_hat"]            = D_HAT
    config["contact"]["enable"]           = True
    config["contact"]["friction"]["enable"] = True
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["sanity_check"]["enable"] = True
    config["newton"]["max_iter"] = 32
    scene = Scene(config)

    # ── Constitution ──
    abd = AffineBodyConstitution()
    stc = SoftTransformConstraint()

    # ── Contact ──
    ct = scene.contact_tabular()
    ct.default_model(0.0, 1e9)
    default_elem = ct.default_element()

    fixed_elem = ct.create("fixed_group")
    ct.insert(fixed_elem, fixed_elem, 0, 0, False)

    # ── IO: mm → m unit conversion ──
    unit_tf = Transform.Identity()
    unit_tf.scale(MM_TO_M)
    io = SimplicialComplexIO(unit_tf)

    # ── Resolve drive mode ──
    _mcfg = _MODE_CONFIG[DRIVE_MODE]
    driven_part  = _mcfg["driven"]
    fixed_part   = _mcfg["fixed"]
    passive_part = _mcfg["passive"]
    print(f"  [MODE] {DRIVE_MODE}  driven={driven_part}  fixed={fixed_part}  passive={passive_part}")

    carrier_tz = -(GEAR_WIDTH / 2) * MM_TO_M

    # ── Part definitions ──
    # Sun gear always uses the handle model; other parts use standard files.
    _obj_files = {
        "sun_gear":  "sun_gear_handle.obj",
        "ring_gear": "ring_gear.obj",
        "carrier":   "carrier.obj",
    }
    parts = []
    for pname in ("sun_gear", "ring_gear", "carrier"):
        is_fixed  = (pname == fixed_part)
        has_stc   = (pname in (driven_part, passive_part))
        parts.append((pname, _obj_files[pname], is_fixed, has_stc))
    for i in range(NUM_PLANETS):
        is_driven_planet = (i == 0)  # planet_0 is driven for pick-and-place
        parts.append((f"planet_{i}", "planet_gear.obj", False, is_driven_planet))

    carrier_obj = None
    carrier_slot = None
    driven_obj   = None
    passive_obj  = None
    planet0_obj  = None
    planet0_init_tf = None  # 4x4 initial transform for planet_0

    print("Loading parts...")
    for name, obj_file, is_fixed, is_driven in parts:
        obj_path = ASSETS / obj_file
        if not obj_path.exists():
            raise FileNotFoundError(
                f"{obj_path} not found. Run view_planetary.py first.")

        mesh = io.read(str(obj_path))
        label_surface(mesh)

        if name.startswith("planet_"):
            idx = int(name.split("_")[1])
            R, t = _planet_transform(idx)
            _apply_rigid_transform(mesh, R, t)
        elif name == "carrier":
            _apply_rigid_transform(mesh, np.eye(3),
                                   np.array([0, 0, carrier_tz]))

        abd.apply_to(mesh, ABD_KAPPA * MPa)

        if is_driven:
            if name in (driven_part, passive_part):
                strength = STC_STRENGTH_DRIVEN if name == driven_part else STC_STRENGTH_PASSIVE
            elif name.startswith("planet_"):
                strength = STC_STRENGTH_PLANET
            else:
                strength = STC_STRENGTH_DRIVEN
            stc.apply_to(mesh, strength)

        if is_fixed:
            view(mesh.instances().find(builtin.is_fixed))[:] = 1
            fixed_elem.apply_to(mesh)
        else:
            default_elem.apply_to(mesh)

        obj = scene.objects().create(name)
        gs, _ = obj.geometries().create(mesh)

        if name == "carrier":
            carrier_obj = obj
            carrier_slot = gs
        if name == driven_part:
            driven_obj = obj
        elif name == passive_part:
            passive_obj = obj
        if name == "planet_0":
            planet0_obj = obj
            R0, t0 = _planet_transform(0)
            tf4 = np.eye(4, dtype=np.float64)
            tf4[:3, :3] = R0
            tf4[:3, 3] = t0
            planet0_init_tf = tf4

        nverts = np.array(view(mesh.positions()), copy=True).reshape(-1, 3).shape[0]
        tag = "FIXED" if is_fixed else ("DRIVE" if is_driven else " FREE")
        print(f"  [{tag:5s}] {name:14s}  {nverts:5d} verts")

    # ── Ground ──
    g = scene.objects().create("ground")
    g.geometries().create(ground(-0.5, Vector3.UnitZ()))

    # ── Two-phase animation ──
    # Phase 1: lift planet_0, hold, lower back
    # Phase 2: release planet_0, start sun rotation
    lift_h = LIFT_HEIGHT_MM * MM_TO_M
    lower_h = LOWER_HEIGHT_MM * MM_TO_M     # only lower this much before release
    lift_speed = LIFT_SPEED_MM_S * MM_TO_M
    t_lift = lift_h / lift_speed            # time to lift
    t_hold = HOLD_TIME                      # hold at top
    t_lower = lower_h / lift_speed          # time to lower partway
    t_phase1 = t_lift + t_hold + t_lower    # total phase 1 duration
    release_h = lift_h - lower_h            # height at release
    t_driven_start = t_phase1 + SETTLE_TIME # driven part starts after settle
    print(f"  [ANIM] Phase 1: lift {t_lift:.2f}s + hold {t_hold:.2f}s + lower {t_lower:.2f}s = {t_phase1:.2f}s")
    print(f"  [ANIM] Release at {release_h*1000:.1f}mm above, gravity drops the rest")
    print(f"  [ANIM] Phase 2: {driven_part} rotation starts at t={t_driven_start:.2f}s (settle {SETTLE_TIME:.1f}s)")

    animator = scene.animator()

    def planet0_animation(info: Animation.UpdateInfo):
        geo: SimplicialComplex = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform = view(geo.instances().find(builtin.aim_transform))
        t = info.frame() * info.dt()

        if t < t_phase1:
            is_constrained[0] = 1
            if t < t_lift:
                dz = lift_speed * t
            elif t < t_lift + t_hold:
                dz = lift_h
            else:
                dz = lift_h - lift_speed * (t - t_lift - t_hold)
                dz = max(dz, release_h)
            tf = np.eye(4, dtype=np.float64)
            tf[2, 3] = dz
            aim_transform[0] = tf
        else:
            is_constrained[0] = 0

    animator.insert(planet0_obj, planet0_animation)

    def driven_animation(info: Animation.UpdateInfo):
        geo: SimplicialComplex = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform = view(geo.instances().find(builtin.aim_transform))
        t = info.frame() * info.dt()
        is_constrained[0] = 1
        if t < t_driven_start:
            aim_transform[0] = _rotation_z_4x4(0.0)
        else:
            dt = t - t_driven_start
            if dt < DRIVEN_RAMP_TIME:
                frac = dt / DRIVEN_RAMP_TIME
                angle = DRIVEN_RAD_PER_S * 0.5 * frac * dt
            else:
                angle_ramp = DRIVEN_RAD_PER_S * 0.5 * DRIVEN_RAMP_TIME
                angle = angle_ramp + DRIVEN_RAD_PER_S * (dt - DRIVEN_RAMP_TIME)
            aim_transform[0] = _rotation_z_4x4(angle)

    animator.insert(driven_obj, driven_animation)

    def passive_animation(info: Animation.UpdateInfo):
        geo: SimplicialComplex = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform = view(geo.instances().find(builtin.aim_transform))
        is_constrained[0] = 1
        aim_transform[0] = np.eye(4, dtype=np.float64)  # lock translation, allow rotation

    animator.insert(passive_obj, passive_animation)

    # ── Init ──
    print("\nInitializing world...")
    world.init(scene)

    # ── Sanity check ──
    print("\n" + "=" * 60)
    valid = world.is_valid()
    if valid:
        print("  PASS  --  No initial penetration detected")
    else:
        print("  FAIL  --  Initial penetration detected!")
        checker = world.sanity_checker()
        checker.report()
    print("=" * 60)

    if not valid:
        print("\n  [DEBUG] Opening Polyscope for visual inspection despite invalid world...")
        print("  [DEBUG] Simulation will NOT run — use this window to check scene geometry.\n")

    # ── Polyscope ──
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_window_size(1600, 1000)

    sgui = SceneGUI(scene, "split")
    sgui.register()
    sgui.set_edge_width(1)

    # ── Load robot reference (visual only, not simulated) ──
    robot_ctrl = None
    root_pos = [0.0, 0.0, 0.0]
    root_rot_deg = [0.0, 0.0, 0.0]
    if URDF_PATH.exists():
        robot_ctrl = _get_urdf_controller()
        load_robot_reference(robot_ctrl, root_pos, root_rot_deg)

    # Lock ground & scene extents after all structures are registered
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height(-0.5)
    ps.set_automatically_compute_scene_extents(False)

    # ── Detect end effectors ──
    end_effectors = []
    left_ee = ""
    right_ee = ""
    if robot_ctrl is not None:
        end_effectors = robot_ctrl.find_end_effectors()
        left_ee = next((e for e in end_effectors if "_L" in e or "left" in e.lower()),
                       end_effectors[0] if end_effectors else "")
        right_ee = next((e for e in end_effectors if "_R" in e or "right" in e.lower()),
                        end_effectors[-1] if end_effectors else "")
        print(f"  [IK] Left EE: {left_ee}  |  Right EE: {right_ee}")

    # ── Orientation presets (Z-up world) ──
    orient_presets: dict[str, np.ndarray] = {
        "None (pos only)": np.zeros((3, 3)),
        "Horiz -Z": np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64),
        "Horiz -Z inv": np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64),
        "Horiz +X": np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=np.float64),
        "Horiz +X inv": np.array([[0,0,1],[0,-1,0],[1,0,0]], dtype=np.float64),
        "Horiz -X": np.array([[0,0,-1],[0,1,0],[1,0,0]], dtype=np.float64),
        "Horiz -X inv": np.array([[0,0,-1],[0,-1,0],[-1,0,0]], dtype=np.float64),
        "Horiz +Y": np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float64),
        "Horiz +Y inv": np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float64),
        "Horiz -Y": np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64),
        "Horiz -Y inv": np.array([[-1,0,0],[0,0,-1],[0,-1,0]], dtype=np.float64),
        "Vert +Z": np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64),
        "Vert +Z inv": np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=np.float64),
    }
    orient_names: list[str] = list(orient_presets.keys())

    run = False
    joint_panel_open = [True]
    ik_panel_open = [True]
    ik_step = [0.005]
    left_orient_idx = [0]
    right_orient_idx = [0]

    def _get_orient(idx: int):
        R = orient_presets[orient_names[idx]]
        if np.allclose(R, 0):
            return None
        return R

    def do_ik(ee_name, delta, orient_idx):
        cur_tf = robot_ctrl.get_link_transform(ee_name)
        cur_pos = cur_tf[:3, 3].copy()
        arm_joints = robot_ctrl.find_arm_joints(ee_name)
        if not arm_joints:
            return
        target_orient = _get_orient(orient_idx)
        orient_mode = "all" if target_orient is not None else None
        robot_ctrl.solve_ik(
            ee_name, cur_pos + delta,
            target_orientation=target_orient,
            orientation_mode=orient_mode,
            arm_joints=arm_joints,
        )
        _update_robot_meshes(robot_ctrl)

    def gui_callback():
        nonlocal run
        if not valid:
            imgui.TextUnformatted("  *** WORLD INVALID — inspect scene, simulation disabled ***")
            imgui.Separator()
        else:
            if imgui.Button("Run / Stop"):
                run = not run
            imgui.SameLine()
            imgui.TextUnformatted(f"  Frame: {world.frame()}  "
                                  f"Time: {world.frame()*DT:.2f}s")
        imgui.TextUnformatted(f"Mode: {DRIVE_MODE}  RPM: {DRIVEN_RPM}  "
                              f"d_hat: {D_HAT}m")

        if robot_ctrl is not None:
            # ── IK Control panel ──
            imgui.Separator()
            _, ik_panel_open[0] = imgui.CollapsingHeader(
                "IK Control", ik_panel_open[0])
            if ik_panel_open[0]:
                c, val = imgui.SliderFloat("IK Step (m)", ik_step[0], 0.001, 0.05)
                if c:
                    ik_step[0] = val
                s = ik_step[0]
                imgui.Separator()

                # Left gripper
                if left_ee and imgui.TreeNode(f"Left: {left_ee}  [WASDQE]"):
                    ltf = robot_ctrl.get_link_transform(left_ee)
                    lp = ltf[:3, 3]
                    imgui.TextUnformatted(f"Pos: ({lp[0]:.4f}, {lp[1]:.4f}, {lp[2]:.4f})")
                    arm_j = robot_ctrl.find_arm_joints(left_ee)
                    imgui.TextUnformatted(f"IK joints ({len(arm_j)}): {', '.join(arm_j)}")
                    imgui.TextUnformatted("Orientation:")
                    c, oidx = imgui.Combo("##orient_L", left_orient_idx[0], orient_names)
                    if c:
                        left_orient_idx[0] = oidx
                    if imgui.Button("Lock current##lock_L"):
                        orient_presets["Locked L"] = ltf[:3, :3].copy()
                        if "Locked L" not in orient_names:
                            orient_names.append("Locked L")
                        left_orient_idx[0] = orient_names.index("Locked L")
                    for label, delta in [
                        ("+X##Lik", [s,0,0]), ("-X##Lik", [-s,0,0]),
                        ("+Y##Lik", [0,s,0]), ("-Y##Lik", [0,-s,0]),
                        ("+Z##Lik", [0,0,s]), ("-Z##Lik", [0,0,-s]),
                    ]:
                        if label != "+X##Lik":
                            imgui.SameLine()
                        if imgui.Button(label):
                            do_ik(left_ee, np.array(delta), left_orient_idx[0])
                    imgui.TreePop()

                imgui.Separator()

                # Right gripper
                if right_ee and imgui.TreeNode(f"Right: {right_ee}  [IJKLUO]"):
                    rtf = robot_ctrl.get_link_transform(right_ee)
                    rp = rtf[:3, 3]
                    imgui.TextUnformatted(f"Pos: ({rp[0]:.4f}, {rp[1]:.4f}, {rp[2]:.4f})")
                    arm_j = robot_ctrl.find_arm_joints(right_ee)
                    imgui.TextUnformatted(f"IK joints ({len(arm_j)}): {', '.join(arm_j)}")
                    imgui.TextUnformatted("Orientation:")
                    c, oidx = imgui.Combo("##orient_R", right_orient_idx[0], orient_names)
                    if c:
                        right_orient_idx[0] = oidx
                    if imgui.Button("Lock current##lock_R"):
                        orient_presets["Locked R"] = rtf[:3, :3].copy()
                        if "Locked R" not in orient_names:
                            orient_names.append("Locked R")
                        right_orient_idx[0] = orient_names.index("Locked R")
                    for label, delta in [
                        ("+X##Rik", [s,0,0]), ("-X##Rik", [-s,0,0]),
                        ("+Y##Rik", [0,s,0]), ("-Y##Rik", [0,-s,0]),
                        ("+Z##Rik", [0,0,s]), ("-Z##Rik", [0,0,-s]),
                    ]:
                        if label != "+X##Rik":
                            imgui.SameLine()
                        if imgui.Button(label):
                            do_ik(right_ee, np.array(delta), right_orient_idx[0])
                    imgui.TreePop()

                imgui.Separator()

                # Keyboard IK polling
                left_delta = np.zeros(3, dtype=np.float64)
                if _try_key_pressed("W"): left_delta[0] += s
                if _try_key_pressed("S"): left_delta[0] -= s
                if _try_key_pressed("A"): left_delta[1] += s
                if _try_key_pressed("D"): left_delta[1] -= s
                if _try_key_pressed("E"): left_delta[2] += s
                if _try_key_pressed("Q"): left_delta[2] -= s
                if np.any(left_delta != 0) and left_ee:
                    do_ik(left_ee, left_delta, left_orient_idx[0])

                right_delta = np.zeros(3, dtype=np.float64)
                if _try_key_pressed("I"): right_delta[0] += s
                if _try_key_pressed("K"): right_delta[0] -= s
                if _try_key_pressed("J"): right_delta[1] += s
                if _try_key_pressed("L"): right_delta[1] -= s
                if _try_key_pressed("O"): right_delta[2] += s
                if _try_key_pressed("U"): right_delta[2] -= s
                if np.any(right_delta != 0) and right_ee:
                    do_ik(right_ee, right_delta, right_orient_idx[0])

                imgui.TextUnformatted("Left  keys: W/S=X  A/D=Y  Q/E=Z")
                imgui.TextUnformatted("Right keys: I/K=X  J/L=Y  U/O=Z")

            # ── Robot joint panel ──
            imgui.Separator()
            _, joint_panel_open[0] = imgui.CollapsingHeader(
                "Robot Joints", joint_panel_open[0])
            if joint_panel_open[0]:
                root_changed = False
                for axis, label in enumerate(["Root X", "Root Y", "Root Z"]):
                    changed, new_val = imgui.SliderFloat(
                        label, root_pos[axis], -1.0, 1.0)
                    if changed:
                        root_pos[axis] = new_val
                        root_changed = True
                for axis, label in enumerate(["Rot X", "Rot Y", "Rot Z"]):
                    changed, new_val = imgui.SliderFloat(
                        label, root_rot_deg[axis], -180.0, 180.0)
                    if changed:
                        root_rot_deg[axis] = new_val
                        root_changed = True
                if root_changed:
                    robot_ctrl.set_root_transform(
                        _build_root_transform(root_pos, root_rot_deg))
                    _update_robot_meshes(robot_ctrl)

                imgui.Separator()
                joints_changed = False
                limits = robot_ctrl.joint_limits
                positions = robot_ctrl.get_joint_positions()
                for name in robot_ctrl.joint_names:
                    lo, hi = limits.get(name, (-np.pi, np.pi))
                    changed, new_val = imgui.SliderFloat(
                        name, positions[name], lo, hi)
                    if changed:
                        positions[name] = new_val
                        joints_changed = True
                if joints_changed:
                    robot_ctrl.set_joint_positions(positions)
                    _update_robot_meshes(robot_ctrl)
                if imgui.Button("Reset Joints"):
                    robot_ctrl.set_joint_positions(
                        {n: 0.0 for n in robot_ctrl.joint_names})
                    _update_robot_meshes(robot_ctrl)
                imgui.SameLine()
                if imgui.Button("Snap to Flange"):
                    root_pos[:] = list(_FLANGE_MOUNT_POS)
                    root_rot_deg[:] = [0.0, 0.0, 0.0]
                    robot_ctrl.set_root_transform(
                        _build_root_transform(root_pos, root_rot_deg))
                    _update_robot_meshes(robot_ctrl)
                    print(f"  [FLANGE] Robot snapped to flange mount {_FLANGE_MOUNT_POS}")
                imgui.SameLine()
                if imgui.Button("Save Joints"):
                    _save_joint_angles(robot_ctrl, root_pos, root_rot_deg)

        if run:
            world.advance()
            world.retrieve()
            sgui.update()

    ps.set_user_callback(gui_callback)
    ps.show()

    if robot_ctrl is not None:
        _save_joint_angles(robot_ctrl, root_pos, root_rot_deg)


if __name__ == "__main__":
    main()
