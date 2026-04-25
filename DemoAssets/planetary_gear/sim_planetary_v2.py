"""
IPC simulation of the planetary gear assembly -- v2 (large-clearance gears, d_hat=1e-3 m).

Uses modul_v2=2.5 teeth (sun+planet) for >=2.4 mm tooth-tip clearance at all mesh points.
Ring gear and carrier are unchanged from v1.

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
WORKSPACE = Path(__file__).parent / "sim_output_v2"
REPO_ROOT = Path(__file__).resolve().parents[2]  # Genesis_IPC_demo
URDF_PATH = REPO_ROOT / "DemoAssets" / "marvin_robot" / "urdf" / "marvin_pika.urdf"
JOINT_ANGLES_PATH = Path(__file__).parent / "robot_joints.json"

# Only these links are rendered as the "shoulder" mounted on the carrier flange.
# Base_R = arm base plate (bolted to carrier flange)
# Link1_R = first shoulder link (shoulder yaw)
# Link2_R = second shoulder link (shoulder pitch)
SHOULDER_LINKS  = {"Base_R", "Link1_R", "Link2_R", "Link3_R"}
SHOULDER_JOINTS = {"Joint1_R", "Joint2_R"}

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
RING_TEETH = SUN_TEETH + 2 * PLANET_TEETH   # = 30
REDUCTION  = 1.0 + RING_TEETH / SUN_TEETH   # = 3.5  (sun_in / carrier_out)

MM_TO_M = 0.001

# Robot arm default mount: base centre at the carrier flange bottom face.
# carrier is shifted by carrier_tz = -GEAR_WIDTH/2 in sim, then flange adds -FLANGE_THICK_MM below.
_FLANGE_MOUNT_POS = [0.0, 0.0,
                     round(-(GEAR_WIDTH / 2 + FLANGE_THICK_MM) * MM_TO_M, 6)]   # metres

# ── IPC / simulation parameters ──
D_HAT           = 5e-4
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
        if node.parent_link not in SHOULDER_LINKS:
            continue
        tf = transforms.get(node.node_name, np.eye(4))
        verts_h = np.hstack([node.local_vertices,
                             np.ones((len(node.local_vertices), 1))])
        world_verts = (tf @ verts_h.T).T[:, :3]
        label = f"shoulder/{node.node_name}"
        surf = ps.register_surface_mesh(label, world_verts, node.faces)
        surf.set_color((0.40, 0.52, 0.65))
        surf.set_transparency(0.40)
        surf.set_smooth_shade(True)


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
    # Compute root_transform so that Joint2_R world frame == gear world frame.
    #
    # Derivation:
    #   T_world(Link2_R) = root_transform @ T_local(Link2_R)
    #   We want  T_world(Link2_R) = T_gear  (identity rotation, z = carrier_tz)
    #   ∴  root_transform = T_gear @ inv(T_local(Link2_R))
    #
    # This also ensures Joint2_R's rotation axis (= Z in joint frame) ==
    # world Z  ==  carrier rotation axis.  So the carrier DRIVES Joint2_R.
    ctrl.set_root_transform(np.eye(4))
    ctrl.set_joint_positions({n: 0.0 for n in ctrl.joint_names})
    T_j2_local = ctrl.get_link_transform("Link2_R").copy()   # 4×4 from FK

    carrier_centre_z = -(GEAR_WIDTH / 2) * MM_TO_M           # ≈ -0.006 m
    T_gear = np.eye(4, dtype=np.float64)
    T_gear[2, 3] = carrier_centre_z                          # gear at z = -0.006

    root_tf = T_gear @ np.linalg.inv(T_j2_local)
    ctrl.set_root_transform(root_tf)
    _update_robot_meshes(ctrl)

    root_pos[:] = root_tf[:3, 3].tolist()
    root_rot_deg[:] = [0.0, 0.0, 0.0]   # kept for re-snap reference only
    shoulder_nodes = [n for n in ctrl.mesh_nodes if n.parent_link in SHOULDER_LINKS]

    # Verify: Joint2_R world pos should now be (0,0,carrier_centre_z)
    j2_world = ctrl.get_link_transform("Link2_R")[:3, 3]
    j2_axis  = ctrl.get_link_transform("Link2_R")[:3, 2]  # Z column = joint axis
    print(f"  [SHOULDER] {len(shoulder_nodes)} mesh nodes  "
          f"({', '.join(sorted(SHOULDER_LINKS))})")
    print(f"  [SHOULDER] Joint2_R world pos  = {np.round(j2_world*1000,1)} mm")
    print(f"  [SHOULDER] Joint2_R world axis = {np.round(j2_axis,3)}  "
          f"(should be [0,0,1])")


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
        "sun_gear":  "sun_gear_handle_v2.obj",
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
        parts.append((f"planet_{i}", "planet_gear_v2.obj", False, is_driven_planet))

    carrier_obj = None
    carrier_slot = None
    driven_obj   = None
    passive_obj  = None
    planet0_obj  = None
    planet0_init_tf = None  # 4x4 initial world transform for planet_0
    T_carrier       = np.eye(4, dtype=np.float64)  # initial transform for carrier

    print("Loading parts...")
    for name, obj_file, is_fixed, is_driven in parts:
        obj_path = ASSETS / obj_file
        if not obj_path.exists():
            raise FileNotFoundError(
                f"{obj_path} not found. Run view_planetary.py first.")

        mesh = io.read(str(obj_path))
        label_surface(mesh)
        # Mesh vertices stay in the canonical OBJ local frame (scaled by MM_TO_M).
        # Initial world placement is written to geo.transforms() after creation.

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

        # Compute and set this part's initial world transform.
        if name.startswith("planet_"):
            idx = int(name.split("_")[1])
            R, t = _planet_transform(idx)
            T_part = np.eye(4, dtype=np.float64)
            T_part[:3, :3] = R
            T_part[:3, 3]  = t
        elif name == "carrier":
            T_part = np.eye(4, dtype=np.float64)
            T_part[2, 3] = carrier_tz
            T_carrier = T_part
        else:
            T_part = np.eye(4, dtype=np.float64)  # sun_gear / ring_gear at origin

        view(gs.geometry().transforms())[0] = T_part

        if name == "carrier":
            carrier_obj = obj
            carrier_slot = gs
        if name == driven_part:
            driven_obj = obj
        elif name == passive_part:
            passive_obj = obj
        if name == "planet_0":
            planet0_obj = obj
            planet0_init_tf = T_part  # same 4×4 computed above

        nverts = mesh.vertices().size()
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

    def _compute_carrier_angle(frame: int) -> float:
        """Theoretical carrier rotation at given simulation frame (radians)."""
        t = frame * DT
        if t < t_driven_start:
            return 0.0
        dt = t - t_driven_start
        if dt < DRIVEN_RAMP_TIME:
            frac = dt / DRIVEN_RAMP_TIME
            driven_angle = DRIVEN_RAD_PER_S * 0.5 * frac * dt
        else:
            angle_ramp = DRIVEN_RAD_PER_S * 0.5 * DRIVEN_RAMP_TIME
            driven_angle = angle_ramp + DRIVEN_RAD_PER_S * (dt - DRIVEN_RAMP_TIME)
        if fixed_part == "carrier":
            return 0.0
        if driven_part == "carrier":
            return driven_angle
        # sun or ring is driven, carrier is passive output
        return driven_angle / REDUCTION

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
            T_lift = np.eye(4, dtype=np.float64)
            T_lift[2, 3] = dz
            aim_transform[0] = T_lift @ planet0_init_tf
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
        aim_transform[0] = T_carrier  # lock translation to T_carrier pos; rotation free

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

    run = False
    shoulder_panel_open = [True]
    auto_sync_arm = [True]   # whether to sync arm joints with carrier

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
            imgui.SameLine()
            c, auto_sync_arm[0] = imgui.Checkbox("Sync shoulder to carrier", auto_sync_arm[0])
            if c and not auto_sync_arm[0]:
                # Reset arm to neutral when unchecking
                if robot_ctrl is not None:
                    robot_ctrl.set_joint_positions({"Joint2_R": 0.0})
                    _update_robot_meshes(robot_ctrl)
        imgui.TextUnformatted(f"Mode: {DRIVE_MODE}  RPM: {DRIVEN_RPM}  d_hat: {D_HAT}m")

        if robot_ctrl is not None:
            # ── Shoulder joint panel ──────────────────────────────────────
            imgui.Separator()
            _, shoulder_panel_open[0] = imgui.CollapsingHeader(
                "Shoulder (Base_R / Link1_R / Link2_R)", shoulder_panel_open[0])
            if shoulder_panel_open[0]:
                imgui.TextUnformatted(
                    f"Shoulder bolted to carrier flange  "
                    f"z = {_FLANGE_MOUNT_POS[2]*1000:.1f} mm  "
                    f"Reduction 1:{REDUCTION:.1f}")

                imgui.Separator()
                joints_changed = False
                limits    = robot_ctrl.joint_limits
                positions = robot_ctrl.get_joint_positions()
                for jname in sorted(SHOULDER_JOINTS):
                    if jname not in limits:
                        continue
                    lo, hi = limits[jname]
                    c, val = imgui.SliderFloat(jname, positions.get(jname, 0.0), lo, hi)
                    if c:
                        positions[jname] = val
                        joints_changed = True
                if joints_changed:
                    robot_ctrl.set_joint_positions(positions)
                    _update_robot_meshes(robot_ctrl)

                imgui.Separator()
                if imgui.Button("Reset shoulder"):
                    robot_ctrl.set_joint_positions(
                        {n: 0.0 for n in SHOULDER_JOINTS})
                    _update_robot_meshes(robot_ctrl)
                imgui.SameLine()
                if imgui.Button("Re-snap to flange"):
                    robot_ctrl.set_root_transform(np.eye(4))
                    robot_ctrl.set_joint_positions(
                        {n: 0.0 for n in robot_ctrl.joint_names})
                    T_j2 = robot_ctrl.get_link_transform("Link2_R").copy()
                    cz = -(GEAR_WIDTH / 2) * MM_TO_M
                    T_gear = np.eye(4, dtype=np.float64)
                    T_gear[2, 3] = cz
                    root_tf = T_gear @ np.linalg.inv(T_j2)
                    robot_ctrl.set_root_transform(root_tf)
                    root_pos[:] = root_tf[:3, 3].tolist()
                    root_rot_deg[:] = [0.0, 0.0, 0.0]
                    _update_robot_meshes(robot_ctrl)

        if run:
            world.advance()
            world.retrieve()
            sgui.update()

            # ── Sync: carrier rotation → Joint2_R angle (arm only, base fixed) ──
            # Base_R / Link1_R stay stationary (housing side).
            # Link2_R / Link3_R follow carrier (output side).
            if robot_ctrl is not None and auto_sync_arm[0]:
                carrier_angle = _compute_carrier_angle(world.frame())
                robot_ctrl.set_joint_positions({"Joint2_R": carrier_angle})
                _update_robot_meshes(robot_ctrl)

    ps.set_user_callback(gui_callback)
    ps.show()

    if robot_ctrl is not None:
        _save_joint_angles(robot_ctrl, root_pos, root_rot_deg)


if __name__ == "__main__":
    main()
