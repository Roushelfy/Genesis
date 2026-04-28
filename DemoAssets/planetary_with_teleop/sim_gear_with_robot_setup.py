"""UIPC simulation of the planetary gear system driven by MARVIN_SHARPA teleop.

Planetary gear setup (adapted from ``sim_planetary.py``, simplified):
  - Ring gear  : fixed housing (is_fixed=True)
  - Sun gear   : STC-driven input  (rotates from t=0 with a 2-second ramp)
  - Carrier    : STC passive output (translation locked, rotation free)
  - Planet gears: free rigid bodies pushed by gear contact

Robot setup:
  - MARVIN_SHARPA (58 DOF) loaded as kinematic AffineBody + SoftTransformConstraint
  - Driven from ``trajectory_gear_sharpa.npz`` robot_qpos at 60 Hz
  - **All robot ↔ gear and robot self-collisions are DISABLED**

Visualisation:
  - SceneGUI shows all UIPC bodies
  - Separate visual-mesh FK overlay for the robot (semi-transparent)
  - ImGui panel: Run/Pause/Step, frame progress, speed

Usage
-----
    python sim_gear_with_robot.py [--dt FLOAT] [--speed FLOAT]

    --dt     UIPC simulation timestep in seconds (default: 0.01)
    --speed  Trajectory playback speed multiplier (default: 1.0)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DEMO_ASSETS = _HERE.parent                            # DemoAssets/
_GEAR = _DEMO_ASSETS / "planetary_gear"
_GEAR_ASSETS = _GEAR / "assets"
_SHARPA_URDF = _DEMO_ASSETS / "marvin_sharpa_description" / "marvin_sharpa.urdf"
_DEFAULT_TRAJ = _GEAR / "trajectory_gear_sharpa_objorigin.npz"
_YOYO_SCRIPTS = _DEMO_ASSETS / "yoyo" / "scripts"
_WORKSPACE = _HERE / "output"

# Make yoyo/scripts importable (URDFController, replay_utils, asset_dir …)
if str(_YOYO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_YOYO_SCRIPTS))

# ---------------------------------------------------------------------------
# Gear parameters  (must match sim_planetary.py / planetary_gear.scad)
# ---------------------------------------------------------------------------

MODUL        = 3
SUN_TEETH    = 12
PLANET_TEETH = 9
NUM_PLANETS  = 3
GEAR_WIDTH   = 12          # mm
CARRIER_THICK = 4.5        # mm
FLANGE_THICK_MM = 6.0      # mm

ORBIT_R_MM  = MODUL * (SUN_TEETH + PLANET_TEETH) / 2   # = 31.5 mm
RING_TEETH  = SUN_TEETH + 2 * PLANET_TEETH             # = 30
REDUCTION   = 1.0 + RING_TEETH / SUN_TEETH             # = 3.5

MM_TO_M = 0.001     # unit conversion: 1 mm = 0.001 m  (never change this)

# OBJ files are in nominal millimetres but were upscaled by ×1.2 in the
# Genesis sim that captured trajectory_gear_sharpa.npz.  All geometric
# quantities derived from the OBJ (orbit radius, carrier offset, etc.) must
# use the same factor so the UIPC scene stays physically consistent with the
# recorded trajectory.
MESH_SCALE = MM_TO_M * 1.2  # = 0.0012  matches Genesis IPCGearEnvConfig

# Genesis simulation initial euler for all gears (from replay_gear_traj.py)
GEAR_ROTATION_DEG = 270.0

# Handle ball centre in OBJ local frame (mm * MESH_SCALE = metres).
# From planetary_gear.scad: handle_arm_len=60, handle_angle=60°,
#   _Z0 = gear_width(12) + handle_post_h(8) = 20 mm, handle_grip_h=18 mm
_HANDLE_BALL_OBJ_M = np.array(
    [60.0 * np.cos(np.radians(60.0)),  # ≈ 30 mm
     60.0 * np.sin(np.radians(60.0)),  # ≈ 52 mm
     38.0],                            # 20 + 18 mm
    dtype=np.float64,
) * MESH_SCALE

_RIGHT_ARM_JOINTS = [f"Joint{k}_R" for k in range(1, 8)]  # Joint1_R … Joint7_R
_IK_TARGET_LINK   = "Link7_R"   # child of Joint7_R (right wrist)

# Right-hand finger structure for MARVIN_SHARPA (sharpa hands).
# Keys are friendly names; "tip" is the URDF distal-phalanx link, "joints"
# is the ordered chain of actuated joints from palm to that tip.
_R_FINGER_CHAINS: dict[str, dict] = {
    "thumb":  {"tip": "right_thumb_DP",
               "joints": ["right_thumb_CMC_FE", "right_thumb_CMC_AA",
                          "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP"]},
    "index":  {"tip": "right_index_DP",
               "joints": ["right_index_MCP_FE",  "right_index_MCP_AA",
                          "right_index_PIP",  "right_index_DIP"]},
    "middle": {"tip": "right_middle_DP",
               "joints": ["right_middle_MCP_FE", "right_middle_MCP_AA",
                          "right_middle_PIP", "right_middle_DIP"]},
    "ring":   {"tip": "right_ring_DP",
               "joints": ["right_ring_MCP_FE",   "right_ring_MCP_AA",
                          "right_ring_PIP",   "right_ring_DIP"]},
    "pinky":  {"tip": "right_pinky_DP",
               "joints": ["right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA",
                          "right_pinky_PIP", "right_pinky_DIP"]},
}
_R_ALL_FINGER_JOINTS = [j for ch in _R_FINGER_CHAINS.values() for j in ch["joints"]]
# All right-arm joints eligible for setup modification
_R_SETUP_JOINTS = _RIGHT_ARM_JOINTS + _R_ALL_FINGER_JOINTS  # 7 + 22 = 29 joints

# Persistent grip-setup file and blend defaults
_SETUP_FILE           = _HERE / "grip_setup.json"
_BLEND_FRAMES_DEFAULT = 60   # N sim frames to interpolate from _Q_LAST to Q_SETUP


def _cube_mesh(half: float) -> tuple[np.ndarray, np.ndarray]:
    """Tiny axis-aligned cube for use as a gizmo handle marker."""
    h = float(half)
    V = np.array([[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],
                  [-h,-h, h],[h,-h, h],[h,h, h],[-h,h, h]], dtype=np.float64)
    F = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],
                  [0,4,5],[0,5,1],[2,6,7],[2,7,3],
                  [1,5,6],[1,6,2],[0,3,7],[0,7,4]], dtype=np.int32)
    return V, F

# Driven sun-gear parameters
DRIVEN_RPM       = -15.0
DRIVEN_RAD_PER_S = DRIVEN_RPM * 2 * np.pi / 60.0
DRIVEN_RAMP_TIME = 1.0          # seconds to ramp from 0 → full speed (after blend)
DRIVEN_WAIT_TIME = 0.5          # seconds to hold still before STC rotation starts
SETTLE_FRAMES    = 30           # let UIPC find its own equilibrium before recording

# IPC / material parameters
D_HAT          = 2e-4 * 1.2      # match Genesis IPCGearEnvConfig: 2e-4 * 1.2 = 2.4e-4
DT_DEFAULT     = 0.01
ABD_KAPPA      = 100.0          # MPa
STC_DRIVEN     = np.array([0.0, 100.0], dtype=np.float64)
STC_PASSIVE    = np.array([10000.0, 0.0],    dtype=np.float64)

# Robot
ROBOT_BASE_Z    = 1.08
ROBOT_STC_STR   = np.array([120.0, 120.0], dtype=np.float64)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_z_4x4(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    tf = np.eye(4, dtype=np.float64)
    tf[0, 0] =  c;  tf[0, 1] = -s
    tf[1, 0] =  s;  tf[1, 1] =  c
    return tf


def _planet_transform(index: int) -> tuple[np.ndarray, np.ndarray]:
    """Initial rotation matrix and translation (metres) for planet gear *index*."""
    orbit_angle = np.radians(index * 360.0 / NUM_PLANETS)
    self_rot    = np.radians(index * 360.0 * SUN_TEETH / PLANET_TEETH)
    c, s = np.cos(self_rot), np.sin(self_rot)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    tx = ORBIT_R_MM * MESH_SCALE * np.cos(orbit_angle)
    ty = ORBIT_R_MM * MESH_SCALE * np.sin(orbit_angle)
    return R, np.array([tx, ty, 0.0])


def _genesis_pose_to_4x4(pos_quat: np.ndarray) -> np.ndarray:
    """Build a 4x4 rigid transform from a Genesis rigid-body pose vector.

    Genesis stores rigid-body state as ``[px, py, pz, qw, qx, qy, qz]``
    (position first, then quaternion with **scalar (w) first**).
    """
    px, py, pz       = pos_quat[:3].astype(np.float64)
    qw, qx, qy, qz   = pos_quat[3:].astype(np.float64)
    w, x, y, z = qw, qx, qy, qz
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]
    return T


def _build_qpos_joint_order(joint_names: list[str]) -> list[str]:
    """Return joint names in Genesis qpos column order for MARVIN_SHARPA.

    Loaded from ``genesis_joint_order.json``.  Arm joints are interleaved
    (Joint1_R, Joint1_L, Joint2_R, …) and finger joints follow a
    breadth-first grouping — neither URDF declaration order nor naive
    interleaving matches the full 58-DOF Genesis ordering.
    """
    import json as _json
    for candidate in (
        _GEAR / "genesis_joint_order.json",
        _DEMO_ASSETS / "marvin_sharpa_description" / "genesis_joint_order.json",
    ):
        if candidate.exists():
            names = _json.loads(candidate.read_text(encoding="utf-8"))
            print(f"  [joint-order] loaded {len(names)} joints from {candidate.name}")
            return names
    print("  [joint-order] WARNING: genesis_joint_order.json not found, "
          "falling back to URDF declaration order (may be wrong!)")
    return list(joint_names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Planetary gear UIPC sim + MARVIN_SHARPA teleop replay"
    )
    parser.add_argument("--dt",    type=float, default=DT_DEFAULT,
                        help="UIPC simulation timestep (default: 0.01 s)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Trajectory playback speed multiplier (default: 1.0)")
    parser.add_argument("--traj",  type=str, default=str(_DEFAULT_TRAJ),
                        help="Path to trajectory .npz")
    parser.add_argument("--robot-viz", action="store_true",
                        help="Robot is visualization-only FK playback; "
                             "no IPC bodies are created and the robot does "
                             "not participate in the UIPC simulation. "
                             "Gear physics still runs normally.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI: run IK track phase and write "
                             "per-frame robot_qpos + gear transforms to --output.")
    parser.add_argument("--output", type=str, default="ik_sequence.npz",
                        help="Output .npz path for headless mode (default: ik_sequence.npz)")
    parser.add_argument("--ik-frames", type=int, default=600,
                        help="Number of IK track frames to record in headless mode "
                             "(default: 600 = 6 s at dt=0.01)")
    parser.add_argument("--init-frame", type=int, default=488,
                        help="Use this trajectory frame to initialise gear transforms "
                             "(default: 699 — near end of teleop, gears assembled)")
    args = parser.parse_args()
    sim_dt     = args.dt
    play_speed = args.speed
    robot_viz  = args.robot_viz

    # ------------------------------------------------------------------
    # Load trajectory + build joint schedule
    # ------------------------------------------------------------------
    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    print(f"[traj] Loading {traj_path.name} ...")
    traj_npz  = np.load(str(traj_path))
    sim_times = traj_npz["sim_time"]           # (N,)
    robot_qpos = traj_npz["robot_qpos"]        # (N, 58)
    n_traj = len(sim_times)
    traj_dt = float(sim_times[1] - sim_times[0]) if n_traj > 1 else 1 / 60.0
    print(f"[traj] {n_traj} frames  traj_dt={traj_dt:.5f} s  "
          f"qpos {robot_qpos.shape}  play_speed={play_speed}x")

    # ------------------------------------------------------------------
    # Gear alignment: use ring_gear's world pose from the trajectory to
    # place the entire gear assembly so it matches the teleop scene.
    #
    # OBJ meshes are kept in their canonical local frame; initial world placement
    # is applied via geo.transforms() (not vertex mutation) after creation.
    # ------------------------------------------------------------------
    init_frame = min(args.init_frame, n_traj - 1)
    ring_pose_0  = traj_npz["rigid_ring_gear"][init_frame]
    print(f"[align] init_frame={init_frame}  ring_gear pos={ring_pose_0[:3]}  "
          f"quat(wxyz)={ring_pose_0[3:]}")

    def _traj_pose_to_T(key: str) -> np.ndarray:
        """Read pose at init_frame from the (already converted) trajectory.

        The trajectory is in OBJ-origin convention (convert_trajectory.py),
        so pos = OBJ-origin world position and q = visual rotation.
        Directly usable as UIPC body transform.
        """
        traj_key = f"rigid_{key}"
        pose = traj_npz[traj_key][init_frame].astype(np.float64)
        return _genesis_pose_to_4x4(pose)


    # Build sim-rate schedule: linearly interpolate trajectory at sim_dt
    effective_traj_dt = traj_dt / max(play_speed, 1e-6)
    steps_per_frame = max(1, int(np.round(effective_traj_dt / sim_dt)))
    total_sim_frames = steps_per_frame * (n_traj - 1) + 1
    print(f"[traj] sim_dt={sim_dt} s  steps_per_traj_frame={steps_per_frame}  "
          f"total_sim_frames={total_sim_frames}")

    # ------------------------------------------------------------------
    # UIPC engine / world / scene
    # ------------------------------------------------------------------
    from uipc import Animation, Logger, view
    from uipc import Vector3
    import uipc.builtin as builtin
    from uipc.core import Engine, World, Scene
    from uipc.geometry import SimplicialComplex, SimplicialComplexIO, label_surface, ground
    from uipc.constitution import AffineBodyConstitution, SoftTransformConstraint
    from uipc.gui import SceneGUI
    from uipc.unit import MPa
    from uipc import Transform

    Logger.set_level(Logger.Level.Info)
    _WORKSPACE.mkdir(parents=True, exist_ok=True)

    engine = Engine("cuda", str(_WORKSPACE))
    world  = World(engine)

    config = Scene.default_config()
    config["dt"]                           = sim_dt
    config["contact"]["d_hat"]             = D_HAT
    config["contact"]["enable"]            = True
    config["contact"]["friction"]["enable"] = True
    config["gravity"]                      = [[0.0], [0.0], [-9.8]]
    config["sanity_check"]["enable"]       = True
    config["linear_system"]["tol_rate"]    = 1e-3
    config["newton"]["max_iter"]           = 32
    scene = Scene(config)

    abd = AffineBodyConstitution()
    stc = SoftTransformConstraint()

    # ------------------------------------------------------------------
    # Contact tabular
    # ------------------------------------------------------------------
    ct = scene.contact_tabular()
    ct.default_model(0.001, 1e9)        # match Genesis coup_friction=0.001
    default_elem = ct.default_element()

    # Ring gear is fixed — separate element, self-contact disabled
    fixed_elem = ct.create("fixed_group")
    ct.insert(fixed_elem, fixed_elem, 0.0, 0.0, False)

    # Robot contact element — will be created AFTER create_ipc_bodies(),
    # so we stash it and insert disable pairs later.
    # (See "Disable robot contact" block below.)

    # ------------------------------------------------------------------
    # Load gear meshes   (mm → m via unit_tf)
    # ------------------------------------------------------------------
    unit_tf = Transform.Identity()
    unit_tf.scale(MESH_SCALE)
    io = SimplicialComplexIO(unit_tf)

    # Determine roles in "sun_in_carrier_out" drive mode
    driven_name  = "sun_gear"
    fixed_name   = "ring_gear"

    _obj_files = {
        "sun_gear":  "sun_gear_handle_v2.obj",
        "ring_gear": "ring_gear.obj",
        "carrier":   "carrier.obj",
    }

    driven_obj  = None
    driven_gs   = None   # geometry slot of the driven sun gear (for IK ball tracking)
    carrier_gs  = None   # geometry slot of the carrier (for headless export)
    ring_gs     = None   # geometry slot of the ring gear (for headless export)
    planet_gs_list: list = []   # geometry slots of planet gears (for headless export)
    # passive_obj removed — carrier is now a free body, physically supported by support_pin

    print("Loading gear parts ...")
    for pname in ("sun_gear", "ring_gear", "carrier"):
        obj_path = _GEAR_ASSETS / _obj_files[pname]
        mesh = io.read(str(obj_path))
        label_surface(mesh)
        # Vertex positions stay in the canonical OBJ local frame (scaled by
        # MESH_SCALE).  Initial world placement is written to geo.transforms()
        # after creation — never mutate mesh.positions() for placement.

        is_fixed_part = (pname == fixed_name)
        # STC: driven gear (sun) rotates, carrier is held at init position
        has_stc   = (pname == driven_name or pname == "carrier")
        strength  = STC_DRIVEN if pname == driven_name else STC_PASSIVE

        abd.apply_to(mesh, ABD_KAPPA * MPa)
        if has_stc:
            stc.apply_to(mesh, strength)

        if is_fixed_part:
            view(mesh.instances().find(builtin.is_fixed))[:] = 1
            fixed_elem.apply_to(mesh)
        else:
            default_elem.apply_to(mesh)

        obj_scene = scene.objects().create(pname)
        gs, _ = obj_scene.geometries().create(mesh)

        # Place each gear using its Genesis pose at init_frame
        T_part = _traj_pose_to_T(pname)
        view(gs.geometry().transforms())[0] = T_part

        if pname == driven_name:
            driven_obj = obj_scene
            driven_gs  = gs      # keep geometry slot for IK ball tracking
        if pname == "carrier":
            carrier_obj = obj_scene
            carrier_gs = gs
        if pname == fixed_name:
            ring_gs = gs

        nverts = mesh.vertices().size()
        tag = "FIXED" if is_fixed_part else ("DRIVE" if pname == driven_name else " FREE")
        print(f"  [{tag:5s}] {pname:12s}  {nverts:5d} verts  "
              f"T[:3,3]={np.round(T_part[:3,3],4)}")

    # Planet gears — placed from Genesis trajectory at init_frame
    planet_objs: list = []
    for i in range(NUM_PLANETS):
        T_planet = _traj_pose_to_T(f"planet_gear_{i}")

        mesh = io.read(str(_GEAR_ASSETS / "planet_gear_v2.obj"))
        label_surface(mesh)
        abd.apply_to(mesh, ABD_KAPPA * MPa)
        default_elem.apply_to(mesh)
        pobj = scene.objects().create(f"planet_gear_{i}")
        gs, _ = pobj.geometries().create(mesh)
        view(gs.geometry().transforms())[0] = T_planet
        planet_objs.append(pobj)
        planet_gs_list.append(gs)
        nverts = mesh.vertices().size()
        print(f"  [ FREE] planet_gear_{i}  {nverts:5d} verts  "
              f"T[:3,3]={np.round(T_planet[:3,3],4)}")

    # Support pin — fixed shaft providing rotation axis + vertical support
    # for both sun gear and carrier via IPC contact.
    # Pin local frame: bottom at z=0.
    # Assembly placement: bottom at z = -(GEAR_WIDTH/2 + FLANGE_THICK_MM) * MESH_SCALE
    pin_obj_path = _GEAR_ASSETS / "support_pin.obj"
    if not pin_obj_path.exists():
        raise FileNotFoundError(f"support_pin.obj not found: {pin_obj_path}")
    pin_mesh = io.read(str(pin_obj_path))
    label_surface(pin_mesh)
    abd.apply_to(pin_mesh, ABD_KAPPA * MPa)
    view(pin_mesh.instances().find(builtin.is_fixed))[:] = 1
    default_elem.apply_to(pin_mesh)   # contacts sun gear bore + carrier bore

    pin_obj_scene = scene.objects().create("support_pin")
    pin_gs, _ = pin_obj_scene.geometries().create(pin_mesh)

    T_pin = _genesis_pose_to_4x4(traj_npz["rigid_support_pin"][init_frame].astype(np.float64))
    view(pin_gs.geometry().transforms())[0] = T_pin
    print(f"  [FIXED] support_pin     {pin_mesh.vertices().size():5d} verts  "
          f"T[:3,3]={np.round(T_pin[:3,3],4)}")

    # Ground plane
    gnd_obj = scene.objects().create("ground")
    gnd_obj.geometries().create(ground(-0.5, Vector3.UnitZ()))

    # ------------------------------------------------------------------
    # Load robot — visual FK controller (always created; provides joint order)
    # ------------------------------------------------------------------
    if not _SHARPA_URDF.exists():
        raise FileNotFoundError(f"URDF not found: {_SHARPA_URDF}")

    print(f"\nLoading robot URDF: {_SHARPA_URDF.name} ...")
    from urdf_controller import URDFController   # from yoyo/scripts

    root_tf = np.eye(4, dtype=np.float64)
    root_tf[2, 3] = ROBOT_BASE_Z

    # vis_ctrl is always created; it drives the polyscope visual overlay and
    # provides the joint order used by both sim and viz modes.
    vis_ctrl = URDFController(str(_SHARPA_URDF), mesh_source="visual")
    vis_ctrl.set_root_transform(root_tf)
    joint_order = _build_qpos_joint_order(vis_ctrl.joint_names)
    n_use = min(len(joint_order), robot_qpos.shape[1])
    vis_ctrl.set_joint_positions(
        {joint_order[i]: float(robot_qpos[-1, i]) for i in range(n_use)}
    )
    print(f"  URDF joints: {len(vis_ctrl.joint_names)}   "
          f"npz qpos cols: {robot_qpos.shape[1]}   mapped: {n_use}")
    print(f"  Arm joints (first 14): {joint_order[:14]}")
    # IK tracking setup
    # Offset = (末帧腕位置) - (UIPC 仿真 t=0 时的 handle ball 位置)
    # 这样保证 t=0 时 IK 目标 = 末帧腕位置 → 点击 Run 不会瞬移
    # IK tracking — 夹爪中心(gripper-center)模型
    #
    # p_rel = ball 在 wrist 局部坐标系下的固定坐标（setup 时计算）
    #
    # 每帧目标：
    #   wrist_origin + R_wrist @ p_rel  = ball_world(t)
    # ↔ target_wrist = ball_world(t) - R_wrist_current @ p_rel
    #
    # 这样无论 wrist 如何转动，"夹爪中心"始终追着 ball_world(t) 这个世界坐标点。
    T_sun_init = _traj_pose_to_T("sun_gear")
    _ball_init_pos  = T_sun_init[:3, :3] @ _HANDLE_BALL_OBJ_M + T_sun_init[:3, 3]
    T_wrist_init    = vis_ctrl.get_link_transform(_IK_TARGET_LINK)   # 4×4
    wrist_init_pos  = T_wrist_init[:3, 3]
    R_wrist_init    = T_wrist_init[:3, :3]
    # p_rel: ball 在 wrist local frame 的坐标（常数）
    _IK_P_REL       = R_wrist_init.T @ (_ball_init_pos - wrist_init_pos)

    print(f"\n[IK] ball_init   = {np.round(_ball_init_pos, 4)}")
    print(f"[IK] wrist_init  = {np.round(wrist_init_pos,  4)}")
    print(f"[IK] p_rel(wrist local) = {np.round(_IK_P_REL, 4)}")

    # Robot starts (and stays) at the LAST trajectory frame for the whole
    # simulation.  The right arm is then driven by IK; everything else is frozen.
    _Q_LAST = robot_qpos[-1].copy()

    # _ik_cache removed: seed from Q_SETUP every frame, no stale state.
    # Warm-start seed for IK: stores right-arm joints from the PREVIOUS solve.
    # Cleared when setup changes so a fresh start uses Q_SETUP.
    _prev_arm: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Sim-phase state machine
    #   "setup"  – pre-simulation, gizmos active, user adjusts grip
    #   "blend"  – sim running, interpolating _Q_LAST → Q_SETUP over N frames
    #   "track"  – sim running, IK tracking handle ball
    # ------------------------------------------------------------------
    # Index _Q_LAST by joint name for quick lookup
    _Q_LAST_BY_NAME: dict[str, float] = {
        joint_order[i]: float(_Q_LAST[i]) for i in range(n_use)
    }
    # Mutable sim-phase state
    _phase        = ["setup"]        # "setup" | "blend" | "track"
    _blend_total  = [_BLEND_FRAMES_DEFAULT]
    _blend_done   = [0]
    _Q_SETUP:  dict[str, float] = {}   # saved grip setup (right arm + fingers)
    _ik_p_rel       = [_IK_P_REL.copy()]     # mutable p_rel, recomputed after blend
    _ik_wrist_rot   = [R_wrist_init.copy()]  # locked wrist rotation (from setup)

    # Load existing setup from file
    if _SETUP_FILE.exists():
        try:
            _saved = json.loads(_SETUP_FILE.read_text(encoding="utf-8"))
            _Q_SETUP.update(_saved.get("joints", {}))
            _blend_total[0] = int(_saved.get("blend_frames", _BLEND_FRAMES_DEFAULT))
            print(f"[setup] Loaded grip setup from {_SETUP_FILE.name}  "
                  f"({len(_Q_SETUP)} joints, blend={_blend_total[0]} frames)")
        except Exception as e:
            print(f"[setup] Could not load {_SETUP_FILE.name}: {e}")

    def _save_setup() -> None:
        """Persist current right-arm + finger joints to grip_setup.json."""
        data = {
            "joints": {j: vis_ctrl.get_joint_positions().get(j, 0.0)
                       for j in _R_SETUP_JOINTS},
            "blend_frames": _blend_total[0],
        }
        _Q_SETUP.update(data["joints"])
        _SETUP_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[setup] Saved {len(data['joints'])} joints -> {_SETUP_FILE.name}")

    # ------------------------------------------------------------------
    # Robot IPC bodies  (only when not --robot-viz)
    # ------------------------------------------------------------------
    ctrl: "URDFController | None" = None   # type: ignore[assignment]

    if not robot_viz:
        ctrl = URDFController(str(_SHARPA_URDF), mesh_source="collision")
        ctrl.set_root_transform(root_tf)
        ctrl.set_joint_positions(
            {joint_order[i]: float(robot_qpos[-1, i]) for i in range(n_use)}
        )

        ctrl.create_ipc_bodies(
            scene,
            object_prefix="robot_link",
            stc_strength=ROBOT_STC_STR,
        )
        print(f"  IPC robot bodies: {len(ctrl.bindings)}")

        # Disable robot ↔ everything contact (must be after create_ipc_bodies)
        robot_contact = ctrl._robot_contact
        ct.insert(robot_contact, robot_contact, 0.0, 0.0, False)   # self-collision
        ct.insert(robot_contact, default_elem,  0.0, 0.0, False)   # vs gear parts
        ct.insert(robot_contact, fixed_elem,    0.0, 0.0, False)   # vs ring gear
        print("  Robot contact: ALL DISABLED")
        print("  Robot mode: IPC simulation (kinematic ABD+STC bodies)")
    else:
        print("  Robot mode: visualization only (FK polyscope playback, no IPC bodies)")

    # ------------------------------------------------------------------
    # Gear animators
    # ------------------------------------------------------------------
    animator = scene.animator()

    # Frame counter: set to world.frame() when track phase begins;
    # driven_animation counts rotation time from this point.
    _track_start_frame = [0]

    def driven_animation(info: Animation.UpdateInfo) -> None:
        """Sun gear rotation:
        - setup / blend : hold still at T_sun_init (robot arm transitions first)
        - track         : rotate with ramp, elapsed time counted from track start
        """
        geo  = info.geo_slots()[0].geometry()
        is_c = view(geo.instances().find(builtin.is_constrained))
        aim_t = view(geo.instances().find(builtin.aim_transform))
        is_c[0] = 1

        if _phase[0] in ("setup", "blend"):
            aim_t[0] = T_sun_init
            return

        # track phase: wait, then rotate with ramp
        track_frame = max(0, info.frame() - _track_start_frame[0])
        t = track_frame * info.dt()
        if t < DRIVEN_WAIT_TIME:
            aim_t[0] = T_sun_init
            return
        t -= DRIVEN_WAIT_TIME
        if t < DRIVEN_RAMP_TIME:
            frac  = t / DRIVEN_RAMP_TIME
            angle = DRIVEN_RAD_PER_S * 0.5 * frac * t
        else:
            angle_ramp = DRIVEN_RAD_PER_S * 0.5 * DRIVEN_RAMP_TIME
            angle = angle_ramp + DRIVEN_RAD_PER_S * (t - DRIVEN_RAMP_TIME)
        aim_t[0] = T_sun_init @ _rotation_z_4x4(angle)

    animator.insert(driven_obj, driven_animation)

    # Carrier: hold at init_frame position via STC
    T_carrier_init = _traj_pose_to_T("carrier")

    def carrier_animation(info: Animation.UpdateInfo) -> None:
        geo  = info.geo_slots()[0].geometry()
        is_c = view(geo.instances().find(builtin.is_constrained))
        aim_t = view(geo.instances().find(builtin.aim_transform))
        is_c[0] = 1
        aim_t[0] = T_carrier_init

    animator.insert(carrier_obj, carrier_animation)

    # ------------------------------------------------------------------
    # Robot animator  (only when robot participates in IPC sim)
    # ------------------------------------------------------------------
    if ctrl is not None:
        def on_robot_frame(info: Animation.UpdateInfo, robot_ctrl: URDFController) -> None:
            # Base: frozen at last trajectory frame
            robot_ctrl.set_joint_positions(
                {joint_order[i]: float(_Q_LAST[i]) for i in range(n_use)}
            )
            ph = _phase[0]
            if ph == "blend":
                # Interpolate right arm + fingers from _Q_LAST towards Q_SETUP
                alpha = _blend_done[0] / max(1, _blend_total[0])
                blend = {j: _Q_LAST_BY_NAME.get(j, 0.0) * (1 - alpha)
                            + _Q_SETUP.get(j, _Q_LAST_BY_NAME.get(j, 0.0)) * alpha
                         for j in _R_SETUP_JOINTS if j in _Q_SETUP}
                if blend:
                    robot_ctrl.set_joint_positions(blend)
            elif ph == "track":
                # Right arm + fingers frozen at Q_SETUP during advance;
                # _sync_display will snap to the actual IK solution afterwards.
                if _Q_SETUP:
                    robot_ctrl.set_joint_positions(
                        {j: _Q_SETUP[j] for j in _R_SETUP_JOINTS if j in _Q_SETUP}
                    )

        ctrl.bind_animator(scene, on_frame=on_robot_frame)
        # Snap IPC body initial transforms to FK world positions.
        ctrl.apply_to_scene(snap=True)

    # ------------------------------------------------------------------
    # Init world
    # ------------------------------------------------------------------
    print("\nInitialising world ...")
    world.init(scene)
    world.retrieve()

    # Sanity check
    valid = world.is_valid()
    if valid:
        print("  PASS  --  No initial penetration detected")
    else:
        print("  FAIL  --  Initial penetration detected! Proceeding anyway for visual inspection.")

    # Settle: run physics so free gears drop onto the carrier/pin under gravity
    # Diagnostic: print body transforms BEFORE settle
    from uipc import view as _diag_view
    _diag_slots = [("sun_gear", driven_gs), ("carrier", carrier_gs),
                   ("ring_gear", ring_gs)]
    for i, pgs in enumerate(planet_gs_list):
        _diag_slots.append((f"planet_{i}", pgs))
    print("\n  [BEFORE settle] body transforms (T[:3,3]):")
    for dname, dgs in _diag_slots:
        T = np.array(_diag_view(dgs.geometry().transforms())[0], dtype=np.float64)
        print(f"    {dname:15s}  pos={np.round(T[:3,3],5)}")

    if SETTLE_FRAMES > 0:
        print(f"  Settling {SETTLE_FRAMES} frames (no animation) ...")
        for _ in range(SETTLE_FRAMES):
            world.advance()
            world.retrieve()

    print("  [AFTER settle] body transforms (T[:3,3]):")
    for dname, dgs in _diag_slots:
        T = np.array(_diag_view(dgs.geometry().transforms())[0], dtype=np.float64)
        print(f"    {dname:15s}  pos={np.round(T[:3,3],5)}")

    # ------------------------------------------------------------------
    # Visual FK overlay  (vis_ctrl was already created above)
    # ------------------------------------------------------------------
    import polyscope as ps
    from polyscope import imgui

    vis_meshes: dict[str, tuple] = {}

    def _init_vis_meshes() -> None:
        tfs = vis_ctrl.get_mesh_transforms()
        for node in vis_ctrl.mesh_nodes:
            tf = tfs.get(node.node_name)
            if tf is None:
                continue
            wv = node.local_vertices @ tf[:3, :3].T + tf[:3, 3]
            lbl = f"vis_{node.node_name}"
            m = ps.register_surface_mesh(lbl, wv, node.faces)
            m.set_transparency(0.40)
            m.set_color((0.55, 0.62, 0.72))
            m.set_smooth_shade(True)
            vis_meshes[node.node_name] = (m, node)

    def _update_vis_meshes(q_row: np.ndarray) -> None:
        vis_ctrl.set_joint_positions(
            {joint_order[i]: float(q_row[i]) for i in range(n_use)}
        )
        tfs = vis_ctrl.get_mesh_transforms()
        for nname, (mesh, node) in vis_meshes.items():
            tf = tfs.get(nname)
            if tf is None:
                continue
            mesh.update_vertex_positions(node.local_vertices @ tf[:3, :3].T + tf[:3, 3])

    def _compute_ik_joints() -> None:
        """Compute IK for right arm to track the handle ball on the rotating gear.

        Uses _Q_LAST (last trajectory frame) as the frozen base pose.
        Only Joint1_R … Joint7_R are modified; everything else stays at _Q_LAST.
        """
        from uipc import view as uipc_view

        if driven_gs is None:
            return

        # Current ball world position (3D point, no orientation involved)
        T_body = np.array(
            uipc_view(driven_gs.geometry().transforms())[0],
            dtype=np.float64, copy=True,
        )
        ball_world = T_body[:3, :3] @ _HANDLE_BALL_OBJ_M + T_body[:3, 3]

        # Seed: left arm + both hands from _Q_LAST (frozen),
        # right fingers from Q_SETUP, right arm from previous solve (warm-start).
        vis_ctrl.set_joint_positions(
            {joint_order[i]: float(_Q_LAST[i]) for i in range(n_use)}
        )
        if _Q_SETUP:
            vis_ctrl.set_joint_positions(
                {j: _Q_SETUP[j] for j in _R_ALL_FINGER_JOINTS if j in _Q_SETUP}
            )
        if _prev_arm:
            vis_ctrl.set_joint_positions(_prev_arm)   # warm-start from last IK solve

        # TCP IK: position the "gripper centre" (ball) at ball_world
        target_wrist = ball_world - _ik_wrist_rot[0] @ _ik_p_rel[0]
        vis_ctrl.solve_ik(
            _IK_TARGET_LINK,
            target_wrist,
            target_orientation=_ik_wrist_rot[0],
            orientation_mode="all",
            arm_joints=_RIGHT_ARM_JOINTS,
        )

        # Save right-arm result as warm-start seed for next frame
        cur = vis_ctrl.get_joint_positions()
        _prev_arm.update({j: cur[j] for j in _RIGHT_ARM_JOINTS if j in cur})

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Headless IK sequence export  (--headless flag)
    # All inner functions (_recompute_track_offset, _compute_ik_joints …)
    # are now defined above; safe to call them here.
    # ------------------------------------------------------------------
    def _recompute_track_offset() -> None:
        """Recompute IK wrist-to-ball offset based on Q_SETUP wrist position.
        Also records the UIPC frame number so driven_animation counts rotation
        from this point (gear was held still during blend).
        """
        from uipc import view as uipc_view
        _track_start_frame[0] = world.frame()
        if _Q_SETUP:
            vis_ctrl.set_joint_positions({j: _Q_SETUP[j] for j in _RIGHT_ARM_JOINTS
                                          if j in _Q_SETUP})
        T_body = np.array(
            uipc_view(driven_gs.geometry().transforms())[0],
            dtype=np.float64, copy=True)
        ball_world = T_body[:3, :3] @ _HANDLE_BALL_OBJ_M + T_body[:3, 3]
        T_wrist    = vis_ctrl.get_link_transform(_IK_TARGET_LINK)
        _ik_p_rel[0]     = T_wrist[:3, :3].T @ (ball_world - T_wrist[:3, 3])
        _ik_wrist_rot[0] = T_wrist[:3, :3].copy()
        print(f"[IK] p_rel recomputed: {np.round(_ik_p_rel[0], 4)}")
        _compute_ik_joints()

    if args.headless:
        from uipc import view as uipc_view

        def _gs_to_genesis_pose(gs) -> np.ndarray:
            """4×4 body transform → Genesis [px,py,pz,qw,qx,qy,qz]."""
            T = np.array(uipc_view(gs.geometry().transforms())[0],
                         dtype=np.float64, copy=True)
            pos = T[:3, 3]
            R   = T[:3, :3]
            tr = R[0,0] + R[1,1] + R[2,2]
            if tr > 0:
                s  = 0.5 / np.sqrt(tr + 1.0)
                qw = 0.25 / s
                qx = (R[2,1] - R[1,2]) * s
                qy = (R[0,2] - R[2,0]) * s
                qz = (R[1,0] - R[0,1]) * s
            elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                qw = (R[2,1] - R[1,2]) / s; qx = 0.25 * s
                qy = (R[0,1] + R[1,0]) / s; qz = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                qw = (R[0,2] - R[2,0]) / s; qx = (R[0,1] + R[1,0]) / s
                qy = 0.25 * s;               qz = (R[1,2] + R[2,1]) / s
            else:
                s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                qw = (R[1,0] - R[0,1]) / s; qx = (R[0,2] + R[2,0]) / s
                qy = (R[1,2] + R[2,1]) / s; qz = 0.25 * s
            return np.array([*pos, qw, qx, qy, qz], dtype=np.float32)

        def _vis_to_qpos() -> np.ndarray:
            cur = vis_ctrl.get_joint_positions()
            return np.array([float(cur.get(joint_order[i], 0.0))
                             for i in range(n_use)], dtype=np.float32)

        # Jump straight to track phase; blend is the stitch script's job
        # (settle already ran after world.init above)
        _phase[0] = "track"
        _track_start_frame[0] = world.frame()
        _recompute_track_offset()

        n_ik = args.ik_frames
        rec_qpos    = np.zeros((n_ik, n_use), dtype=np.float32)
        rec_time    = np.zeros((n_ik,),       dtype=np.float32)
        rec_sun     = np.zeros((n_ik, 7),     dtype=np.float32)
        rec_car     = np.zeros((n_ik, 7),     dtype=np.float32)
        rec_ring    = np.zeros((n_ik, 7),     dtype=np.float32)
        rec_pin     = np.zeros((n_ik, 7),     dtype=np.float32)
        rec_planets = [np.zeros((n_ik, 7), dtype=np.float32)
                       for _ in range(NUM_PLANETS)]

        # Frame 0 = post-settle UIPC steady state (recorded before first IK advance)
        rec_qpos[0]  = _vis_to_qpos()
        rec_time[0]  = 0.0
        rec_sun[0]   = _gs_to_genesis_pose(driven_gs)
        rec_car[0]   = _gs_to_genesis_pose(carrier_gs)
        rec_ring[0]  = _gs_to_genesis_pose(ring_gs)
        rec_pin[0]   = _gs_to_genesis_pose(pin_gs)
        for pi, pgs in enumerate(planet_gs_list):
            rec_planets[pi][0] = _gs_to_genesis_pose(pgs)

        print(f"\n[headless] recording {n_ik} frames -> {args.output}")
        t_start = time.perf_counter()
        for step in range(1, n_ik):
            world.advance()
            world.retrieve()
            _compute_ik_joints()

            rec_qpos[step]  = _vis_to_qpos()
            rec_time[step]  = float((step) * sim_dt)
            rec_sun[step]   = _gs_to_genesis_pose(driven_gs)
            rec_car[step]   = _gs_to_genesis_pose(carrier_gs)
            rec_ring[step]  = _gs_to_genesis_pose(ring_gs)
            rec_pin[step]   = _gs_to_genesis_pose(pin_gs)
            for pi, pgs in enumerate(planet_gs_list):
                rec_planets[pi][step] = _gs_to_genesis_pose(pgs)

            if step % 100 == 0 or step == n_ik - 1:
                elapsed = time.perf_counter() - t_start
                print(f"  {step:5d}/{n_ik}  t={rec_time[step]:.3f}s  "
                      f"({elapsed:.1f}s wall)")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            sim_time          = rec_time,
            robot_qpos        = rec_qpos,
            rigid_sun_gear    = rec_sun,
            rigid_carrier     = rec_car,
            rigid_ring_gear   = rec_ring,
            rigid_support_pin = rec_pin,
            **{f"rigid_planet_gear_{i}": rec_planets[i] for i in range(NUM_PLANETS)},
        )
        print(f"[headless] saved {n_ik} frames -> {out_path}  "
              f"({out_path.stat().st_size/1024:.0f} KB)")
        return

    # ------------------------------------------------------------------
    # Polyscope + SceneGUI
    # ------------------------------------------------------------------
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height(-0.5)
    ps.set_window_size(1600, 1000)
    ps.set_automatically_compute_scene_extents(False)

    sgui = SceneGUI(scene, "split")
    sgui.register()
    sgui.set_edge_width(1.0)

    _init_vis_meshes()

    # ------------------------------------------------------------------
    # Handle-ball world-position marker
    # The SCAD-defined handle ball centre (_HANDLE_BALL_OBJ_M) is a fixed
    # point in the sun gear's OBJ local frame.  Its world position equals:
    #   ball_world = T_body[:3,:3] @ _HANDLE_BALL_OBJ_M + T_body[:3,3]
    # where T_body is the current UIPC body transform.  We render it as an
    # orange point cloud so the user can verify palm-tracking alignment.
    # ------------------------------------------------------------------
    _ball_init_pos = T_sun_init[:3, :3] @ _HANDLE_BALL_OBJ_M + T_sun_init[:3, 3]
    ps_ball = ps.register_point_cloud(
        "gear/handle_ball",
        _ball_init_pos[np.newaxis],   # shape (1, 3)
        radius=0.015,
    )
    ps_ball.set_color((1.0, 0.30, 0.0))   # vivid orange

    def _update_ball_marker() -> None:
        """Recompute handle-ball world position from the current UIPC sun gear
        body transform and refresh the polyscope point cloud."""
        if driven_gs is None:
            return
        from uipc import view as uipc_view
        T_body = np.array(
            uipc_view(driven_gs.geometry().transforms())[0],
            dtype=np.float64,
            copy=True,
        )
        ball_world = T_body[:3, :3] @ _HANDLE_BALL_OBJ_M + T_body[:3, 3]
        ps_ball.update_point_positions(ball_world[np.newaxis])

    # If a saved setup exists, apply it immediately so the robot starts
    # in the configured grip pose before any simulation.
    if _Q_SETUP:
        vis_ctrl.set_joint_positions(_Q_SETUP)
        tfs = vis_ctrl.get_mesh_transforms()
        for nname, (mesh, node) in vis_meshes.items():
            tf = tfs.get(nname)
            if tf is not None:
                mesh.update_vertex_positions(node.local_vertices @ tf[:3, :3].T + tf[:3, 3])

    # ------------------------------------------------------------------
    # Setup-phase gizmos
    #   wrist_gizmo  – 6-DOF cube at Link7_R; drives right-arm IK
    #   tip_gizmos   – translate-only cubes at each right fingertip; drive per-finger IK
    # ------------------------------------------------------------------
    _cube_V, _cube_F = _cube_mesh(0.018)     # wrist: 1.8 cm half-size
    _tip_V,  _tip_F  = _cube_mesh(0.007)     # fingertip: 0.7 cm half-size

    wrist_gizmo = ps.register_surface_mesh(
        "setup/wrist", _cube_V, _cube_F, smooth_shade=False)
    wrist_gizmo.set_color((1.0, 0.85, 0.15))
    wrist_gizmo.set_transparency(0.55)
    wrist_gizmo.set_transform(vis_ctrl.get_link_transform(_IK_TARGET_LINK))
    wrist_gizmo.set_transform_gizmo_enabled(False)
    _wg = wrist_gizmo.get_transformation_gizmo()
    _wg.set_allow_translation(True);  _wg.set_allow_rotation(True)
    _wg.set_allow_scaling(False);     _wg.set_interact_in_local_space(False)

    tip_gizmos: dict[str, ps.SurfaceMesh] = {}
    for fname, chain in _R_FINGER_CHAINS.items():
        sm = ps.register_surface_mesh(
            f"setup/tip_{fname}", _tip_V, _tip_F, smooth_shade=False)
        sm.set_color((1.0, 0.55, 0.15))
        sm.set_transparency(0.65)
        T = np.eye(4)
        T[:3, 3] = vis_ctrl.get_link_position(chain["tip"])
        sm.set_transform(T)
        sm.set_transform_gizmo_enabled(False)
        g = sm.get_transformation_gizmo()
        g.set_allow_translation(True);  g.set_allow_rotation(False)
        g.set_allow_scaling(False);     g.set_interact_in_local_space(False)
        tip_gizmos[fname] = sm

    # Track last-known transforms to detect gizmo drags
    _last_wrist_T  = [vis_ctrl.get_link_transform(_IK_TARGET_LINK).copy()]
    _last_tip_T    = {fn: np.eye(4) for fn in _R_FINGER_CHAINS}
    for fn, chain in _R_FINGER_CHAINS.items():
        _last_tip_T[fn][:3, 3] = vis_ctrl.get_link_position(chain["tip"])

    def _sync_gizmos_to_fk() -> None:
        """Move all gizmos to match the current vis_ctrl FK result."""
        T = vis_ctrl.get_link_transform(_IK_TARGET_LINK)
        wrist_gizmo.set_transform(T)
        _last_wrist_T[0] = T.copy()
        for fn, chain in _R_FINGER_CHAINS.items():
            pos = vis_ctrl.get_link_position(chain["tip"])
            Tt = np.eye(4); Tt[:3, 3] = pos
            tip_gizmos[fn].set_transform(Tt)
            _last_tip_T[fn] = Tt.copy()

    def _setup_show_gizmos(enabled: bool) -> None:
        wrist_gizmo.set_transform_gizmo_enabled(enabled)
        for sm in tip_gizmos.values():
            sm.set_transform_gizmo_enabled(enabled)
        wrist_gizmo.set_enabled(enabled)
        for sm in tip_gizmos.values():
            sm.set_enabled(enabled)

    # Collision mesh overlay: only when robot is in IPC sim mode
    if ctrl is not None:
        ctrl.register_visual_only_meshes(transparency=0.25, edge_width=0.5,
                                         color=(0.3, 0.85, 0.5))
        ctrl.update_visual_meshes()

    # Enable setup gizmos for the initial setup phase
    _setup_show_gizmos(True)

    # ------------------------------------------------------------------
    # GUI state
    # ------------------------------------------------------------------
    run = [False]
    steps_per_tick = [1]
    show_vis = [True]
    show_coll = [True]
    last_wall = [time.perf_counter()]

    def _sim_step() -> None:
        world.advance()
        world.retrieve()

    def _sync_display() -> None:
        ph = _phase[0]

        if ph == "blend":
            # Advance blend counter
            _blend_done[0] += 1
            alpha = min(1.0, _blend_done[0] / max(1, _blend_total[0]))
            blend = {j: _Q_LAST_BY_NAME.get(j, 0.0) * (1 - alpha)
                        + _Q_SETUP.get(j, _Q_LAST_BY_NAME.get(j, 0.0)) * alpha
                     for j in _R_SETUP_JOINTS if j in _Q_SETUP}
            vis_ctrl.set_joint_positions({joint_order[i]: float(_Q_LAST[i]) for i in range(n_use)})
            if blend:
                vis_ctrl.set_joint_positions(blend)
            if _blend_done[0] >= _blend_total[0]:
                _phase[0] = "track"
                _recompute_track_offset()
        else:
            # "track" phase: IK with updated offset
            _compute_ik_joints()

        # Refresh visual meshes
        tfs = vis_ctrl.get_mesh_transforms()
        for nname, (mesh, node) in vis_meshes.items():
            tf = tfs.get(nname)
            if tf is not None:
                mesh.update_vertex_positions(
                    node.local_vertices @ tf[:3, :3].T + tf[:3, 3])
        # Snap ctrl (IPC bodies) to match vis_ctrl immediately — eliminates the
        # one-frame lag that caused the transparent/solid position difference.
        if ctrl is not None:
            ctrl.set_joint_positions(vis_ctrl.get_joint_positions())
            ctrl.apply_to_scene(snap=True)   # sets both aim_transform + body transform
            ctrl.update_visual_meshes()
        _update_ball_marker()   # keep the orange tracking point in sync
        sgui.update()

    def gui_callback() -> None:
        ph = _phase[0]
        mode_tag = "viz-only" if robot_viz else "IPC sim"
        imgui.Text(f"=== Planetary Gear + MARVIN_SHARPA  [{mode_tag}] ===")
        imgui.Separator()

        # ============================================================
        # SETUP phase panel
        # ============================================================
        if ph == "setup":
            # ---- Poll gizmo transforms every frame (gui_callback runs every frame) ----
            robot_dirty = False

            T_wrist_now = wrist_gizmo.get_transform()
            if not np.allclose(T_wrist_now, _last_wrist_T[0], atol=1e-6):
                _last_wrist_T[0] = T_wrist_now.copy()
                # Seed from current pose, then solve right-arm IK
                vis_ctrl.set_joint_positions(_Q_LAST_BY_NAME)
                if _Q_SETUP:
                    vis_ctrl.set_joint_positions(_Q_SETUP)
                vis_ctrl.solve_ik(
                    _IK_TARGET_LINK,
                    T_wrist_now[:3, 3],
                    target_orientation=T_wrist_now[:3, :3],
                    orientation_mode="all",
                    arm_joints=_RIGHT_ARM_JOINTS,
                )
                _sync_gizmos_to_fk()   # snap other gizmos to new FK
                robot_dirty = True

            for fname, chain in _R_FINGER_CHAINS.items():
                T_tip_now = tip_gizmos[fname].get_transform()
                if not np.allclose(T_tip_now, _last_tip_T[fname], atol=1e-6):
                    _last_tip_T[fname] = T_tip_now.copy()
                    vis_ctrl.solve_ik(
                        chain["tip"],
                        T_tip_now[:3, 3],
                        arm_joints=chain["joints"],
                    )
                    # Snap gizmo back to actual FK tip position
                    pos = vis_ctrl.get_link_position(chain["tip"])
                    Tt = np.eye(4); Tt[:3, 3] = pos
                    tip_gizmos[fname].set_transform(Tt)
                    _last_tip_T[fname] = Tt.copy()
                    robot_dirty = True

            if robot_dirty:
                tfs = vis_ctrl.get_mesh_transforms()
                for nname, (mesh, node) in vis_meshes.items():
                    tf = tfs.get(nname)
                    if tf is not None:
                        mesh.update_vertex_positions(
                            node.local_vertices @ tf[:3, :3].T + tf[:3, 3])
                _update_ball_marker()   # keep orange dot in sync during setup
                sgui.update()

            # ---- Draw UI ----
            imgui.TextColored((0.3, 1.0, 0.5, 1.0), "[ SETUP — Adjust grip before simulation ]")
            imgui.Separator()

            imgui.Text("Drag wrist gizmo  →  right-arm IK")
            imgui.Text("Drag fingertip gizmos  →  per-finger IK")
            imgui.Separator()

            # Blend-frames slider
            c, v = imgui.SliderInt("Blend frames", _blend_total[0], 10, 300)
            if c:
                _blend_total[0] = int(max(10, v))

            imgui.Separator()

            # Save current grip angles and start sim
            if imgui.Button("Save Setup & Start Simulation"):
                _save_setup()
                _setup_show_gizmos(False)
                _prev_arm.clear()   # fresh warm-start from Q_SETUP
                _blend_done[0] = 0
                _phase[0] = "blend"
                run[0] = True
            imgui.SameLine()
            if imgui.Button("Save Setup Only"):
                _save_setup()

            imgui.Separator()
            # Finger joint sliders for fine-tuning
            if imgui.CollapsingHeader("Right finger angles"):
                cur = vis_ctrl.get_joint_positions()
                changed_joints: dict[str, float] = {}
                for fname, chain in _R_FINGER_CHAINS.items():
                    imgui.Text(fname)
                    lims = vis_ctrl.joint_limits
                    for jname in chain["joints"]:
                        if jname not in cur:
                            continue
                        lo, hi = lims.get(jname, (-np.pi, np.pi))
                        c2, v2 = imgui.SliderFloat(
                            f"##{jname}", cur.get(jname, 0.0), float(lo), float(hi))
                        if c2:
                            changed_joints[jname] = v2
                if changed_joints:
                    vis_ctrl.set_joint_positions(changed_joints)
                    _sync_gizmos_to_fk()
                    # refresh display
                    tfs = vis_ctrl.get_mesh_transforms()
                    for nname, (mesh, node) in vis_meshes.items():
                        tf = tfs.get(nname)
                        if tf is not None:
                            mesh.update_vertex_positions(
                                node.local_vertices @ tf[:3,:3].T + tf[:3,3])
            return

        # ============================================================
        # SIM phase panel
        # ============================================================
        if ph == "blend":
            progress = _blend_done[0] / max(1, _blend_total[0])
            imgui.TextColored((1.0, 0.9, 0.2, 1.0),
                              f"Blending to grip pose... {progress*100:.0f}%")
        else:
            imgui.TextColored((0.4, 1.0, 0.4, 1.0), "Tracking handle ball (IK)")
        imgui.Separator()

        # --- Run / Pause ---
        if imgui.Button("Pause" if run[0] else "Run "):
            run[0] = not run[0]
        imgui.SameLine()
        if imgui.Button("Step"):
            _sim_step()
            _sync_display()
        imgui.SameLine()
        if imgui.Button("Back to Setup"):
            run[0] = False
            _phase[0] = "setup"
            _blend_done[0] = 0
            _prev_arm.clear()   # discard warm-start when returning to setup
            # Restore Q_SETUP pose in vis_ctrl
            vis_ctrl.set_joint_positions({joint_order[i]: float(_Q_LAST[i]) for i in range(n_use)})
            if _Q_SETUP:
                vis_ctrl.set_joint_positions(_Q_SETUP)
            _sync_gizmos_to_fk()
            _setup_show_gizmos(True)
            tfs = vis_ctrl.get_mesh_transforms()
            for nname, (mesh, node) in vis_meshes.items():
                tf = tfs.get(nname)
                if tf is not None:
                    mesh.update_vertex_positions(node.local_vertices @ tf[:3,:3].T + tf[:3,3])

        # --- Steps per tick ---
        c, v = imgui.SliderInt("Steps / tick", steps_per_tick[0], 1, 8)
        if c:
            steps_per_tick[0] = int(max(1, v))

        # --- Progress ---
        sf   = world.frame()
        tf_s = sf * sim_dt
        imgui.Text(f"Sim frame {sf:5d}   t = {tf_s:.3f} s")

        # --- Visibility toggles ---
        imgui.Separator()
        c, v = imgui.Checkbox("Show visual meshes", show_vis[0])
        if c:
            show_vis[0] = v
            for _, (mesh, _) in vis_meshes.items():
                mesh.set_enabled(v)
        if ctrl is not None:
            imgui.SameLine()
            c, v = imgui.Checkbox("Show collision meshes", show_coll[0])
            if c:
                show_coll[0] = v
                for _, (mesh, _) in getattr(ctrl, "_visual_meshes", {}).items():
                    mesh.set_enabled(v)

        # --- Advance if running ---
        if run[0]:
            for _ in range(steps_per_tick[0]):
                _sim_step()
                if world.frame() >= total_sim_frames:
                    run[0] = False
                    print("[sim] Complete.")
                    break
            _sync_display()

    ps.set_user_callback(gui_callback)
    print("\nOpening Polyscope window …")
    ps.show()


if __name__ == "__main__":
    main()
