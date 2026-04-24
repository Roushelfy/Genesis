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
_DEFAULT_TRAJ = _GEAR / "trajectory_gear_sharpa.npz"
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

# Driven sun-gear parameters
DRIVEN_RPM       = -15.0
DRIVEN_RAD_PER_S = DRIVEN_RPM * 2 * np.pi / 60.0
DRIVEN_RAMP_TIME = 2.0          # seconds to ramp from 0 → full speed

# IPC / material parameters
D_HAT          = 2e-4 * 1.2
DT_DEFAULT     = 0.01
ABD_KAPPA      = 100.0          # MPa
STC_DRIVEN     = np.array([100.0, 100.0], dtype=np.float64)
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
    """Build Genesis qpos column → joint-name mapping for MARVIN_SHARPA.

    Arm joints are interleaved R/L (Joint1_R, Joint1_L, ..., Joint7_R, Joint7_L)
    followed by all remaining finger joints in URDF declaration order.
    """
    joint_set = set(joint_names)
    ordered: list[str] = []
    for k in range(1, 8):
        for side in ("R", "L"):
            name = f"Joint{k}_{side}"
            if name in joint_set:
                ordered.append(name)
    arm_set = set(ordered)
    for name in joint_names:
        if name not in arm_set:
            ordered.append(name)
    return ordered


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
    ring_pose_0  = traj_npz["rigid_ring_gear"][0]   # constant; [px,py,pz,qw,qx,qy,qz]
    T_align      = _genesis_pose_to_4x4(ring_pose_0)
    print(f"[align] ring_gear world pose: pos={ring_pose_0[:3]}  "
          f"quat(wxyz)={ring_pose_0[3:]}")


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
    ct.default_model(0.0, 1e9)          # gear parts use the default element
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

    carrier_tz = -(GEAR_WIDTH / 2) * MESH_SCALE   # carrier offset below gear plane

    # Determine roles in "sun_in_carrier_out" drive mode
    driven_name  = "sun_gear"
    fixed_name   = "ring_gear"
    passive_name = "carrier"

    _obj_files = {
        "sun_gear":  "sun_gear_handle.obj",
        "ring_gear": "ring_gear.obj",
        "carrier":   "carrier.obj",
    }

    driven_obj  = None
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
        # Only the driven part gets STC; carrier is free (support_pin provides
        # the rotation axis and vertical support through IPC contact).
        has_stc   = (pname == driven_name)
        strength  = STC_DRIVEN

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

        # Compute this part's initial world transform and snap it into place.
        # Carrier sits carrier_tz below the gear plane in the assembly frame.
        if pname == "carrier":
            T_local = np.eye(4, dtype=np.float64)
            T_local[2, 3] = carrier_tz
            T_part = T_align @ T_local
        else:
            T_part = T_align.copy()      # sun_gear / ring_gear at gear centre

        view(gs.geometry().transforms())[0] = T_part

        if pname == driven_name:
            driven_obj = obj_scene

        nverts = mesh.vertices().size()
        tag = "FIXED" if is_fixed_part else ("DRIVE" if pname == driven_name else " FREE")
        print(f"  [{tag:5s}] {pname:12s}  {nverts:5d} verts  "
              f"T[:3,3]={np.round(T_part[:3,3],4)}")

    # Planet gears
    planet_objs: list = []
    for i in range(NUM_PLANETS):
        R_orbit, t_orbit = _planet_transform(i)
        T_planet_local = np.eye(4, dtype=np.float64)
        T_planet_local[:3, :3] = R_orbit
        T_planet_local[:3, 3]  = t_orbit
        T_planet = T_align @ T_planet_local

        mesh = io.read(str(_GEAR_ASSETS / "planet_gear.obj"))
        label_surface(mesh)
        abd.apply_to(mesh, ABD_KAPPA * MPa)
        default_elem.apply_to(mesh)
        pobj = scene.objects().create(f"planet_gear_{i}")
        gs, _ = pobj.geometries().create(mesh)
        view(gs.geometry().transforms())[0] = T_planet
        planet_objs.append(pobj)
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

    T_pin_local = np.eye(4, dtype=np.float64)
    T_pin_local[2, 3] = -(GEAR_WIDTH / 2 + FLANGE_THICK_MM) * MESH_SCALE
    T_pin = T_align @ T_pin_local
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
        {joint_order[i]: float(robot_qpos[0, i]) for i in range(n_use)}
    )
    print(f"  URDF joints: {len(vis_ctrl.joint_names)}   "
          f"npz qpos cols: {robot_qpos.shape[1]}   mapped: {n_use}")
    print(f"  Arm joints (first 14): {joint_order[:14]}")

    # ------------------------------------------------------------------
    # Robot IPC bodies  (only when not --robot-viz)
    # ------------------------------------------------------------------
    ctrl: "URDFController | None" = None   # type: ignore[assignment]

    if not robot_viz:
        ctrl = URDFController(str(_SHARPA_URDF), mesh_source="collision")
        ctrl.set_root_transform(root_tf)
        ctrl.set_joint_positions(
            {joint_order[i]: float(robot_qpos[0, i]) for i in range(n_use)}
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

    def driven_animation(info: Animation.UpdateInfo) -> None:
        """Sun gear: rotate around the assembly Z-axis from t=0 with a ramp.

        aim_transform = T_align @ R_z(angle)

        Because rest vertices are in the canonical local frame (centred near
        origin), the current world transform is directly T_align @ R_z(angle).
        No conjugation is needed (unlike the old vertex-mutation approach).
        """
        geo = info.geo_slots()[0].geometry()
        is_c  = view(geo.instances().find(builtin.is_constrained))
        aim_t = view(geo.instances().find(builtin.aim_transform))
        t = info.frame() * info.dt()
        is_c[0] = 1
        if t < DRIVEN_RAMP_TIME:
            frac  = t / DRIVEN_RAMP_TIME
            angle = DRIVEN_RAD_PER_S * 0.5 * frac * t
        else:
            angle_ramp = DRIVEN_RAD_PER_S * 0.5 * DRIVEN_RAMP_TIME
            angle = angle_ramp + DRIVEN_RAD_PER_S * (t - DRIVEN_RAMP_TIME)
        aim_t[0] = T_align @ _rotation_z_4x4(angle)

    animator.insert(driven_obj, driven_animation)

    # ------------------------------------------------------------------
    # Robot animator  (only when robot participates in IPC sim)
    # ------------------------------------------------------------------
    if ctrl is not None:
        def on_robot_frame(info: Animation.UpdateInfo, robot_ctrl: URDFController) -> None:
            sim_frame = int(info.frame())
            traj_idx  = sim_frame // steps_per_frame
            sub_step  = sim_frame  % steps_per_frame
            traj_idx  = min(traj_idx, n_traj - 2)
            alpha = sub_step / steps_per_frame
            q = (1.0 - alpha) * robot_qpos[traj_idx] + alpha * robot_qpos[traj_idx + 1]
            robot_ctrl.set_joint_positions(
                {joint_order[i]: float(q[i]) for i in range(n_use)}
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

    # Collision mesh overlay: only when robot is in IPC sim mode
    if ctrl is not None:
        ctrl.register_visual_only_meshes(transparency=0.25, edge_width=0.5,
                                         color=(0.3, 0.85, 0.5))
        ctrl.update_visual_meshes()

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
        sf = world.frame()
        ti = min(sf // steps_per_frame, n_traj - 1)
        _update_vis_meshes(robot_qpos[ti])
        if ctrl is not None:
            ctrl.update_visual_meshes()
        sgui.update()

    def gui_callback() -> None:
        mode_tag = "viz-only" if robot_viz else "IPC sim"
        imgui.Text(f"=== Planetary Gear + MARVIN_SHARPA  [{mode_tag}] ===")
        imgui.Separator()

        # --- Run / Pause ---
        if imgui.Button("Pause" if run[0] else "Run "):
            run[0] = not run[0]
        imgui.SameLine()
        if imgui.Button("Step"):
            _sim_step()
            _sync_display()

        # --- Steps per tick ---
        c, v = imgui.SliderInt("Steps / tick", steps_per_tick[0], 1, 8)
        if c:
            steps_per_tick[0] = int(max(1, v))

        # --- Progress ---
        sf   = world.frame()
        tf_s = sf * sim_dt
        ti   = min(sf // steps_per_frame, n_traj - 1)
        traj_t = float(sim_times[ti])
        imgui.Text(
            f"Sim  frame {sf:6d} / {total_sim_frames}   "
            f"t = {tf_s:.3f} s"
        )
        imgui.Text(
            f"Traj frame {ti:5d} / {n_traj - 1}   "
            f"traj_t = {traj_t:.3f} s"
        )

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
                    print("[sim] Trajectory complete.")
                    break
            _sync_display()

    ps.set_user_callback(gui_callback)
    print("\nOpening Polyscope window …")
    ps.show()


if __name__ == "__main__":
    main()
