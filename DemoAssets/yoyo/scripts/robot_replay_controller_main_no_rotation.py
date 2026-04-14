"""Replay a recorded trajectory in the Polyscope/UIPC GUI.

Loads ``trajectory.npz`` (v5_init teleop format) recorded with marvin_sharpa
and replays it in a unified GUI with:

  - Trajectory scrubbing (FK-only, no physics)
  - User Scene transform editing
  - Reference OBJ overlay
  - UIPC simulation (start on demand, trajectory drives joints)

Usage:
    python robot_replay_controller_main.py --traj-dir ../v5_init
"""

from __future__ import annotations

import argparse
import atexit
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polyscope as ps
from polyscope import imgui

from replay_utils import (
    TrajectoryData,
    get_sharpa_urdf,
    load_genesis_joint_order,
    load_trajectory_npz,
    qpos_to_joint_dict,
    trajectory_to_frame_schedule,
)
from urdf_controller import URDFController
from urdf_gui import (
    SceneState,
    URDFGuiApp,
    build_scene_transform,
    closest_point_on_triangle,
    make_rotation,
    parse_obj,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_YOYO_DIR = _REPO_ROOT / "DemoAssets" / "yoyo"

_BALL_OBJ = _SCRIPT_DIR.parent / "v3" / "yoyo-ball.obj"
_STRING_OBJ = _SCRIPT_DIR.parent / "v3" / "yoyo_string_6153.obj"
_BEARING_OUTER_OBJ = _SCRIPT_DIR.parent / "v1" / "bearing_outer.obj"
_BEARING_SPHERES_OBJ = _SCRIPT_DIR.parent / "v1" / "bearing_spheres.obj"

_ROBOT_BASE_POS = np.array([0.0, 0.0, 1.08], dtype=np.float64)

_SIM_LINK_PATTERNS = ["*right_hand*", "*right_thumb*", "*right_index*",
                      "*right_middle*", "*right_ring*", "*right_pinky*",
                      "*left_hand*", "*left_thumb*", "*left_index*",
                      "*left_middle*", "*left_ring*", "*left_pinky*"]
_STC_STRENGTH = np.array([120.0, 120.0], dtype=np.float64)
_STITCH_BINDING_PATTERN = "*right_middle_MP*"

_CONFIG_DIR = _SCRIPT_DIR / "_robot_replay_controller"
_CONFIG_FILE = _CONFIG_DIR / "gui_config.json"

_ROTATION_VELOCITY = -2.0 * np.pi * 10


# ---------------------------------------------------------------------------
# Yoyo UIPC scene (for sim mode)
# ---------------------------------------------------------------------------


def load_yoyo_scene(scene, world, ss: SceneState) -> None:
    """Add the full yoyo (ball + string + bearings) to *scene* for UIPC sim."""
    from uipc import Animation, builtin, view
    from uipc.constitution import (
        AffineBodyConstitution,
        HookeanSpring,
        KirchhoffRodBending,
        RotatingMotor,
    )
    from uipc.geometry import (
        SimplicialComplexIO,
        apply_region,
        label_region,
        label_surface,
        mesh_partition,
    )
    from uipc.unit import GPa, MPa

    ss.clear()
    ss.fem_objects = {"yoyo_string"}
    io = SimplicialComplexIO()

    cfg = scene.config()
    view(cfg.find("dt"))[0] = 0.001
    view(cfg.find("contact/enable"))[0] = 1
    view(cfg.find("contact/friction/enable"))[0] = 1
    view(cfg.find("contact/d_hat"))[0] = 0.0001
    view(cfg.find("newton/velocity_tol"))[0] = 0.1
    view(cfg.find("newton/transrate_tol"))[0] = 10
    view(cfg.find("sanity_check/enable"))[0] = 1
    # view(cfg.find("extras/debug/dump_surface"))[0] = 1

    tabular = scene.contact_tabular()
    tabular.default_model(0.3, 1e9, True)

    ball_contact = tabular.create("ball")
    string_contact = tabular.create("string")
    bearing_contact = tabular.create("bearing")
    tabular.insert(ball_contact, string_contact, 0.7, 800.0 * MPa, True)
    tabular.insert(string_contact, string_contact, 0.2, 800.0 * MPa, True)
    tabular.insert(bearing_contact, bearing_contact, 0.01, 800.0 * MPa, True)
    tabular.insert(bearing_contact, ball_contact, 0.05, 800.0 * MPa, True)
    tabular.insert(bearing_contact, string_contact, 0.8, 800.0 * MPa, True)

    robot_contact = tabular.default_element()
    tabular.insert(ball_contact, robot_contact, 0.5, 800.0 * MPa, True)
    tabular.insert(string_contact, robot_contact, 0.7, 800.0 * MPa, True)
    tabular.insert(bearing_contact, robot_contact, 0.3, 800.0 * MPa, True)

    ball_mesh = io.read(str(_BALL_OBJ))
    label_surface(ball_mesh)
    AffineBodyConstitution().apply_to(ball_mesh, 200.0 * MPa)
    ball_contact.apply_to(ball_mesh)
    view(ball_mesh.instances().find(builtin.is_dynamic))[0] = 1

    motor = RotatingMotor()
    motor.apply_to(
        ball_mesh,
        strength=1,
        motor_axis=np.array([[0.0], [1.0], [0.0]], dtype=np.float64),
        motor_rot_vel=_ROTATION_VELOCITY,
    )

    ball_obj = scene.objects().create("yoyo_ball")
    ball_gs, _ = ball_obj.geometries().create(ball_mesh)
    ss.objects["yoyo_ball"] = ball_obj
    ss.geo_slots["yoyo_ball"] = ball_gs
    ss.obj_sources["yoyo_ball"] = _BALL_OBJ
    ss.original_transforms["yoyo_ball"] = np.array(view(ball_mesh.transforms())[0], copy=True)

    def animate_yoyo_ball(info: Animation.UpdateInfo) -> None:
        geo = info.geo_slots()[0].geometry()
        view(geo.instances().find(builtin.is_constrained))[0] = 1
        RotatingMotor.animate(geo, info.dt())

    scene.animator().insert(ball_obj, animate_yoyo_ball)

    string_mesh = io.read(str(_STRING_OBJ))
    label_surface(string_mesh)
    HookeanSpring().apply_to(string_mesh, 1.0 * GPa, thickness=0.00038, mass_density=100.0)
    KirchhoffRodBending().apply_to(string_mesh, 1.0e4)
    string_contact.apply_to(string_mesh)
    mesh_partition(string_mesh, 16)

    string_obj = scene.objects().create("yoyo_string")
    string_gs, string_rest_gs = string_obj.geometries().create(string_mesh)
    ss.objects["yoyo_string"] = string_obj
    ss.geo_slots["yoyo_string"] = string_gs
    ss.rest_geo_slots["yoyo_string"] = string_rest_gs
    ss.obj_sources["yoyo_string"] = _STRING_OBJ
    ss.original_transforms["yoyo_string"] = np.array(view(string_mesh.transforms())[0], copy=True)
    ss.original_positions["yoyo_string"] = np.array(
        view(string_gs.geometry().positions()), copy=True
    ).reshape(-1, 3)

    bearing_outer_mesh = io.read(str(_BEARING_OUTER_OBJ))
    label_surface(bearing_outer_mesh)
    AffineBodyConstitution().apply_to(bearing_outer_mesh, 200.0 * MPa)
    bearing_contact.apply_to(bearing_outer_mesh)
    bearing_outer_obj = scene.objects().create("bearing_outer")
    bo_gs, _ = bearing_outer_obj.geometries().create(bearing_outer_mesh)
    ss.objects["bearing_outer"] = bearing_outer_obj
    ss.geo_slots["bearing_outer"] = bo_gs
    ss.obj_sources["bearing_outer"] = _BEARING_OUTER_OBJ
    ss.original_transforms["bearing_outer"] = np.array(
        view(bearing_outer_mesh.transforms())[0], copy=True
    )

    bearing_spheres_mesh = io.read(str(_BEARING_SPHERES_OBJ))
    label_surface(bearing_spheres_mesh)
    label_region(bearing_spheres_mesh)
    sphere_regions = apply_region(bearing_spheres_mesh)
    for i, sphere_mesh in enumerate(sphere_regions):
        label_surface(sphere_mesh)
        AffineBodyConstitution().apply_to(sphere_mesh, 200.0 * MPa)
        bearing_contact.apply_to(sphere_mesh)
        sp_obj = scene.objects().create(f"bearing_sphere_{i}")
        sp_gs, _ = sp_obj.geometries().create(sphere_mesh)
        ss.objects[f"bearing_sphere_{i}"] = sp_obj
        ss.geo_slots[f"bearing_sphere_{i}"] = sp_gs
        ss.original_transforms[f"bearing_sphere_{i}"] = np.array(
            view(sphere_mesh.transforms())[0], copy=True
        )
        split_obj_path = _SCRIPT_DIR.parent / "v3" / f"bearing_sphere_{i}.obj"
        if not split_obj_path.exists():
            io.write(str(split_obj_path), sphere_mesh)
        ss.obj_sources[f"bearing_sphere_{i}"] = split_obj_path

    print(
        f"[replay-scene] Yoyo loaded: ball + string + bearing_outer"
        f" + {len(sphere_regions)} bearing spheres"
    )


def setup_yoyo_stitch(scene, controller, ss: SceneState) -> None:
    """Stitch yoyo string vertex 0 to the nearest triangle on the right gripper."""
    # return

    from uipc import view
    from uipc.constitution import ElasticModuli, SoftVertexTriangleStitch
    from uipc.geometry import apply_transform

    binding = controller.find_binding(_STITCH_BINDING_PATTERN)
    if binding is None:
        print(f"[stitch] No binding matched '{_STITCH_BINDING_PATTERN}', skipping")
        return

    string_gs = ss.geo_slots.get("yoyo_string")
    string_rest_gs = ss.rest_geo_slots.get("yoyo_string")
    if string_gs is None or string_rest_gs is None:
        print("[stitch] yoyo_string not loaded, skipping")
        return

    gripper_geo_slot = binding.geo_slot
    gripper_rest_geo_slot = binding.rest_geo_slot
    string_geo = string_gs.geometry()
    gripper_geo = gripper_geo_slot.geometry()
    gripper_rest_geo = gripper_rest_geo_slot.geometry()

    gripper_tf = np.array(view(gripper_geo.transforms()), copy=False).reshape(-1, 4, 4)[0]
    view(gripper_rest_geo.transforms())[0] = gripper_tf

    string_pos = np.array(view(string_geo.positions()), copy=False).reshape(-1, 3)
    gripper_world = apply_transform(gripper_geo)[0]
    gripper_world_pos = np.array(view(gripper_world.positions()), copy=False).reshape(-1, 3)
    gripper_world_tris = np.array(
        view(gripper_world.triangles().topo()), copy=False
    ).reshape(-1, 3)

    v0 = string_pos[0]
    best_tri, best_dist = 0, float("inf")
    for ti in range(gripper_world_tris.shape[0]):
        tri_verts = gripper_world_pos[gripper_world_tris[ti]]
        cp = closest_point_on_triangle(v0, tri_verts[0], tri_verts[1], tri_verts[2])
        d = float(np.linalg.norm(v0 - cp))
        if d < best_dist:
            best_dist, best_tri = d, ti

    pairs = np.array([[0, best_tri]], dtype=np.int32)
    svts = SoftVertexTriangleStitch()
    stitch_geo = svts.create_geometry(
        (string_gs, gripper_geo_slot),
        (string_rest_gs, gripper_rest_geo_slot),
        pairs,
        ElasticModuli.youngs_poisson(1.0e6, 0.49),
    )
    stitch_obj = scene.objects().create("string_gripper_stitch")
    stitch_obj.geometries().create(stitch_geo)

    ss.stitch_vis["string_gs"] = string_gs
    ss.stitch_vis["gripper_gs"] = gripper_geo_slot
    ss.stitch_vis["pairs"] = pairs
    ss.stitch_vis["gripper_tris"] = np.array(
        view(gripper_geo.triangles().topo()), copy=True
    ).reshape(-1, 3)
    print(f"[stitch] string vertex 0 -> gripper triangle {best_tri} (dist={best_dist:.4f})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unified app
# ---------------------------------------------------------------------------


def run(
    traj: TrajectoryData,
    urdf_path: str,
    genesis_joint_names: list[str],
    sim_dt_override: float | None = None,
    init_frame: int = 0,
    play_speed: float = 1.0,
) -> None:
    """Unified GUI: trajectory scrubbing + scene setup + UIPC simulation."""
    from uipc import SceneIO, Timer, view
    from uipc.gui import SceneGUI

    from asset_dir import AssetDir

    # ── Persistent objects (survive scene rebuilds) ─────────────────────
    app = URDFGuiApp(
        urdf_path=urdf_path,
        sim_link_patterns=_SIM_LINK_PATTERNS,
        stc_strength=_STC_STRENGTH,
        load_scene_fn=load_yoyo_scene,
        setup_stitch_fn=setup_yoyo_stitch,
        record_file=_SCRIPT_DIR / "motion_keyframes.json",
        joint_file=_SCRIPT_DIR / "joint_angles.json",
        config_file=_CONFIG_FILE,
        entry_file=__file__,
    )

    output_dir = Path(AssetDir.output_path(__file__))

    controller = URDFController(urdf_path, mesh_source="collision")
    app._controller = controller
    app._joint_names = controller.joint_names
    app._limits = controller.joint_limits
    app._end_effectors = controller.find_end_effectors()
    print(f"[urdf-gui] Actuated joints ({len(app._joint_names)}): {app._joint_names}")
    print(f"[urdf-gui] End effectors: {app._end_effectors}")

    root_tf = np.eye(4, dtype=np.float64)
    root_tf[:3, 3] = _ROBOT_BASE_POS
    controller.set_root_transform(root_tf)

    init_frame = max(0, min(init_frame, traj.n_frames - 1))

    _FREEZE_ROBOT = True

    # ── Mutable UIPC context (rebuilt by _build_scene) ─────────────────
    ctx: dict[str, Any] = {}

    def _read_snapshot() -> dict | None:
        snap_json = _CONFIG_DIR / "scene_snapshot.json"
        if not snap_json.exists():
            return None
        try:
            return json.loads(snap_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            return None

    def _build_scene(snapshot: dict | None = None) -> None:
        """(Re)build the UIPC engine/world/scene. Applies *snapshot* if given."""
        app.scene_state.clear()
        controller.bindings.clear()

        if snapshot and snapshot.get("joints"):
            controller.set_joint_positions(snapshot["joints"])
        else:
            controller.set_joint_positions(
                qpos_to_joint_dict(traj.robot_qpos[init_frame], genesis_joint_names)
            )

        engine, world, scene = app._create_engine_scene(output_dir)
        app._setup_robot(scene)
        app._apply_scene_and_stitch(scene, world)

        if sim_dt_override is not None:
            view(scene.config().find("dt"))[0] = sim_dt_override

        sim_dt = float(view(scene.config().find("dt"))[0])

        effective_traj_dt = traj.dt / max(play_speed, 1e-6)
        schedule = trajectory_to_frame_schedule(
            traj.robot_qpos[init_frame:], genesis_joint_names,
            traj_dt=effective_traj_dt, sim_dt=sim_dt,
        )
        print(f"[replay] UIPC sim_dt={sim_dt}, traj dt={traj.dt}, play_speed={play_speed}x, "
              f"traj frames={traj.n_frames - init_frame} (from {init_frame}) "
              f"-> schedule={len(schedule)} frames")

        if snapshot:
            ss = app.scene_state
            for name, tf_list in snapshot.get("rigid_transforms", {}).items():
                gs = ss.geo_slots.get(name)
                if gs is None:
                    continue
                geo = gs.geometry()
                view(geo.transforms())[0] = np.array(tf_list, dtype=np.float64)
            for name, pos_list in snapshot.get("fem_positions", {}).items():
                gs = ss.geo_slots.get(name)
                if gs is None:
                    continue
                geo = gs.geometry()
                pos = np.array(pos_list, dtype=np.float64).reshape(-1, 3, 1)
                view(geo.positions())[:] = pos
            print(f"[snapshot] Applied snapshot (sim_frame={snapshot.get('sim_frame')})")

        def on_frame(info: Any, ctrl: URDFController) -> None:
            if _FREEZE_ROBOT:
                return
            frame = int(info.frame())
            if frame < len(schedule):
                ctrl.set_joint_positions(schedule[frame]["joints"])

        has_ipc_bodies = bool(controller.bindings)
        if has_ipc_bodies:
            controller.bind_animator(scene, on_frame=on_frame)
        else:
            print("[replay] WARNING: No IPC bodies created — FK scrubbing only.")

        world.init(scene)
        world.retrieve()

        sgui = SceneGUI(scene, "split")
        sio = SceneIO(scene)
        sgui.register()
        sgui.set_edge_width(1.0)
        app._sgui = sgui
        app._sio = sio

        controller.register_visual_only_meshes(transparency=0.35, edge_width=0.5,
                                               color=(0.2, 0.9, 0.4))
        controller.update_visual_meshes()

        stitch_net = None
        stitch_result = app.scene_state.build_stitch_line_nodes()
        if stitch_result is not None:
            sn, se = stitch_result
            stitch_net = ps.register_curve_network(
                "stitch_line", sn, se, radius=0.001, color=(1.0, 0.2, 0.2)
            )

        ctx.update({
            "engine": engine, "world": world, "scene": scene,
            "sgui": sgui, "sio": sio, "schedule": schedule,
            "sim_dt": sim_dt, "has_ipc_bodies": has_ipc_bodies,
            "stitch_net": stitch_net,
            "total_schedule_frames": len(schedule),
            "Timer": Timer, "SceneIO": SceneIO, "SceneGUI": SceneGUI,
        })
        app._sim_ctx = ctx

    # ── First build (optionally from snapshot) ─────────────────────────
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    _build_scene(snapshot=_read_snapshot())

    # ── Visual mesh overlay (persists across rebuilds) ─────────────────
    vis_controller = URDFController(urdf_path, mesh_source="visual")
    vis_controller.set_root_transform(root_tf)
    vis_controller.set_joint_positions(
        qpos_to_joint_dict(traj.robot_qpos[init_frame], genesis_joint_names)
    )

    vis_meshes: dict[str, tuple] = {}
    vis_transforms = vis_controller.get_mesh_transforms()
    for node in vis_controller.mesh_nodes:
        tf = vis_transforms.get(node.node_name)
        if tf is None:
            continue
        world_v = node.local_vertices @ tf[:3, :3].T + tf[:3, 3]
        name = f"vismesh_{node.node_name}"
        m = ps.register_surface_mesh(name, world_v, node.faces)
        m.set_transparency(0.45)
        m.set_color((0.75, 0.75, 0.75))
        vis_meshes[node.node_name] = (m, node)

    vis_visible = {"enabled": True}

    def _update_vis_meshes() -> None:
        if not vis_visible["enabled"]:
            return
        vis_controller.set_joint_positions(
            {n: controller._joint_state.get(n, 0.0) for n in controller.joint_names}
        )
        transforms = vis_controller.get_mesh_transforms()
        for nname, (mesh, node) in vis_meshes.items():
            tf = transforms.get(nname)
            if tf is None:
                continue
            mesh.update_vertex_positions(node.local_vertices @ tf[:3, :3].T + tf[:3, 3])

    coll_visible = {"enabled": True}

    print(f"[replay] Collision meshes for IPC, {len(vis_meshes)} visual overlay nodes")

    # ── Reference OBJ state ────────────────────────────────────────────
    ref_state: dict[str, Any] = {"entries": [], "input_buf": "", "sel_ref": 0}
    if _CONFIG_FILE.exists():
        try:
            cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            for item in cfg.get("ref_objs", []):
                _load_ref_obj(ref_state, item["path"], item.get("transform"))
        except (json.JSONDecodeError, KeyError):
            pass

    # ── User scene transform ───────────────────────────────────────────
    user_scene_tf: list[float] = [0.0] * 6
    if _CONFIG_FILE.exists():
        try:
            cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            saved = cfg.get("user_scene_transform")
            if saved and len(saved) == 6:
                user_scene_tf[:] = saved
        except (json.JSONDecodeError, KeyError):
            pass

    def _save_config() -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "ref_objs": [{"path": e["path"], "transform": e["transform"]}
                         for e in ref_state["entries"]],
            "user_scene_transform": list(user_scene_tf),
        }
        try:
            view_mat = ps.get_camera_view_matrix()
            data["camera_view_matrix"] = view_mat.tolist()
            data["camera_fov_vertical_deg"] = ps.get_vertical_fov_degrees()
        except Exception:
            pass
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    atexit.register(_save_config)

    # ── Playback state (FK scrubbing) ──────────────────────────────────
    play = {
        "frame": 0,
        "playing": False,
        "speed": 1.0,
        "loop": True,
        "last_time": time.monotonic(),
        "accum": 0.0,
    }

    # ── Simulation state ───────────────────────────────────────────────
    sim = {
        "active": False,
        "run": False,
        "steps_per_tick": 1,
        "dump_enabled": True,
        "export_surface": False,
        "total_frames": ctx["total_schedule_frames"],
        "recover_frame": 0,
    }

    # ── Shared helpers ─────────────────────────────────────────────────

    def _sync_view() -> None:
        if ctx["has_ipc_bodies"]:
            controller.apply_to_scene(snap=True)
        controller.update_visual_meshes()
        _update_vis_meshes()
        ctx["sgui"].update()

    def _update_stitch() -> None:
        snet = ctx.get("stitch_net")
        if snet is not None:
            result = app.scene_state.build_stitch_line_nodes()
            if result is not None:
                snet.update_node_positions(result[0])

    def _apply_traj_frame(idx: int) -> None:
        idx = max(0, min(idx, traj.n_frames - 1))
        play["frame"] = idx
        controller.set_joint_positions(
            qpos_to_joint_dict(traj.robot_qpos[idx], genesis_joint_names)
        )
        _sync_view()

    def _sim_step() -> None:
        ctx["world"].advance()
        ctx["world"].retrieve()
        if sim["dump_enabled"]:
            ctx["world"].dump()
        if sim["export_surface"]:
            ctx["sio"].write_surface(f"{output_dir}/surface_{ctx['world'].frame()}.obj")

    _apply_traj_frame(init_frame)

    # ── Scene snapshot save / load ─────────────────────────────────────

    def _save_snapshot() -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        data["joints"] = {n: v for n, v in controller._joint_state.items()}
        data["sim_frame"] = int(ctx["world"].frame())
        data["traj_frame"] = int(play["frame"])
        rigid_tfs: dict[str, list] = {}
        fem_pos: dict[str, list] = {}
        ss = app.scene_state
        for name, gs in ss.geo_slots.items():
            geo = gs.geometry()
            if name in ss.fem_objects:
                pos = np.array(view(geo.positions()), copy=True).reshape(-1, 3)
                fem_pos[name] = pos.tolist()
            else:
                tf = np.array(view(geo.transforms()), copy=True).reshape(-1, 4, 4)[0]
                rigid_tfs[name] = tf.tolist()
        data["rigid_transforms"] = rigid_tfs
        data["fem_positions"] = fem_pos
        snap_json = _CONFIG_DIR / "scene_snapshot.json"
        with open(snap_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[snapshot] Saved {len(rigid_tfs)} rigid + {len(fem_pos)} FEM objects, "
              f"joints={len(data['joints'])}, sim_frame={data['sim_frame']} -> {snap_json}")

    def _load_snapshot() -> None:
        snap = _read_snapshot()
        if snap is None:
            print("[snapshot] No snapshot file found")
            return
        print("[snapshot] Rebuilding UIPC scene from snapshot ...")
        _build_scene(snapshot=snap)
        sim["active"] = False
        sim["run"] = False
        sim["total_frames"] = ctx["total_schedule_frames"]
        play["playing"] = False
        _update_vis_meshes()
        _update_stitch()
        print("[snapshot] Scene rebuilt successfully")

    # ── Restore camera from config ─────────────────────────────────────
    if _CONFIG_FILE.exists():
        try:
            cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            cam_mat = cfg.get("camera_view_matrix")
            if cam_mat is not None:
                ps.set_camera_view_matrix(np.array(cam_mat, dtype=np.float64))
            cam_fov = cfg.get("camera_fov_vertical_deg")
            if cam_fov is not None:
                ps.set_vertical_fov_degrees(float(cam_fov))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # ── GUI panels ─────────────────────────────────────────────────────

    def _draw_playback_panel() -> None:
        if not imgui.TreeNode("Trajectory Playback"):
            return

        n = traj.n_frames
        t_val = traj.sim_times[play["frame"]] if play["frame"] < n else 0.0
        total_t = float(traj.sim_times[-1]) if n > 0 else 0.0
        imgui.Text(f"Time: {t_val:.3f}s / {total_t:.3f}s")
        imgui.Text(f"Frame: {play['frame']} / {n - 1}")

        if sim["active"]:
            imgui.TextColored((1.0, 0.8, 0.2, 1.0), "(Simulation active -- scrubbing disabled)")
            imgui.TreePop()
            return

        if imgui.Button("Play" if not play["playing"] else "Pause"):
            play["playing"] = not play["playing"]
            play["last_time"] = time.monotonic()
            play["accum"] = 0.0
        imgui.SameLine()
        if imgui.Button("Reset##traj"):
            play["playing"] = False
            _apply_traj_frame(0)

        changed, val = imgui.SliderInt("Frame##scrub", play["frame"], 0, n - 1)
        if changed:
            play["playing"] = False
            _apply_traj_frame(val)

        changed, spd = imgui.SliderFloat("Speed##traj", play["speed"], 0.1, 5.0)
        if changed:
            play["speed"] = spd

        changed, loop = imgui.Checkbox("Loop##traj", play["loop"])
        if changed:
            play["loop"] = loop

        imgui.TreePop()

    def _draw_ref_panel() -> None:
        if not imgui.TreeNode("Reference OBJ"):
            return
        c, buf = imgui.InputText("OBJ Path##ref", ref_state["input_buf"])
        if c:
            ref_state["input_buf"] = buf
        if imgui.Button("Load##ref_load"):
            p = ref_state["input_buf"].strip()
            if p:
                _load_ref_obj(ref_state, p)
                _save_config()
        imgui.SameLine()
        if imgui.Button("Clear All##ref_clear"):
            _unload_all_refs(ref_state)
            _save_config()

        entries = ref_state["entries"]
        if entries:
            labels = [f"{i}: {Path(e['path']).name}" for i, e in enumerate(entries)]
            c, sel = imgui.Combo("##ref_sel", ref_state["sel_ref"], labels)
            if c:
                ref_state["sel_ref"] = sel
            si = min(ref_state["sel_ref"], len(entries) - 1)
            entry = entries[si]
            if imgui.Button(f"Remove##{si}"):
                _unload_ref(ref_state, si)
                ref_state["sel_ref"] = max(0, min(ref_state["sel_ref"], len(entries) - 1))
                _save_config()
            else:
                t = entry["transform"]
                tf_changed = False
                for axis, idx_val in [("Tx", 0), ("Ty", 1), ("Tz", 2)]:
                    c, v = imgui.SliderFloat(f"{axis}##{si}", t[idx_val], -2.0, 2.0)
                    if c:
                        t[idx_val] = v
                        tf_changed = True
                for axis, idx_val in [("Rx", 3), ("Ry", 4), ("Rz", 5)]:
                    c, v = imgui.SliderFloat(f"{axis}##{si}", t[idx_val], -180.0, 180.0)
                    if c:
                        t[idx_val] = v
                        tf_changed = True
                if tf_changed:
                    _apply_ref_transform(entry)
                    _save_config()
                if imgui.Button(f"Reset Transform##{si}"):
                    entry["transform"] = [0.0] * 6
                    _apply_ref_transform(entry)
                    _save_config()
        imgui.TreePop()

    def _draw_scene_panel() -> None:
        if not imgui.TreeNode("User Scene"):
            return
        uobj = app.scene_state.get_objects()
        imgui.Text(f"Objects: {', '.join(sorted(uobj.keys())) if uobj else '(none)'}")

        tf_changed = False
        for label, idx, speed, lo, hi in [
            ("Scene Tx", 0, 0.001, -10.0, 10.0),
            ("Scene Ty", 1, 0.001, -10.0, 10.0),
            ("Scene Tz", 2, 0.001, -10.0, 10.0),
            ("Scene Rx", 3, 0.5, -360.0, 360.0),
            ("Scene Ry", 4, 0.5, -360.0, 360.0),
            ("Scene Rz", 5, 0.5, -360.0, 360.0),
        ]:
            c, v = imgui.DragFloat(label, user_scene_tf[idx], speed, lo, hi)
            if c:
                user_scene_tf[idx] = v
                tf_changed = True
        if imgui.Button("Reset Scene Transform"):
            user_scene_tf[:] = [0.0] * 6
            tf_changed = True
        if tf_changed:
            app.scene_state.apply_global_transform(build_scene_transform(user_scene_tf))
            ctx["sgui"].update()
            _save_config()
        imgui.TreePop()

    def _draw_sim_panel() -> None:
        if not imgui.TreeNode("Simulation"):
            return

        imgui.Text(f"UIPC dt: {ctx['sim_dt']:.6f}s  |  Traj dt: {traj.dt:.6f}s  |  Speed: {play_speed}x")

        if not sim["active"]:
            imgui.Text(f"Trajectory: {sim['total_frames']} frames ready.")
            if not ctx["has_ipc_bodies"]:
                imgui.TextColored((1.0, 0.4, 0.2, 1.0),
                                  "No IPC bodies -- sim link patterns matched nothing.")
                imgui.TreePop()
                return
            imgui.Text("Scene initialized. Click to start UIPC physics.")
            if imgui.Button("Start Simulation"):
                play["playing"] = False
                sim["active"] = True
                sim["run"] = False
                print(f"[sim] Started: {sim['total_frames']} frames")
        else:
            if imgui.Button("Play / Pause##sim"):
                sim["run"] = not sim["run"]
            imgui.SameLine()
            if imgui.Button("Step Once##sim"):
                _sim_step()
                controller.update_visual_meshes()
                ctx["sgui"].update()
                _update_stitch()

            changed, spd = imgui.SliderInt("Steps/tick##sim", sim["steps_per_tick"], 1, 8)
            if changed:
                sim["steps_per_tick"] = int(max(1, spd))

            changed, v = imgui.Checkbox("Dump state##sim", sim["dump_enabled"])
            if changed:
                sim["dump_enabled"] = v
            changed, v = imgui.Checkbox("Export surface OBJ##sim", sim["export_surface"])
            if changed:
                sim["export_surface"] = v

            cur_frame = ctx["world"].frame()
            tf = sim["total_frames"]
            imgui.Text(f"Frame: {cur_frame} / {tf}")
            if tf > 0:
                imgui.Text(f"Progress: {min(cur_frame / max(tf, 1), 1.0) * 100:.1f}%")

            if sim["run"]:
                for _ in range(sim["steps_per_tick"]):
                    _sim_step()
                    if ctx["world"].frame() >= tf and tf > 0:
                        sim["run"] = False
                        print("[sim] Playback complete")
                        break
                controller.update_visual_meshes()
                ctx["sgui"].update()
                _update_stitch()
                Timer.report()

            imgui.Separator()
            changed, val = imgui.InputInt("Target Frame##sim", sim["recover_frame"])
            if changed:
                sim["recover_frame"] = max(0, val)
            if imgui.Button("Recover##sim"):
                target = sim["recover_frame"]
                if ctx["world"].recover(target):
                    ctx["world"].retrieve()
                    controller.update_visual_meshes()
                    ctx["sgui"].update()
                    _update_stitch()
                    print(f"[sim] recovered to frame {target}")
                else:
                    print(f"[sim] recover to frame {target} failed")

            imgui.SameLine()
            if imgui.Button("Replay to##sim"):
                target = sim["recover_frame"]
                sim["replaying"] = True
                sim["replay_target"] = target
                print(f"[sim] replay from {ctx['world'].frame()} to {target} ...")

            if sim.get("replaying"):
                replay_target = sim["replay_target"]
                steps_this_tick = min(sim["steps_per_tick"], replay_target - ctx["world"].frame())
                for _ in range(max(1, steps_this_tick)):
                    next_f = ctx["world"].frame() + 1
                    if ctx["world"].recover(next_f):
                        ctx["world"].retrieve()
                        if sim["export_surface"]:
                            ctx["sio"].write_surface(f"{output_dir}/surface_{ctx['world'].frame()}.obj")
                    else:
                        _sim_step()
                    if ctx["world"].frame() >= replay_target:
                        sim["replaying"] = False
                        print(f"[sim] replay reached frame {ctx['world'].frame()}")
                        break
                controller.update_visual_meshes()
                ctx["sgui"].update()
                _update_stitch()
                imgui.Text(f"Replaying: {ctx['world'].frame()} / {replay_target}")

        imgui.TreePop()

    # ── Main callback ──────────────────────────────────────────────────

    def on_update() -> None:
        imgui.Text("=== Trajectory Replay Controller ===")
        if imgui.Button("Save Config"):
            _save_config()
            print(f"[config] Saved to {_CONFIG_FILE}")
        imgui.SameLine()
        if imgui.Button("Save Snapshot"):
            _save_snapshot()
        imgui.SameLine()
        if imgui.Button("Load Snapshot"):
            _load_snapshot()

        changed, vis = imgui.Checkbox("Show Visual Meshes", vis_visible["enabled"])
        if changed:
            vis_visible["enabled"] = vis
            for _, (mesh, _node) in vis_meshes.items():
                mesh.set_enabled(vis)
        imgui.SameLine()
        changed, cvis = imgui.Checkbox("Show Collision Meshes", coll_visible["enabled"])
        if changed:
            coll_visible["enabled"] = cvis
            for nname, (mesh, node) in controller._visual_meshes.items():
                mesh.set_enabled(cvis)

        imgui.Separator()
        _draw_playback_panel()
        imgui.Separator()
        _draw_ref_panel()
        imgui.Separator()
        _draw_scene_panel()
        imgui.Separator()
        _draw_sim_panel()

        # FK playback tick (only when sim is not active)
        if play["playing"] and not sim["active"]:
            now = time.monotonic()
            dt_wall = now - play["last_time"]
            play["last_time"] = now
            play["accum"] += dt_wall * play["speed"]

            n = traj.n_frames
            adv = int(play["accum"] / traj.dt) if traj.dt > 0 else 0
            if adv > 0:
                play["accum"] -= adv * traj.dt
                new_frame = play["frame"] + adv
                if new_frame >= n:
                    if play["loop"]:
                        new_frame = new_frame % n
                    else:
                        new_frame = n - 1
                        play["playing"] = False
                _apply_traj_frame(new_frame)

    on_update._refs = (ctx, controller, app)
    ps.set_user_callback(on_update)
    ps.show()


# ---------------------------------------------------------------------------
# Reference OBJ helpers (standalone, not tied to URDFGuiApp)
# ---------------------------------------------------------------------------


def _load_ref_obj(ref: dict, path_str: str, transform: list[float] | None = None) -> bool:
    p = Path(path_str)
    if not p.exists():
        print(f"[ref] File not found: {p}")
        return False
    try:
        verts, faces, edges = parse_obj(p)
    except Exception as e:
        print(f"[ref] Failed to parse {p}: {e}")
        return False
    if len(verts) == 0:
        return False
    idx = len(ref["entries"])
    tf = transform if transform else [0.0] * 6
    entry: dict = {"path": path_str, "transform": tf, "verts": verts, "surf": None, "curve": None}
    R = make_rotation(tf[3], tf[4], tf[5])
    offset = np.array(tf[:3])
    transformed = verts @ R.T + offset
    if faces is not None:
        sname = f"ref_surf_{p.stem}_{idx}"
        m = ps.register_surface_mesh(sname, transformed, faces)
        m.set_transparency(0.4)
        m.set_color((0.6, 0.8, 1.0))
        entry["surf"] = sname
    if edges is not None:
        cname = f"ref_line_{p.stem}_{idx}"
        ps.register_curve_network(cname, transformed, edges, color=(1.0, 0.4, 0.2), radius=0.001)
        entry["curve"] = cname
    ref["entries"].append(entry)
    return True


def _unload_ref(ref: dict, idx: int) -> None:
    e = ref["entries"][idx]
    if e["surf"]:
        ps.remove_surface_mesh(e["surf"], error_if_absent=False)
    if e["curve"]:
        ps.remove_curve_network(e["curve"])
    ref["entries"].pop(idx)


def _unload_all_refs(ref: dict) -> None:
    while ref["entries"]:
        _unload_ref(ref, 0)


def _apply_ref_transform(entry: dict) -> None:
    t = entry["transform"]
    R = make_rotation(t[3], t[4], t[5])
    offset = np.array(t[:3])
    transformed = entry["verts"] @ R.T + offset
    if entry["surf"]:
        ps.get_surface_mesh(entry["surf"]).update_vertex_positions(transformed)
    if entry["curve"]:
        ps.get_curve_network(entry["curve"]).update_node_positions(transformed)


# ---------------------------------------------------------------------------
# Export-recover (no GUI)
# ---------------------------------------------------------------------------


def run_export_recover(
    traj: TrajectoryData,
    urdf_path: str,
    genesis_joint_names: list[str],
    max_frame: int,
    frame_skip: int = 10,
    seq_dir: Path | None = None,
    sim_dt_override: float | None = None,
    init_frame: int = 0,
    play_speed: float = 1.0,
) -> None:
    """Recover UIPC frames 0..max_frame and export seq data (NPY) for Genesis replay."""
    from uipc import view

    from asset_dir import AssetDir
    from urdf_gui import SequenceExporter

    output_dir = Path(AssetDir.output_path(__file__))
    if seq_dir is None:
        seq_dir = _YOYO_DIR / "v5" / "seq"

    app = URDFGuiApp(
        urdf_path=urdf_path,
        sim_link_patterns=_SIM_LINK_PATTERNS,
        stc_strength=_STC_STRENGTH,
        load_scene_fn=load_yoyo_scene,
        setup_stitch_fn=setup_yoyo_stitch,
        record_file=_SCRIPT_DIR / "motion_keyframes.json",
        joint_file=_SCRIPT_DIR / "joint_angles.json",
        config_file=_CONFIG_FILE,
        entry_file=__file__,
    )

    controller = URDFController(urdf_path, mesh_source="collision")
    app._controller = controller
    app._joint_names = controller.joint_names
    app._limits = controller.joint_limits
    app._end_effectors = controller.find_end_effectors()

    root_tf = np.eye(4, dtype=np.float64)
    root_tf[:3, 3] = _ROBOT_BASE_POS
    controller.set_root_transform(root_tf)

    init_frame = max(0, min(init_frame, traj.n_frames - 1))
    controller.set_joint_positions(
        qpos_to_joint_dict(traj.robot_qpos[init_frame], genesis_joint_names)
    )

    engine, world, scene = app._create_engine_scene(output_dir)
    app._setup_robot(scene)
    app._apply_scene_and_stitch(scene, world)

    if sim_dt_override is not None:
        view(scene.config().find("dt"))[0] = sim_dt_override

    sim_dt = float(view(scene.config().find("dt"))[0])

    effective_traj_dt = traj.dt / max(play_speed, 1e-6)
    schedule = trajectory_to_frame_schedule(
        traj.robot_qpos[init_frame:], genesis_joint_names,
        traj_dt=effective_traj_dt, sim_dt=sim_dt,
    )
    print(f"[export-recover] sim_dt={sim_dt}, traj dt={traj.dt}, play_speed={play_speed}x, "
          f"traj frames={traj.n_frames - init_frame} (from {init_frame}) -> schedule={len(schedule)} frames")

    has_ipc_bodies = bool(controller.bindings)
    if has_ipc_bodies:
        def on_frame(info, ctrl):
            frame = int(info.frame())
            if frame < len(schedule):
                ctrl.set_joint_positions(schedule[frame]["joints"])
        controller.bind_animator(scene, on_frame=on_frame)

    urdf_rel = urdf_path

    joint_names = controller.joint_names
    exporter = SequenceExporter(
        joint_names, app.scene_state, dt=sim_dt, urdf_rel=urdf_rel,
        robot_base_pos=_ROBOT_BASE_POS.tolist(),
    )

    def _joints_at_frame(f: int) -> dict[str, float]:
        if f < len(schedule):
            return schedule[f]["joints"]
        if schedule:
            return schedule[-1]["joints"]
        return {n: 0.0 for n in joint_names}

    world.init(scene)
    world.retrieve()
    exporter.capture(world.frame(), _joints_at_frame(world.frame()))

    exported = 0
    for target in range(1, max_frame + 1):
        if not world.recover(target):
            print(f"[export-recover] no dump at frame {target}, stopping")
            break
        world.retrieve()
        if target % frame_skip == 0:
            exporter.capture(target, _joints_at_frame(target))
            exported += 1
            if exported % 100 == 0:
                print(f"[export-recover] {exported} frames captured ...")

    exporter.save(seq_dir)
    print(f"[export-recover] Done: {exported} frames (skip={frame_skip}) -> {seq_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recorded trajectory in Polyscope GUI"
    )
    parser.add_argument(
        "--traj-dir", type=str, default=str(_YOYO_DIR / "v5_init"),
        help="Directory containing trajectory.npz and meshes",
    )
    parser.add_argument(
        "--dt", type=float, default=None,
        help="Override UIPC simulation timestep (default: 0.001 from scene config)",
    )
    parser.add_argument(
        "--init-frame", type=int, default=41,
        help="Initial trajectory frame to display on startup (default: 41)",
    )
    parser.add_argument(
        "--play-speed", type=float, default=0.5,
        help="Replay speed multiplier for trajectory schedule (default: 0.5)",
    )
    parser.add_argument(
        "--export-recover", type=int, default=-1, metavar="MAX_FRAME",
        help="No-GUI: recover frames 0..MAX_FRAME, export seq NPY, then exit.",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=10,
        help="Export every N-th frame (default 10).",
    )
    parser.add_argument(
        "--seq-dir", type=str, default=None,
        help="Override sequence export directory (default: DemoAssets/yoyo/v5/seq).",
    )
    args = parser.parse_args()

    print("[replay] Downloading/locating marvin_sharpa URDF ...")
    urdf_path = get_sharpa_urdf()
    print(f"[replay] URDF: {urdf_path}")

    print(f"[replay] Loading trajectory from {args.traj_dir} ...")
    traj = load_trajectory_npz(args.traj_dir)
    print(f"[replay] {traj.n_frames} frames, {traj.robot_qpos.shape[1]} DOFs, dt={traj.dt:.6f}s")

    print("[replay] Loading Genesis joint ordering ...")
    genesis_joint_names = load_genesis_joint_order([
        traj.traj_dir, Path(urdf_path).parent, _SCRIPT_DIR,
    ])
    print(f"[replay] Genesis joint order: {len(genesis_joint_names)} DOFs")

    if args.export_recover >= 0:
        seq_dir = Path(args.seq_dir) if args.seq_dir else None
        run_export_recover(
            traj, urdf_path, genesis_joint_names,
            max_frame=args.export_recover,
            frame_skip=max(1, args.frame_skip),
            seq_dir=seq_dir,
            sim_dt_override=args.dt,
            init_frame=args.init_frame,
            play_speed=args.play_speed,
        )
    else:
        run(traj, urdf_path, genesis_joint_names,
            sim_dt_override=args.dt, init_frame=args.init_frame,
            play_speed=args.play_speed)


if __name__ == "__main__":
    main()
