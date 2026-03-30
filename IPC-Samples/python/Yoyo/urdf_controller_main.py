"""GUI entry script for inspecting and operating the URDF robot.

Modes:
  --sim       Launch UIPC physics simulation directly with SceneGUI.
  (default)   FK/IK inspector with fully initialized UIPC scene displayed
              via SceneGUI.  Simulation advancing starts when "Start
              Simulation" is clicked.

Controls (inspector mode):
  - Joint Angles panel: sliders for each actuated joint
  - End Effectors panel: live position + orientation readout
  - IK Control panel: dual-gripper keyboard control
      Left  gripper: W/S = X,  A/D = Y,  Q/E = Z
      Right gripper: I/K = X,  J/L = Y,  U/O = Z

Simulation mode:
  - Loads motion_keyframes.json and replays via SoftTransformConstraint.
  - Provides user_load_scene() callback for custom scene objects.
  - Supports recover from saved simulation state.
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

from urdf_controller import URDFController
from load_user_scene import (
    apply_global_transform,
    build_stitch_line_nodes,
    export_frame_npy,
    get_user_objects,
    stitch_string_to_gripper,
    user_load_scene,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_FILE = _SCRIPT_DIR / "joint_angles.json"
_RECORD_FILE = _SCRIPT_DIR / "motion_keyframes.json"
_CONFIG_FILE = _SCRIPT_DIR / "urdf_gui_config.json"

# ---- Orientation presets (Z-up world) ----
# Each preset is a 3x3 rotation matrix for the end-effector frame.
_ORIENTATION_PRESETS: dict[str, np.ndarray] = {
    "None (pos only)": np.zeros((3, 3)),  # sentinel: skip orientation
    # --- approach -Z (down), two roll variants ---
    "Horiz -Z": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    "Horiz -Z inv": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64),
    # --- approach +X, two roll variants ---
    "Horiz +X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    "Horiz +X inv": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64),
    # --- approach -X, two roll variants ---
    "Horiz -X": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    "Horiz -X inv": np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], dtype=np.float64),
    # --- approach +Y, two roll variants ---
    "Horiz +Y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
    "Horiz +Y inv": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64),
    # --- approach -Y, two roll variants ---
    "Horiz -Y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    "Horiz -Y inv": np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64),
    # --- approach +Z (up) ---
    "Vert +Z": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64),
    "Vert +Z inv": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
}
_PRESET_NAMES = list(_ORIENTATION_PRESETS.keys())


def _rot_to_display(R: np.ndarray) -> str:
    """Format a 3x3 rotation as local X/Y/Z axis directions."""
    return (
        f"  X: ({R[0,0]:+.3f}, {R[1,0]:+.3f}, {R[2,0]:+.3f})\n"
        f"  Y: ({R[0,1]:+.3f}, {R[1,1]:+.3f}, {R[2,1]:+.3f})\n"
        f"  Z: ({R[0,2]:+.3f}, {R[1,2]:+.3f}, {R[2,2]:+.3f})"
    )


def _try_key_pressed(key_char: str) -> bool:
    """Best-effort keyboard detection across imgui versions."""
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


def _build_scene_transform(t: list[float]) -> np.ndarray:
    """Build a 4x4 matrix from [tx, ty, tz, rx_deg, ry_deg, rz_deg]."""
    tx, ty, tz, rx, ry, rz = t
    cx, sx = np.cos(np.radians(rx)), np.sin(np.radians(rx))
    cy, sy = np.cos(np.radians(ry)), np.sin(np.radians(ry))
    cz, sz = np.cos(np.radians(rz)), np.sin(np.radians(rz))
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R = Rz @ Ry @ Rx
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = [tx, ty, tz]
    return mat


def main() -> None:
    from asset_dir import AssetDir
    from uipc import Logger, SceneIO, Timer, view
    from uipc.core import Engine, Scene, World
    from uipc.gui import SceneGUI
    from uipc.unit import GPa

    repo_root = Path(__file__).resolve().parents[3]
    urdf_path = repo_root / "DemoAssets" / "marvin_bimanual" / "urdf" / "marvin_pika.urdf"
    output_dir = Path(AssetDir.output_path(__file__))

    print(f"[urdf-gui] Loading URDF: {urdf_path}")
    controller = URDFController(urdf_path, mesh_source="visual")

    joint_names = controller.joint_names
    limits = controller.joint_limits
    end_effectors = controller.find_end_effectors()

    print(f"[urdf-gui] Actuated joints ({len(joint_names)}): {joint_names}")
    print(f"[urdf-gui] End effectors: {end_effectors}")

    # ---- Mutable state ----
    state = {
        "joints": {n: 0.0 for n in joint_names},
        "ee_idx": 0,
        "ik_step": 0.01,
        "left_orient_idx": 0,
        "right_orient_idx": 0,
    }

    # ---- Motion recording / playback state ----
    rec = {
        "keyframes": [],
        "interp_time": 1.0,
        "recording": False,
        "playing": False,
        "play_idx": 0,
        "play_t0": 0.0,
        "play_start_joints": {},
        "sel_kf": 0,
    }

    # ---- Auto-load joints & keyframes (before UIPC scene setup) ----
    if _JOINT_FILE.exists():
        try:
            loaded = json.loads(_JOINT_FILE.read_text(encoding="utf-8"))
            joints_data = loaded.get("joints", loaded) if isinstance(loaded, dict) else loaded
            for n in joint_names:
                if n in joints_data:
                    state["joints"][n] = joints_data[n]
            if isinstance(loaded, dict):
                lo = loaded.get("left_orient", "")
                ro = loaded.get("right_orient", "")
                if lo in _PRESET_NAMES:
                    state["left_orient_idx"] = _PRESET_NAMES.index(lo)
                if ro in _PRESET_NAMES:
                    state["right_orient_idx"] = _PRESET_NAMES.index(ro)
            print(f"[urdf-gui] Auto-loaded joint angles from {_JOINT_FILE}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[urdf-gui] Failed to auto-load joints: {e}")

    if _RECORD_FILE.exists():
        try:
            loaded_kfs = json.loads(_RECORD_FILE.read_text(encoding="utf-8"))
            rec["keyframes"].clear()
            for entry in loaded_kfs:
                rec["keyframes"].append({
                    "joints": entry["joints"],
                    "dt": entry.get("dt", 1.0),
                    "left_orient": entry.get("left_orient", ""),
                    "right_orient": entry.get("right_orient", ""),
                })
            if rec["keyframes"]:
                kf0 = rec["keyframes"][0]
                for n in joint_names:
                    if n in kf0["joints"]:
                        state["joints"][n] = kf0["joints"][n]
                lo = kf0.get("left_orient", "")
                ro = kf0.get("right_orient", "")
                if lo in _PRESET_NAMES:
                    state["left_orient_idx"] = _PRESET_NAMES.index(lo)
                if ro in _PRESET_NAMES:
                    state["right_orient_idx"] = _PRESET_NAMES.index(ro)
            print(f"[urdf-gui] Auto-loaded {len(rec['keyframes'])} keyframes from {_RECORD_FILE}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[urdf-gui] Failed to auto-load keyframes: {e}")

    controller.set_joint_positions(state["joints"])

    # ---- Build UIPC scene and initialize ----
    Logger.set_level(Logger.Level.Info)
    Timer.enable_all()
    engine = Engine("cuda", str(output_dir))
    world = World(engine)

    config = Scene.default_config()
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["linear_system"]["tol_rate"] = 1e-4
    config["newton"]["max_iter"] = 256
    scene = Scene(config)
    scene.animator().substep(1)

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0 * GPa)

    _SIM_LINK_PATTERNS = [
        "*Link8*",
        "*Link9*",
    ]
    controller.create_ipc_bodies(
        scene,
        object_prefix="robot_link",
        stc_strength=np.array([120.0, 120.0], dtype=np.float64),
        include_patterns=_SIM_LINK_PATTERNS,
    )
    controller.configure_contact(scene, enable_self=False, enable_default=True)
    controller.apply_to_scene(snap=True)

    # ---- Load user scene transform from config (needed before world.init) ----
    user_scene_tf: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if _CONFIG_FILE.exists():
        try:
            _pre_cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            _saved = _pre_cfg.get("user_scene_transform")
            if _saved and len(_saved) == 6:
                user_scene_tf[:] = _saved
        except (json.JSONDecodeError, KeyError):
            pass

    user_load_scene(scene, world)
    apply_global_transform(_build_scene_transform(user_scene_tf))

    binding_names = [(b.node_name, b.parent_link, b.object_name) for b in controller.bindings]
    print(f"[stitch-debug] Available bindings: {binding_names}")
    right_gripper_binding = controller.find_binding("*Link9_R*")
    if right_gripper_binding is not None:
        print(f"[stitch-debug] Matched binding: {right_gripper_binding.node_name}")
        stitch_string_to_gripper(
            scene,
            right_gripper_binding.geo_slot,
            right_gripper_binding.rest_geo_slot,
        )
    else:
        print("[stitch-debug] No binding matched '*Link9_R*'")

    sim_dt = float(view(scene.config().find("dt"))[0])
    print(f"[urdf-gui] sim_dt = {sim_dt} (set by user_load_scene)")

    keyframes = _load_keyframes()
    schedule = _build_frame_schedule(keyframes, sim_dt, 1)

    _sim_ctx: dict[str, Any] = {
        "engine": engine, "world": world, "scene": scene,
        "schedule": schedule, "output_dir": output_dir,
        "controller": controller, "sim_dt": sim_dt,
        "Timer": Timer, "SceneIO": SceneIO, "SceneGUI": SceneGUI,
    }

    def on_frame(info: Any, ctrl: URDFController) -> None:
        s = _sim_ctx["schedule"]
        frame = int(info.frame())
        if frame < len(s):
            ctrl.set_joint_positions(s[frame]["joints"])

    controller.bind_animator(scene, on_frame=on_frame)

    # Init world so SceneGUI can display the full scene
    world.init(scene)
    world.retrieve()

    # ---- Polyscope init ----
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    # SceneGUI shows the full simulation scene (robot IPC bodies + user objects)
    sgui = SceneGUI(scene, "split")
    sio = SceneIO(scene)
    sgui.register()
    sgui.set_edge_width(1.0)
    _sim_ctx["sgui"] = sgui
    _sim_ctx["sio"] = sio

    # Non-simulated robot links as transparent visual overlays
    controller.register_visual_only_meshes(transparency=0.45)
    controller.update_visual_meshes()

    # Stitch line visualisation
    _stitch_net = None
    _stitch_result = build_stitch_line_nodes()
    if _stitch_result is not None:
        _stitch_nodes, _stitch_edges = _stitch_result
        _stitch_net = ps.register_curve_network(
            "stitch_line", _stitch_nodes, _stitch_edges, radius=0.001, color=(1.0, 0.2, 0.2)
        )

    # Identify left / right grippers
    left_ee = next((e for e in end_effectors if "_L" in e or "left" in e.lower()), end_effectors[0])
    right_ee = next((e for e in end_effectors if "_R" in e or "right" in e.lower()), end_effectors[-1])
    print(f"[urdf-gui] Left gripper: {left_ee}  |  Right gripper: {right_ee}")

    # ---- Reference OBJ state ----
    # Each entry: {"path": str, "transform": [tx,ty,tz, rx,ry,rz], "surf": name|None, "curve": name|None, "verts": ndarray}
    ref: dict = {
        "entries": [],
        "input_buf": "",
        "sel_ref": 0,
    }

    def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Parse OBJ, returning (vertices, faces_or_None, edges_or_None)."""
        verts, faces, edges = [], [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v" and len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "f":
                    idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    if len(idx) >= 3:
                        for i in range(1, len(idx) - 1):
                            faces.append([idx[0], idx[i], idx[i + 1]])
                elif parts[0] == "l":
                    idx = [int(p) - 1 for p in parts[1:]]
                    for i in range(len(idx) - 1):
                        edges.append([idx[i], idx[i + 1]])
        v = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))
        fa = np.array(faces, dtype=np.int32) if faces else None
        ed = np.array(edges, dtype=np.int32) if edges else None
        return v, fa, ed

    def _make_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
        """Build 3x3 rotation from Euler angles (degrees), XYZ order."""
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _apply_ref_transform(entry: dict) -> None:
        """Update polyscope geometry with current transform."""
        t = entry["transform"]
        R = _make_rotation(t[3], t[4], t[5])
        offset = np.array(t[:3])
        transformed = entry["verts"] @ R.T + offset
        if entry["surf"]:
            ps.get_surface_mesh(entry["surf"]).update_vertex_positions(transformed)
        if entry["curve"]:
            ps.get_curve_network(entry["curve"]).update_node_positions(transformed)

    def _load_ref_obj(path_str: str, transform: list[float] | None = None) -> bool:
        """Parse and register an OBJ. Returns True on success."""
        p = Path(path_str)
        if not p.exists():
            print(f"[ref] File not found: {p}")
            return False
        try:
            verts, faces, edges = _parse_obj(p)
        except Exception as e:
            print(f"[ref] Failed to parse {p}: {e}")
            return False
        if len(verts) == 0:
            print(f"[ref] No vertices in {p}")
            return False

        idx = len(ref["entries"])
        tf = transform if transform else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        entry: dict = {"path": path_str, "transform": tf, "verts": verts, "surf": None, "curve": None}

        R = _make_rotation(tf[3], tf[4], tf[5])
        offset = np.array(tf[:3])
        transformed = verts @ R.T + offset

        if faces is not None:
            sname = f"ref_surf_{p.stem}_{idx}"
            m = ps.register_surface_mesh(sname, transformed, faces)
            m.set_transparency(0.4)
            m.set_color((0.6, 0.8, 1.0))
            entry["surf"] = sname
            print(f"[ref] Faces -> '{sname}' ({len(faces)} tris)")

        if edges is not None:
            cname = f"ref_line_{p.stem}_{idx}"
            ps.register_curve_network(cname, transformed, edges, color=(1.0, 0.4, 0.2), radius=0.001)
            entry["curve"] = cname
            print(f"[ref] Lines -> '{cname}' ({len(edges)} edges)")

        ref["entries"].append(entry)
        return True

    def _unload_ref(idx: int) -> None:
        e = ref["entries"][idx]
        if e["surf"]:
            ps.remove_surface_mesh(e["surf"], error_if_absent=False)
        if e["curve"]:
            ps.remove_curve_network(e["curve"])
        ref["entries"].pop(idx)

    def _unload_all_refs() -> None:
        while ref["entries"]:
            _unload_ref(0)

    def _save_config() -> None:
        data = {
            "ref_objs": [
                {"path": e["path"], "transform": e["transform"]}
                for e in ref["entries"]
            ],
            "user_scene_transform": list(user_scene_tf),
        }
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Load config from disk (restore previous ref OBJs)
    if _CONFIG_FILE.exists():
        try:
            cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            for item in cfg.get("ref_objs", []):
                _load_ref_obj(item["path"], item.get("transform"))
            for p in cfg.get("ref_obj_paths", []):
                if not any(e["path"] == p for e in ref["entries"]):
                    _load_ref_obj(p)
        except (json.JSONDecodeError, KeyError):
            pass

    atexit.register(_save_config)

    def sync_view() -> None:
        """Push current FK to SceneGUI and visual-only meshes."""
        controller.apply_to_scene(snap=True)
        controller.update_visual_meshes()
        sgui.update()

    sync_view()

    def _get_orientation(orient_idx: int) -> np.ndarray | None:
        """Return target 3x3 rotation or None if position-only."""
        name = _PRESET_NAMES[orient_idx]
        R = _ORIENTATION_PRESETS[name]
        if np.allclose(R, 0):
            return None
        return R

    def _do_ik(ee_name: str, delta: np.ndarray, orient_idx: int) -> None:
        """Run IK for one end-effector with a position delta."""
        cur_tf = controller.get_link_transform(ee_name)
        cur_pos = cur_tf[:3, 3].copy()
        arm_joints = controller.find_arm_joints(ee_name)
        if len(arm_joints) == 0:
            return
        target_orient = _get_orientation(orient_idx)
        orient_mode = "all" if target_orient is not None else None
        controller.solve_ik(
            ee_name,
            cur_pos + delta,
            target_orientation=target_orient,
            orientation_mode=orient_mode,
            arm_joints=arm_joints,
        )
        state["joints"] = controller.get_joint_positions()
        sync_view()

    def _auto_save() -> None:
        """Persist keyframes to disk after every mutation."""
        data = [
            {
                "joints": kf["joints"],
                "dt": kf["dt"],
                "left_orient": kf.get("left_orient", ""),
                "right_orient": kf.get("right_orient", ""),
            }
            for kf in rec["keyframes"]
        ]
        with open(_RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    _sim_live = {
        "active": False,
        "run": False,
        "steps_per_tick": 1,
        "dump_enabled": True,
        "export_surface": True,
        "export_npy": True,
        "recover_frame": 0,
        "total_frames": 0,
    }

    _seq_dir = _SCRIPT_DIR / "results" / "v3" / "seq"

    def _do_sim_step(_sio_ref) -> None:
        world.advance()
        world.retrieve()
        if _sim_live["dump_enabled"]:
            world.dump()
        if _sim_live["export_surface"]:
            _sio_ref.write_surface(f"{output_dir}/surface_{world.frame()}.obj")
        if _sim_live.get("export_npy", True):
            f = world.frame()
            export_frame_npy(_seq_dir, f)
            joints_dir = _seq_dir / "joints"
            joints_dir.mkdir(parents=True, exist_ok=True)
            joint_vals = {n: controller._joint_state.get(n, 0.0) for n in controller.joint_names}
            np.save(str(joints_dir / f"{f}.npy"), joint_vals)

    def on_update() -> None:
        imgui.Text("=== URDF Controller Inspector ===")
        imgui.Separator()

        # ---- Joint Angles ----
        if imgui.TreeNode("Joint Angles"):
            changed = False
            for name in joint_names:
                lo, hi = limits[name]
                c, v = imgui.SliderFloat(name, state["joints"][name], lo, hi)
                if c:
                    state["joints"][name] = v
                    changed = True
            if changed:
                controller.set_joint_positions(state["joints"])
                sync_view()

            if imgui.Button("Reset All Joints"):
                state["joints"] = {n: 0.0 for n in joint_names}
                controller.set_joint_positions(state["joints"])
                sync_view()

            imgui.SameLine()
            if imgui.Button("Save Joints"):
                save_path = _JOINT_FILE
                data = {
                    "joints": state["joints"],
                    "left_orient": _PRESET_NAMES[state["left_orient_idx"]],
                    "right_orient": _PRESET_NAMES[state["right_orient_idx"]],
                }
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"[urdf-gui] Joint angles saved to {save_path}")

            imgui.SameLine()
            if imgui.Button("Load Joints"):
                load_path = _JOINT_FILE
                if load_path.exists():
                    with open(load_path, encoding="utf-8") as f:
                        loaded = json.load(f)
                    joints_data = loaded.get("joints", loaded) if isinstance(loaded, dict) else loaded
                    for n in joint_names:
                        if n in joints_data:
                            state["joints"][n] = joints_data[n]
                    controller.set_joint_positions(state["joints"])
                    sync_view()
                    if isinstance(loaded, dict):
                        lo = loaded.get("left_orient", "")
                        ro = loaded.get("right_orient", "")
                        if lo in _PRESET_NAMES:
                            state["left_orient_idx"] = _PRESET_NAMES.index(lo)
                        if ro in _PRESET_NAMES:
                            state["right_orient_idx"] = _PRESET_NAMES.index(ro)
                    print(f"[urdf-gui] Joint angles loaded from {load_path}")
                else:
                    print(f"[urdf-gui] File not found: {load_path}")
            imgui.TreePop()

        imgui.Separator()

        # ---- End Effectors ----
        if imgui.TreeNode("End Effectors"):
            for ee in end_effectors:
                tf = controller.get_link_transform(ee)
                pos = tf[:3, 3]
                rot = tf[:3, :3]
                imgui.Text(f"{ee}:")
                imgui.Text(f"  Pos: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                imgui.Text(_rot_to_display(rot))
            imgui.TreePop()

        imgui.Separator()

        # ---- IK Control ----
        if imgui.TreeNode("IK Control"):
            c, step = imgui.SliderFloat("IK Step (m)", state["ik_step"], 0.001, 0.05)
            if c:
                state["ik_step"] = step
            s = state["ik_step"]

            imgui.Separator()

            # ---- Left Gripper (WASDQE) ----
            if imgui.TreeNode(f"Left: {left_ee}  [WASDQE]"):
                left_tf = controller.get_link_transform(left_ee)
                left_pos = left_tf[:3, 3]
                imgui.Text(f"Pos: ({left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f})")
                left_arm = controller.find_arm_joints(left_ee)
                imgui.Text(f"IK joints ({len(left_arm)}): {', '.join(left_arm)}")

                imgui.Text("Orientation:")
                c, oidx = imgui.Combo("##orient_L", state["left_orient_idx"], _PRESET_NAMES)
                if c:
                    state["left_orient_idx"] = oidx
                if imgui.Button("Lock current##lock_L"):
                    R = left_tf[:3, :3].copy()
                    _ORIENTATION_PRESETS["Locked L"] = R
                    if "Locked L" not in _PRESET_NAMES:
                        _PRESET_NAMES.append("Locked L")
                    state["left_orient_idx"] = _PRESET_NAMES.index("Locked L")

                if imgui.Button("+X##Lik"):
                    _do_ik(left_ee, np.array([s, 0, 0]), state["left_orient_idx"])
                imgui.SameLine()
                if imgui.Button("-X##Lik"):
                    _do_ik(left_ee, np.array([-s, 0, 0]), state["left_orient_idx"])
                imgui.SameLine()
                if imgui.Button("+Y##Lik"):
                    _do_ik(left_ee, np.array([0, s, 0]), state["left_orient_idx"])
                imgui.SameLine()
                if imgui.Button("-Y##Lik"):
                    _do_ik(left_ee, np.array([0, -s, 0]), state["left_orient_idx"])
                imgui.SameLine()
                if imgui.Button("+Z##Lik"):
                    _do_ik(left_ee, np.array([0, 0, s]), state["left_orient_idx"])
                imgui.SameLine()
                if imgui.Button("-Z##Lik"):
                    _do_ik(left_ee, np.array([0, 0, -s]), state["left_orient_idx"])
                imgui.TreePop()

            imgui.Separator()

            # ---- Right Gripper (IJKLUO) ----
            if imgui.TreeNode(f"Right: {right_ee}  [IJKLUO]"):
                right_tf = controller.get_link_transform(right_ee)
                right_pos = right_tf[:3, 3]
                imgui.Text(f"Pos: ({right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f})")
                right_arm = controller.find_arm_joints(right_ee)
                imgui.Text(f"IK joints ({len(right_arm)}): {', '.join(right_arm)}")

                imgui.Text("Orientation:")
                c, oidx = imgui.Combo("##orient_R", state["right_orient_idx"], _PRESET_NAMES)
                if c:
                    state["right_orient_idx"] = oidx
                if imgui.Button("Lock current##lock_R"):
                    R = right_tf[:3, :3].copy()
                    _ORIENTATION_PRESETS["Locked R"] = R
                    if "Locked R" not in _PRESET_NAMES:
                        _PRESET_NAMES.append("Locked R")
                    state["right_orient_idx"] = _PRESET_NAMES.index("Locked R")

                if imgui.Button("+X##Rik"):
                    _do_ik(right_ee, np.array([s, 0, 0]), state["right_orient_idx"])
                imgui.SameLine()
                if imgui.Button("-X##Rik"):
                    _do_ik(right_ee, np.array([-s, 0, 0]), state["right_orient_idx"])
                imgui.SameLine()
                if imgui.Button("+Y##Rik"):
                    _do_ik(right_ee, np.array([0, s, 0]), state["right_orient_idx"])
                imgui.SameLine()
                if imgui.Button("-Y##Rik"):
                    _do_ik(right_ee, np.array([0, -s, 0]), state["right_orient_idx"])
                imgui.SameLine()
                if imgui.Button("+Z##Rik"):
                    _do_ik(right_ee, np.array([0, 0, s]), state["right_orient_idx"])
                imgui.SameLine()
                if imgui.Button("-Z##Rik"):
                    _do_ik(right_ee, np.array([0, 0, -s]), state["right_orient_idx"])
                imgui.TreePop()

            imgui.Separator()

            # ---- Keyboard polling (always active) ----
            left_delta = np.zeros(3, dtype=np.float64)
            if _try_key_pressed("W"):
                left_delta[0] += s
            if _try_key_pressed("S"):
                left_delta[0] -= s
            if _try_key_pressed("A"):
                left_delta[1] += s
            if _try_key_pressed("D"):
                left_delta[1] -= s
            if _try_key_pressed("E"):
                left_delta[2] += s
            if _try_key_pressed("Q"):
                left_delta[2] -= s
            if np.any(left_delta != 0):
                _do_ik(left_ee, left_delta, state["left_orient_idx"])

            right_delta = np.zeros(3, dtype=np.float64)
            if _try_key_pressed("I"):
                right_delta[0] += s
            if _try_key_pressed("K"):
                right_delta[0] -= s
            if _try_key_pressed("J"):
                right_delta[1] += s
            if _try_key_pressed("L"):
                right_delta[1] -= s
            if _try_key_pressed("O"):
                right_delta[2] += s
            if _try_key_pressed("U"):
                right_delta[2] -= s
            if np.any(right_delta != 0):
                _do_ik(right_ee, right_delta, state["right_orient_idx"])

            imgui.Text("Left  keys: W/S=X  A/D=Y  Q/E=Z")
            imgui.Text("Right keys: I/K=X  J/L=Y  U/O=Z")
            imgui.TreePop()

        imgui.Separator()

        # ---- Motion Recording / Playback ----
        if imgui.TreeNode("Motion Recording"):
            kfs = rec["keyframes"]

            # -- Record controls --
            c, dt = imgui.SliderFloat("Interp dt (s)", rec["interp_time"], 0.1, 5.0)
            if c:
                rec["interp_time"] = dt

            enter_pressed = False
            try:
                enter_pressed = bool(imgui.IsKeyPressed(imgui.ImGuiKey_Enter))
            except (TypeError, AttributeError):
                pass

            if imgui.Button("Record Keyframe [Enter]") or enter_pressed:
                kf = {
                    "joints": dict(state["joints"]),
                    "dt": rec["interp_time"],
                    "left_orient": _PRESET_NAMES[state["left_orient_idx"]],
                    "right_orient": _PRESET_NAMES[state["right_orient_idx"]],
                }
                insert_pos = rec["sel_kf"] + 1 if kfs else 0
                kfs.insert(insert_pos, kf)
                rec["sel_kf"] = insert_pos
                _auto_save()
                print(f"[record] Keyframe inserted at #{insert_pos} (dt={kf['dt']:.2f}s), total={len(kfs)}")

            imgui.SameLine()
            if imgui.Button("Undo Last") and len(kfs) > 0:
                kfs.pop()
                _auto_save()
                print(f"[record] Last keyframe removed, {len(kfs)} remaining")

            imgui.Text(f"Keyframes: {len(kfs)}")

            # -- Keyframe list --
            if len(kfs) > 0:
                labels = [
                    f"KF {i}  dt={kf['dt']:.2f}s  L:{kf.get('left_orient', '-')}  R:{kf.get('right_orient', '-')}"
                    for i, kf in enumerate(kfs)
                ]
                c, sel = imgui.Combo("##kf_list", rec["sel_kf"], labels)
                if c:
                    rec["sel_kf"] = sel

                if imgui.Button("Go to##kf_goto"):
                    idx = min(rec["sel_kf"], len(kfs) - 1)
                    state["joints"] = dict(kfs[idx]["joints"])
                    controller.set_joint_positions(state["joints"])
                    sync_view()
                    lo = kfs[idx].get("left_orient", "")
                    ro = kfs[idx].get("right_orient", "")
                    if lo in _PRESET_NAMES:
                        state["left_orient_idx"] = _PRESET_NAMES.index(lo)
                    if ro in _PRESET_NAMES:
                        state["right_orient_idx"] = _PRESET_NAMES.index(ro)
                imgui.SameLine()
                if imgui.Button("Delete##kf_del"):
                    idx = min(rec["sel_kf"], len(kfs) - 1)
                    kfs.pop(idx)
                    rec["sel_kf"] = max(0, min(rec["sel_kf"], len(kfs) - 1))
                    _auto_save()
                    print(f"[record] Keyframe deleted, {len(kfs)} remaining")
                imgui.SameLine()
                if imgui.Button("Update##kf_upd"):
                    idx = min(rec["sel_kf"], len(kfs) - 1)
                    kfs[idx]["joints"] = dict(state["joints"])
                    kfs[idx]["dt"] = rec["interp_time"]
                    kfs[idx]["left_orient"] = _PRESET_NAMES[state["left_orient_idx"]]
                    kfs[idx]["right_orient"] = _PRESET_NAMES[state["right_orient_idx"]]
                    _auto_save()
                    print(f"[record] Keyframe {idx} updated")

                # Edit dt of selected keyframe
                idx = min(rec["sel_kf"], len(kfs) - 1)
                c, new_dt = imgui.SliderFloat(f"dt (KF {idx})", kfs[idx]["dt"], 0.1, 5.0)
                if c:
                    kfs[idx]["dt"] = new_dt
                    _auto_save()

            imgui.Separator()

            # -- Save / Load / Clear --
            if imgui.Button("Save Motion"):
                data = [
                    {
                        "joints": kf["joints"],
                        "dt": kf["dt"],
                        "left_orient": kf.get("left_orient", ""),
                        "right_orient": kf.get("right_orient", ""),
                    }
                    for kf in kfs
                ]
                with open(_RECORD_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"[record] Saved {len(kfs)} keyframes to {_RECORD_FILE}")

            imgui.SameLine()
            if imgui.Button("Load Motion"):
                if _RECORD_FILE.exists():
                    with open(_RECORD_FILE, encoding="utf-8") as f:
                        loaded = json.load(f)
                    kfs.clear()
                    for entry in loaded:
                        kfs.append({
                            "joints": entry["joints"],
                            "dt": entry.get("dt", 1.0),
                            "left_orient": entry.get("left_orient", ""),
                            "right_orient": entry.get("right_orient", ""),
                        })
                    rec["sel_kf"] = 0
                    print(f"[record] Loaded {len(kfs)} keyframes from {_RECORD_FILE}")
                else:
                    print(f"[record] File not found: {_RECORD_FILE}")

            imgui.SameLine()
            if imgui.Button("Clear All"):
                kfs.clear()
                rec["sel_kf"] = 0
                _auto_save()
                print("[record] All keyframes cleared")

            imgui.Separator()

            # -- Timeline scrub --
            if len(kfs) > 0:
                total_time = sum(kf["dt"] for kf in kfs)
                if "timeline_t" not in rec:
                    rec["timeline_t"] = 0.0
                    rec["timeline_dragging"] = False
                changed, t_val = imgui.SliderFloat(
                    "Timeline", rec["timeline_t"], 0.0, total_time, f"%.2f / {total_time:.2f}s"
                )
                if changed:
                    rec["timeline_t"] = t_val
                    rec["timeline_dragging"] = True
                    acc = 0.0
                    seg = 0
                    for i, kf in enumerate(kfs):
                        if acc + kf["dt"] >= t_val:
                            seg = i
                            break
                        acc += kf["dt"]
                        seg = i
                    local_t = t_val - acc
                    duration = kfs[seg]["dt"]
                    alpha = min(local_t / max(duration, 1e-6), 1.0)
                    src = kfs[seg - 1]["joints"] if seg > 0 else kfs[0]["joints"]
                    dst = kfs[seg]["joints"]
                    for n in joint_names:
                        v0 = src.get(n, 0.0)
                        v1 = dst.get(n, 0.0)
                        state["joints"][n] = v0 + (v1 - v0) * alpha
                    controller.set_joint_positions(state["joints"])
                    sync_view()
                    rec["sel_kf"] = seg
                    lo = kfs[seg].get("left_orient", "")
                    ro = kfs[seg].get("right_orient", "")
                    if lo in _PRESET_NAMES:
                        state["left_orient_idx"] = _PRESET_NAMES.index(lo)
                    if ro in _PRESET_NAMES:
                        state["right_orient_idx"] = _PRESET_NAMES.index(ro)

            imgui.Separator()

            # -- Step / Playback --
            if len(kfs) > 0 and not rec["playing"] and not rec.get("stepping"):
                next_idx = rec["sel_kf"]
                if next_idx < len(kfs) and imgui.Button("Move to Next"):
                    rec["stepping"] = True
                    rec["step_target"] = next_idx
                    rec["play_t0"] = time.monotonic()
                    rec["play_start_joints"] = dict(state["joints"])
                    print(f"[step] Animating to keyframe {next_idx}")

            if not rec["playing"] and not rec.get("stepping"):
                can_play = len(kfs) > 0
                if can_play and imgui.Button("Play"):
                    rec["playing"] = True
                    rec["play_idx"] = 0
                    rec["play_t0"] = time.monotonic()
                    rec["play_start_joints"] = dict(state["joints"])
                    print("[play] Playback started")
            elif rec.get("stepping"):
                if imgui.Button("Stop##step"):
                    rec["stepping"] = False
                    print("[step] Stopped")

                target_kf = kfs[rec["step_target"]]
                now = time.monotonic()
                elapsed = now - rec["play_t0"]
                duration = target_kf["dt"]
                alpha = min(elapsed / max(duration, 1e-6), 1.0)

                src = rec["play_start_joints"]
                dst = target_kf["joints"]
                for n in joint_names:
                    v0 = src.get(n, 0.0)
                    v1 = dst.get(n, 0.0)
                    state["joints"][n] = v0 + (v1 - v0) * alpha
                controller.set_joint_positions(state["joints"])
                sync_view()

                imgui.Text(f"Step -> KF {rec['step_target']}  {alpha * 100:.0f}%")

                if alpha >= 1.0:
                    lo = target_kf.get("left_orient", "")
                    ro = target_kf.get("right_orient", "")
                    if lo in _PRESET_NAMES:
                        state["left_orient_idx"] = _PRESET_NAMES.index(lo)
                    if ro in _PRESET_NAMES:
                        state["right_orient_idx"] = _PRESET_NAMES.index(ro)
                    rec["sel_kf"] = min(rec["step_target"] + 1, len(kfs) - 1)
                    rec["stepping"] = False
                    print(f"[step] Arrived at keyframe {rec['step_target']}")

            else:
                if imgui.Button("Stop"):
                    rec["playing"] = False
                    print("[play] Playback stopped")

                # Playback tick: linear interpolation between keyframes
                now = time.monotonic()
                target_idx = rec["play_idx"]
                if target_idx < len(kfs):
                    target_kf = kfs[target_idx]
                    elapsed = now - rec["play_t0"]
                    duration = target_kf["dt"]
                    alpha = min(elapsed / max(duration, 1e-6), 1.0)

                    src = rec["play_start_joints"]
                    dst = target_kf["joints"]
                    for n in joint_names:
                        v0 = src.get(n, 0.0)
                        v1 = dst.get(n, 0.0)
                        state["joints"][n] = v0 + (v1 - v0) * alpha
                    controller.set_joint_positions(state["joints"])
                    sync_view()

                    if "timeline_t" in rec:
                        t_acc = sum(kfs[j]["dt"] for j in range(target_idx))
                        rec["timeline_t"] = t_acc + alpha * duration

                    if alpha >= 1.0:
                        rec["play_start_joints"] = dict(target_kf["joints"])
                        lo = target_kf.get("left_orient", "")
                        ro = target_kf.get("right_orient", "")
                        if lo in _PRESET_NAMES:
                            state["left_orient_idx"] = _PRESET_NAMES.index(lo)
                        if ro in _PRESET_NAMES:
                            state["right_orient_idx"] = _PRESET_NAMES.index(ro)
                        rec["play_idx"] = target_idx + 1
                        rec["play_t0"] = now
                        if rec["play_idx"] >= len(kfs):
                            rec["playing"] = False
                            print("[play] Playback finished")

                imgui.Text(f"Playing: {rec['play_idx']}/{len(kfs)}")

            imgui.TreePop()

        imgui.Separator()

        # ---- Reference OBJ ----
        if imgui.TreeNode("Reference OBJ"):
            c, buf = imgui.InputText("OBJ Path", ref["input_buf"])
            if c:
                ref["input_buf"] = buf

            if imgui.Button("Load##ref_load"):
                p = ref["input_buf"].strip()
                if p and _load_ref_obj(p):
                    _save_config()

            imgui.SameLine()
            if imgui.Button("Clear All##ref_clear"):
                _unload_all_refs()
                _save_config()
                print("[ref] All reference meshes removed")

            entries = ref["entries"]
            if len(entries) > 0:
                labels = [f"{i}: {Path(e['path']).name}" for i, e in enumerate(entries)]
                c, sel = imgui.Combo("##ref_sel", ref["sel_ref"], labels)
                if c:
                    ref["sel_ref"] = sel

                si = min(ref["sel_ref"], len(entries) - 1)
                entry = entries[si]

                if imgui.Button(f"Remove##{si}"):
                    _unload_ref(si)
                    ref["sel_ref"] = max(0, min(ref["sel_ref"], len(entries) - 1))
                    _save_config()
                else:
                    t = entry["transform"]
                    tf_changed = False

                    c, v = imgui.SliderFloat(f"Tx##{si}", t[0], -2.0, 2.0)
                    if c:
                        t[0] = v
                        tf_changed = True
                    c, v = imgui.SliderFloat(f"Ty##{si}", t[1], -2.0, 2.0)
                    if c:
                        t[1] = v
                        tf_changed = True
                    c, v = imgui.SliderFloat(f"Tz##{si}", t[2], -2.0, 2.0)
                    if c:
                        t[2] = v
                        tf_changed = True
                    c, v = imgui.SliderFloat(f"Rx##{si}", t[3], -180.0, 180.0)
                    if c:
                        t[3] = v
                        tf_changed = True
                    c, v = imgui.SliderFloat(f"Ry##{si}", t[4], -180.0, 180.0)
                    if c:
                        t[4] = v
                        tf_changed = True
                    c, v = imgui.SliderFloat(f"Rz##{si}", t[5], -180.0, 180.0)
                    if c:
                        t[5] = v
                        tf_changed = True

                    if tf_changed:
                        _apply_ref_transform(entry)
                        _save_config()

                    if imgui.Button(f"Reset Transform##{si}"):
                        entry["transform"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        _apply_ref_transform(entry)
                        _save_config()

            imgui.TreePop()

        imgui.Separator()

        # ---- User Scene (global transform) ----
        if imgui.TreeNode("User Scene"):
            uobj = get_user_objects()
            imgui.Text(f"Objects: {', '.join(sorted(uobj.keys())) if uobj else '(none)'}")
            imgui.Text("(transforms applied via instance transforms)")

            ut = user_scene_tf
            tf_changed = False
            c, v = imgui.DragFloat("Scene Tx", ut[0], 0.001, -10.0, 10.0)
            if c:
                ut[0] = v
                tf_changed = True
            c, v = imgui.DragFloat("Scene Ty", ut[1], 0.001, -10.0, 10.0)
            if c:
                ut[1] = v
                tf_changed = True
            c, v = imgui.DragFloat("Scene Tz", ut[2], 0.001, -10.0, 10.0)
            if c:
                ut[2] = v
                tf_changed = True
            c, v = imgui.DragFloat("Scene Rx", ut[3], 0.5, -360.0, 360.0)
            if c:
                ut[3] = v
                tf_changed = True
            c, v = imgui.DragFloat("Scene Ry", ut[4], 0.5, -360.0, 360.0)
            if c:
                ut[4] = v
                tf_changed = True
            c, v = imgui.DragFloat("Scene Rz", ut[5], 0.5, -360.0, 360.0)
            if c:
                ut[5] = v
                tf_changed = True
            if imgui.Button("Reset Scene Transform"):
                user_scene_tf[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                tf_changed = True

            if tf_changed:
                apply_global_transform(_build_scene_transform(user_scene_tf))
                sgui.update()
                _save_config()

            imgui.TreePop()

        imgui.Separator()

        # ---- Simulation ----
        if imgui.TreeNode("Simulation"):
            kf_count = len(rec["keyframes"])
            imgui.Text(f"Keyframes: {kf_count}")

            if not _sim_live["active"]:
                imgui.Text("Scene initialized. Click to start advancing.")
                if imgui.Button("Start Simulation"):
                    _unload_all_refs()
                    fresh_schedule = _build_frame_schedule(rec["keyframes"], sim_dt, 1)
                    _sim_ctx["schedule"] = fresh_schedule
                    _sim_live["active"] = True
                    _sim_live["total_frames"] = len(fresh_schedule)
                    _sim_live["run"] = False
                    print(f"[sim] Started: {len(fresh_schedule)} frames")
            else:
                _sgui = _sim_ctx["sgui"]
                _sio = _sim_ctx["sio"]

                if imgui.Button("Play / Pause##sim"):
                    _sim_live["run"] = not _sim_live["run"]
                imgui.SameLine()
                if imgui.Button("Step Once##sim"):
                    _do_sim_step(_sio)
                    controller.update_visual_meshes()
                    _sgui.update()

                c, spd = imgui.SliderInt("Steps/tick##sim", _sim_live["steps_per_tick"], 1, 8)
                if c:
                    _sim_live["steps_per_tick"] = int(max(1, spd))

                c, v = imgui.Checkbox("Dump state##sim", _sim_live["dump_enabled"])
                if c:
                    _sim_live["dump_enabled"] = v
                c, v = imgui.Checkbox("Export surface OBJ##sim", _sim_live["export_surface"])
                if c:
                    _sim_live["export_surface"] = v

                cur_frame = world.frame()
                tf = _sim_live["total_frames"]
                imgui.Text(f"Frame: {cur_frame} / {tf}")
                if tf > 0:
                    imgui.Text(f"Progress: {min(cur_frame / max(tf, 1), 1.0) * 100:.1f}%")

                if _sim_live["run"]:
                    for _ in range(_sim_live["steps_per_tick"]):
                        _do_sim_step(_sio)
                        if world.frame() >= tf and tf > 0:
                            _sim_live["run"] = False
                            print("[sim] Playback complete")
                            break
                    controller.update_visual_meshes()
                    _sgui.update()
                    _sim_ctx["Timer"].report()

                imgui.Separator()
                c, val = imgui.InputInt("Target Frame##sim", _sim_live["recover_frame"])
                if c:
                    _sim_live["recover_frame"] = max(0, val)
                if imgui.Button("Recover##sim"):
                    target = _sim_live["recover_frame"]
                    if world.recover(target):
                        world.retrieve()
                        controller.update_visual_meshes()
                        _sgui.update()
                        print(f"[sim] recovered to frame {target}")
                    else:
                        print(f"[sim] recover to frame {target} failed")

                imgui.SameLine()
                if imgui.Button("Replay to##sim"):
                    target = _sim_live["recover_frame"]
                    world.recover(0)
                    world.retrieve()
                    _sim_live["replaying"] = True
                    _sim_live["replay_target"] = target
                    print(f"[sim] replaying from 0 to {target} ...")

                if _sim_live.get("replaying"):
                    replay_target = _sim_live["replay_target"]
                    steps_this_tick = min(_sim_live["steps_per_tick"], replay_target - world.frame())
                    for _ in range(max(1, steps_this_tick)):
                        _do_sim_step(_sio)
                        if world.frame() >= replay_target:
                            _sim_live["replaying"] = False
                            print(f"[sim] replay reached frame {world.frame()}")
                            break
                    controller.update_visual_meshes()
                    _sgui.update()
                    imgui.Text(f"Replaying: {world.frame()} / {replay_target}")

                imgui.Separator()
                if imgui.Button("Rebuild Schedule##sim"):
                    fresh = _build_frame_schedule(rec["keyframes"], sim_dt, 1)
                    _sim_ctx["schedule"] = fresh
                    _sim_live["total_frames"] = len(fresh)
                    print(f"[sim] Rebuilt: {len(rec['keyframes'])} kfs -> {len(fresh)} frames")

            imgui.TreePop()

    ps.set_user_callback(on_update)
    ps.show()

def _load_keyframes() -> list[dict]:
    """Load motion_keyframes.json and return list of keyframe dicts."""
    if not _RECORD_FILE.exists():
        return []
    data = json.loads(_RECORD_FILE.read_text(encoding="utf-8"))
    return [
        {
            "joints": kf["joints"],
            "dt": kf.get("dt", 1.0),
            "left_orient": kf.get("left_orient", ""),
            "right_orient": kf.get("right_orient", ""),
        }
        for kf in data
    ]


def _build_frame_schedule(
    keyframes: list[dict], sim_dt: float, substep: int
) -> list[dict]:
    """Expand keyframes into a per-frame joint-angle schedule.

    Each keyframe's ``dt`` is the real-time duration to interpolate from the
    previous pose to this pose.  We convert to simulation frames
    (``frame_count = ceil(dt / sim_dt)``), then linearly interpolate joint
    angles for every frame in between.

    Returns a list of ``{"joints": {name: angle}, ...}`` -- one per sim frame.
    """
    if not keyframes:
        return []

    joint_names = list(keyframes[0]["joints"].keys())
    schedule: list[dict] = []

    prev_joints = dict(keyframes[0]["joints"])
    schedule.append({"joints": dict(prev_joints)})

    for kf in keyframes[1:]:
        duration = kf["dt"]
        n_frames = max(1, int(np.ceil(duration / sim_dt)))
        dst = kf["joints"]
        for fi in range(1, n_frames + 1):
            alpha = fi / n_frames
            frame_joints = {}
            for n in joint_names:
                v0 = prev_joints.get(n, 0.0)
                v1 = dst.get(n, 0.0)
                frame_joints[n] = v0 + (v1 - v0) * alpha
            schedule.append({
                "joints": frame_joints,
                "left_orient": kf.get("left_orient", ""),
                "right_orient": kf.get("right_orient", ""),
            })
        prev_joints = dict(dst)

    return schedule


def run_simulation(
    user_scene_fn: Any = None,
    recover_frame: int = 0,
) -> None:
    """Launch UIPC simulation directly (``--sim`` CLI mode).

    Builds the full scene, calls ``world.init()``, opens Polyscope GUI.

    Args:
        user_scene_fn: Optional ``callable(scene, world)`` to add custom
            objects before ``world.init()``.
        recover_frame: If > 0, call ``world.recover(frame)`` instead of
            starting from scratch.
    """
    from asset_dir import AssetDir
    from uipc import Logger, SceneIO, Timer, view
    from uipc.core import Engine, Scene, World
    from uipc.gui import SceneGUI
    from uipc.unit import GPa

    repo_root = Path(__file__).resolve().parents[3]
    urdf_path = repo_root / "DemoAssets" / "marvin_bimanual" / "urdf" / "marvin_pika.urdf"
    output_dir = Path(AssetDir.output_path(__file__))

    controller = URDFController(urdf_path, mesh_source="visual")
    joint_names = controller.joint_names

    if _JOINT_FILE.exists():
        try:
            loaded = json.loads(_JOINT_FILE.read_text(encoding="utf-8"))
            jd = loaded.get("joints", loaded) if isinstance(loaded, dict) else loaded
            init_joints = {n: jd.get(n, 0.0) for n in joint_names}
        except (json.JSONDecodeError, KeyError):
            init_joints = {n: 0.0 for n in joint_names}
    else:
        init_joints = {n: 0.0 for n in joint_names}

    sim_keyframes = _load_keyframes()
    if sim_keyframes:
        kf0 = sim_keyframes[0]
        for n in joint_names:
            if n in kf0["joints"]:
                init_joints[n] = kf0["joints"][n]

    controller.set_joint_positions(init_joints)

    Logger.set_level(Logger.Level.Info)
    Timer.enable_all()
    engine = Engine("cuda", str(output_dir))
    world = World(engine)

    config = Scene.default_config()
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["linear_system"]["tol_rate"] = 1e-4
    config["newton"]["max_iter"] = 256
    scene = Scene(config)
    scene.animator().substep(1)

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0 * GPa)

    _SIM_LINK_PATTERNS = ["*Link8*", "*Link9*"]
    controller.create_ipc_bodies(
        scene,
        object_prefix="robot_link",
        stc_strength=np.array([120.0, 120.0], dtype=np.float64),
        include_patterns=_SIM_LINK_PATTERNS,
    )
    controller.configure_contact(scene, enable_self=False, enable_default=True)
    controller.apply_to_scene(snap=True)

    if user_scene_fn is not None:
        user_scene_fn(scene, world)
        _sim_scene_tf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if _CONFIG_FILE.exists():
            try:
                _pre = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
                _st = _pre.get("user_scene_transform")
                if _st and len(_st) == 6:
                    _sim_scene_tf = _st
            except (json.JSONDecodeError, KeyError):
                pass
        apply_global_transform(_build_scene_transform(_sim_scene_tf))

        sim_right_binding = controller.find_binding("*Link9_R*")
        if sim_right_binding is not None:
            stitch_string_to_gripper(
                scene,
                sim_right_binding.geo_slot,
                sim_right_binding.rest_geo_slot,
            )

    sim_dt = float(view(scene.config().find("dt"))[0])
    schedule = _build_frame_schedule(sim_keyframes, sim_dt, 1)
    total_frames = len(schedule)
    print(f"[sim] {len(sim_keyframes)} keyframes -> {total_frames} interpolated frames (dt={sim_dt})")

    def on_frame(info: Any, ctrl: URDFController) -> None:
        frame = int(info.frame())
        if frame < total_frames:
            ctrl.set_joint_positions(schedule[frame]["joints"])

    controller.bind_animator(scene, on_frame=on_frame)

    world.init(scene)
    world.retrieve()

    if recover_frame > 0:
        print(f"[sim] Recovering to frame {recover_frame} ...")
        if not world.recover(recover_frame):
            print(f"[sim] WARNING: recover({recover_frame}) failed, starting from 0")

    ps.init()
    ps.set_up_dir("z_up")

    controller.register_visual_only_meshes(transparency=0.45)
    controller.update_visual_meshes()

    sgui = SceneGUI(scene, "split")
    sio = SceneIO(scene)
    sgui.register()
    sgui.set_edge_width(1.0)

    sim_stitch_net = None
    sim_stitch_result = build_stitch_line_nodes()
    if sim_stitch_result is not None:
        sim_stitch_nodes, sim_stitch_edges = sim_stitch_result
        sim_stitch_net = ps.register_curve_network(
            "stitch_line", sim_stitch_nodes, sim_stitch_edges, radius=0.001, color=(1.0, 0.2, 0.2)
        )

    _rs_seq_dir = Path(__file__).resolve().parent / "results" / "v3" / "seq"

    sim_state = {
        "run": False,
        "steps_per_tick": 1,
        "dump_enabled": True,
        "export_surface": True,
        "recover_frame": 0,
        "rec_interp_time": 1.0,
    }

    def _sim_step() -> None:
        world.advance()
        world.retrieve()
        if sim_state["dump_enabled"]:
            world.dump()
        if sim_state["export_surface"]:
            sio.write_surface(f"{output_dir}/surface_{world.frame()}.obj")
        f = world.frame()
        export_frame_npy(_rs_seq_dir, f)
        joints_dir = _rs_seq_dir / "joints"
        joints_dir.mkdir(parents=True, exist_ok=True)
        joint_vals = {n: controller._joint_state.get(n, 0.0) for n in controller.joint_names}
        np.save(str(joints_dir / f"{f}.npy"), joint_vals)

    def _update_stitch_line() -> None:
        if sim_stitch_net is not None:
            result = build_stitch_line_nodes()
            if result is not None:
                sim_stitch_net.update_node_positions(result[0])

    def _rebuild_schedule() -> None:
        nonlocal schedule, total_frames, sim_keyframes
        sim_keyframes = _load_keyframes()
        schedule = _build_frame_schedule(sim_keyframes, sim_dt, 1)
        total_frames = len(schedule)
        print(f"[sim] Rebuilt schedule: {len(sim_keyframes)} keyframes -> {total_frames} frames")

    def on_sim_update() -> None:
        nonlocal schedule, total_frames

        imgui.Text("=== UIPC Simulation ===")
        imgui.Separator()

        if imgui.Button("Play / Pause"):
            sim_state["run"] = not sim_state["run"]
        imgui.SameLine()
        if imgui.Button("Step Once"):
            _sim_step()
            controller.update_visual_meshes()
            sgui.update()
            _update_stitch_line()

        c, spd = imgui.SliderInt("Steps/tick", sim_state["steps_per_tick"], 1, 8)
        if c:
            sim_state["steps_per_tick"] = int(max(1, spd))

        c, v = imgui.Checkbox("Dump state", sim_state["dump_enabled"])
        if c:
            sim_state["dump_enabled"] = v
        c, v = imgui.Checkbox("Export surface OBJ", sim_state["export_surface"])
        if c:
            sim_state["export_surface"] = v

        cur_frame = world.frame()
        imgui.Text(f"Frame: {cur_frame} / {total_frames}")
        if total_frames > 0:
            progress = min(cur_frame / max(total_frames, 1), 1.0)
            imgui.Text(f"Progress: {progress * 100:.1f}%")

        if sim_state["run"]:
            for _ in range(sim_state["steps_per_tick"]):
                _sim_step()
                if world.frame() >= total_frames and total_frames > 0:
                    sim_state["run"] = False
                    print("[sim] Playback complete")
                    break
            controller.update_visual_meshes()
            sgui.update()
            _update_stitch_line()
            Timer.report()

        # ── Recover ──
        imgui.Separator()
        c, val = imgui.InputInt("Recover Frame", sim_state["recover_frame"])
        if c:
            sim_state["recover_frame"] = max(0, val)
        if imgui.Button("Recover##sim"):
            target = sim_state["recover_frame"]
            if world.recover(target):
                world.retrieve()
                controller.update_visual_meshes()
                sgui.update()
                _update_stitch_line()
                print(f"[sim] recovered to frame {target}")
            else:
                print(f"[sim] recover to frame {target} failed")

        # ── Live re-record ──
        imgui.Separator()
        imgui.Text("Re-record (appends from current pose)")
        c, dt = imgui.SliderFloat("New KF dt##sim_rec", sim_state["rec_interp_time"], 0.1, 5.0)
        if c:
            sim_state["rec_interp_time"] = dt

        if imgui.Button("Record Current Pose##sim_rec"):
            cur_joints = {}
            for b in controller.bindings:
                geo = b.geo_slot.geometry()
                tf = np.array(view(geo.transforms()), copy=False).reshape(-1, 4, 4)[0]
                cur_joints[b.node_name] = tf
            fk_joints = {n: controller._joint_state.get(n, 0.0) for n in controller.joint_names}
            kf = {
                "joints": fk_joints,
                "dt": sim_state["rec_interp_time"],
                "left_orient": "",
                "right_orient": "",
            }
            sim_keyframes.append(kf)
            data = [
                {
                    "joints": k["joints"],
                    "dt": k["dt"],
                    "left_orient": k.get("left_orient", ""),
                    "right_orient": k.get("right_orient", ""),
                }
                for k in sim_keyframes
            ]
            with open(_RECORD_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[sim-rec] Appended keyframe #{len(sim_keyframes)} (dt={kf['dt']:.2f}s)")

        imgui.SameLine()
        if imgui.Button("Rebuild Schedule##sim_rec"):
            _rebuild_schedule()

    on_sim_update._refs = (engine, world, scene, sgui, sio, controller)
    ps.set_user_callback(on_sim_update)
    ps.show()


# =====================================================================
# Entry point
# =====================================================================

def run_export_recover(max_frame: int) -> None:
    """No-GUI mode: init scene, recover frames 0..max_frame, export NPY each frame."""
    from uipc import Logger, SceneIO, Timer, view
    from uipc.core import Engine, Scene, World
    from uipc.unit import GPa

    repo_root = Path(__file__).resolve().parents[3]
    urdf_path = repo_root / "DemoAssets" / "marvin_bimanual" / "urdf" / "marvin_pika.urdf"
    from asset_dir import AssetDir
    output_dir = Path(AssetDir.output_path(__file__))
    seq_dir = Path(__file__).resolve().parent / "results" / "v3" / "seq"

    controller = URDFController(urdf_path, mesh_source="visual")
    joint_names = controller.joint_names

    if _JOINT_FILE.exists():
        try:
            loaded = json.loads(_JOINT_FILE.read_text(encoding="utf-8"))
            jd = loaded.get("joints", loaded) if isinstance(loaded, dict) else loaded
            init_joints = {n: jd.get(n, 0.0) for n in joint_names}
        except (json.JSONDecodeError, KeyError):
            init_joints = {n: 0.0 for n in joint_names}
    else:
        init_joints = {n: 0.0 for n in joint_names}

    sim_keyframes = _load_keyframes()
    if sim_keyframes:
        for n in joint_names:
            if n in sim_keyframes[0]["joints"]:
                init_joints[n] = sim_keyframes[0]["joints"][n]
    controller.set_joint_positions(init_joints)

    Logger.set_level(Logger.Level.Info)
    engine = Engine("cuda", str(output_dir))
    world = World(engine)

    config = Scene.default_config()
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["linear_system"]["tol_rate"] = 1e-4
    config["newton"]["max_iter"] = 256
    scene = Scene(config)
    scene.animator().substep(1)

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0 * GPa)

    _SIM_LINK_PATTERNS = ["*Link8*", "*Link9*"]
    controller.create_ipc_bodies(
        scene,
        object_prefix="robot_link",
        stc_strength=np.array([120.0, 120.0], dtype=np.float64),
        include_patterns=_SIM_LINK_PATTERNS,
    )
    controller.configure_contact(scene, enable_self=False, enable_default=True)
    controller.apply_to_scene(snap=True)

    user_load_scene(scene, world)
    _sim_scene_tf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if _CONFIG_FILE.exists():
        try:
            _pre = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            _st = _pre.get("user_scene_transform")
            if _st and len(_st) == 6:
                _sim_scene_tf = _st
        except (json.JSONDecodeError, KeyError):
            pass
    apply_global_transform(_build_scene_transform(_sim_scene_tf))

    sim_right_binding = controller.find_binding("*Link9_R*")
    if sim_right_binding is not None:
        stitch_string_to_gripper(
            scene,
            sim_right_binding.geo_slot,
            sim_right_binding.rest_geo_slot,
        )

    world.init(scene)
    world.retrieve()
    export_frame_npy(seq_dir, world.frame())
    joints_dir = seq_dir / "joints"
    joints_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(joints_dir / f"{world.frame()}.npy"),
            {n: controller._joint_state.get(n, 0.0) for n in joint_names})

    exported = 0
    for target in range(1, max_frame + 1):
        if not world.recover(target):
            print(f"[export-recover] no dump at frame {target}, stopping")
            break
        world.retrieve()
        export_frame_npy(seq_dir, target)
        np.save(str(joints_dir / f"{target}.npy"),
                {n: controller._joint_state.get(n, 0.0) for n in joint_names})
        exported += 1
        if exported % 100 == 0:
            print(f"[export-recover] {exported} frames exported ...")

    print(f"[export-recover] Done: {exported} frames exported to {seq_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="URDF robot controller / simulator")
    parser.add_argument("--sim", action="store_true", help="Launch UIPC simulation mode")
    parser.add_argument("--recover", type=int, default=0, help="Recover simulation from frame N")
    parser.add_argument(
        "--export-recover",
        type=int,
        default=-1,
        metavar="MAX_FRAME",
        help="No-GUI: recover frames 0..MAX_FRAME, export NPY, then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.export_recover >= 0:
        run_export_recover(args.export_recover)
    elif args.sim:
        run_simulation(user_scene_fn=user_load_scene, recover_frame=args.recover)
    else:
        main()
