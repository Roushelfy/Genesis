"""Reusable GUI library for URDF robot inspection and UIPC simulation.

Provides ``URDFGuiApp``, a configurable application class that handles:
- FK/IK joint inspector with polyscope GUI
- Motion keyframe recording and playback
- UIPC physics simulation with live controls
- Reference OBJ overlay and scene transform editing
- Simulation state export (recover + NPY)

Usage (thin entry script)::

    from urdf_gui import URDFGuiApp, SceneState

    def load_my_scene(scene, world, ss):
        ...

    app = URDFGuiApp(
        urdf_path=...,
        load_scene_fn=load_my_scene,
        ...
    )
    app.main_cli()
"""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polyscope as ps
from polyscope import imgui

from urdf_controller import URDFController

# ---------------------------------------------------------------------------
# Orientation presets (Z-up world)
# ---------------------------------------------------------------------------

ORIENTATION_PRESETS: dict[str, np.ndarray] = {
    "None (pos only)": np.zeros((3, 3)),
    "Horiz -Z": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    "Horiz -Z inv": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64),
    "Horiz +X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    "Horiz +X inv": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64),
    "Horiz -X": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    "Horiz -X inv": np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], dtype=np.float64),
    "Horiz +Y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
    "Horiz +Y inv": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64),
    "Horiz -Y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    "Horiz -Y inv": np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64),
    "Vert +Z": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64),
    "Vert +Z inv": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
}
PRESET_NAMES: list[str] = list(ORIENTATION_PRESETS.keys())


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def rot_to_display(R: np.ndarray) -> str:
    """Format a 3x3 rotation as local X/Y/Z axis directions."""
    return (
        f"  X: ({R[0, 0]:+.3f}, {R[1, 0]:+.3f}, {R[2, 0]:+.3f})\n"
        f"  Y: ({R[0, 1]:+.3f}, {R[1, 1]:+.3f}, {R[2, 1]:+.3f})\n"
        f"  Z: ({R[0, 2]:+.3f}, {R[1, 2]:+.3f}, {R[2, 2]:+.3f})"
    )


def try_key_pressed(key_char: str) -> bool:
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


def build_scene_transform(t: list[float]) -> np.ndarray:
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


def parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Parse OBJ, returning (vertices, faces_or_None, edges_or_None)."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    edges: list[list[int]] = []
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


def make_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build 3x3 rotation from Euler angles (degrees), XYZ order."""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def closest_point_on_triangle(p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Closest point on triangle (v0,v1,v2) to point p.  Ericson p.141."""
    ab = v1 - v0
    ac = v2 - v0
    ap = p - v0
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    if d1 <= 0 and d2 <= 0:
        return v0.copy()
    bp = p - v1
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    if d3 >= 0 and d4 <= d3:
        return v1.copy()
    cp = p - v2
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)
    if d6 >= 0 and d5 <= d6:
        return v2.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        w = d1 / (d1 - d3)
        return v0 + w * ab
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return v0 + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return v1 + w * (v2 - v1)
    denom = 1.0 / (va + vb + vc)
    sv = vb * denom
    sw = vc * denom
    return v0 + sv * ab + sw * ac


def load_keyframes(record_file: Path) -> list[dict]:
    """Load motion_keyframes.json and return list of keyframe dicts."""
    if not record_file.exists():
        return []
    data = json.loads(record_file.read_text(encoding="utf-8"))
    return [
        {
            "joints": kf["joints"],
            "dt": kf.get("dt", 1.0),
            "left_orient": kf.get("left_orient", ""),
            "right_orient": kf.get("right_orient", ""),
        }
        for kf in data
    ]


def build_frame_schedule(keyframes: list[dict], sim_dt: float, substep: int) -> list[dict]:
    """Expand keyframes into a per-frame joint-angle schedule.

    Each keyframe's ``dt`` is the real-time duration to interpolate from the
    previous pose to this pose.  Returns a list of per-sim-frame dicts.
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
            schedule.append(
                {
                    "joints": frame_joints,
                    "left_orient": kf.get("left_orient", ""),
                    "right_orient": kf.get("right_orient", ""),
                }
            )
        prev_joints = dict(dst)
    return schedule


# ---------------------------------------------------------------------------
# SceneState - mutable container for user-scene objects
# ---------------------------------------------------------------------------


class SceneState:
    """Holds references and cached data for user-scene objects.

    Populated by the ``load_scene_fn`` callback, then used by the GUI
    library for global transforms, stitch visualisation, and export.
    """

    def __init__(self) -> None:
        self.objects: dict[str, Any] = {}
        self.geo_slots: dict[str, Any] = {}
        self.rest_geo_slots: dict[str, Any] = {}
        self.original_transforms: dict[str, np.ndarray] = {}
        self.original_positions: dict[str, np.ndarray] = {}
        self.obj_sources: dict[str, Path] = {}
        self.fem_objects: set[str] = set()
        self.stitch_vis: dict[str, Any] = {}

    def clear(self) -> None:
        self.objects.clear()
        self.geo_slots.clear()
        self.rest_geo_slots.clear()
        self.original_transforms.clear()
        self.original_positions.clear()
        self.obj_sources.clear()
        self.fem_objects.clear()
        self.stitch_vis.clear()

    def get_objects(self) -> dict[str, Any]:
        return self.objects

    def apply_global_transform(self, mat44: np.ndarray) -> None:
        """Apply a global 4x4 transform to all user-scene objects.

        ABD objects: ``view(geo.transforms())[0] = mat44 @ orig``.
        FEM objects: transform cached original positions.
        """
        from uipc import view

        for name, gs in self.geo_slots.items():
            orig_tf = self.original_transforms.get(name, np.eye(4))
            geo = gs.geometry()
            if name in self.fem_objects:
                orig_pts = self.original_positions[name]
                delta = mat44 @ np.linalg.inv(orig_tf)
                R = delta[:3, :3]
                t = delta[:3, 3]
                pts = np.array(view(geo.positions()), copy=False).reshape(-1, 3)
                pts[:] = orig_pts @ R.T + t
                rest_gs = self.rest_geo_slots.get(name)
                if rest_gs is not None:
                    rest_pts = np.array(view(rest_gs.geometry().positions()), copy=False).reshape(-1, 3)
                    rest_pts[:] = orig_pts @ R.T + t
            else:
                view(geo.transforms())[0] = mat44 @ orig_tf

    def build_stitch_line_nodes(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Build (nodes, edges) arrays for the stitch curve visualisation."""
        from uipc import view

        if "string_gs" not in self.stitch_vis:
            return None
        string_pos = np.array(view(self.stitch_vis["string_gs"].geometry().positions()), copy=False).reshape(-1, 3)
        gripper_geo = self.stitch_vis["gripper_gs"].geometry()
        gripper_local = np.array(view(gripper_geo.positions()), copy=False).reshape(-1, 3)
        gripper_tris = self.stitch_vis["gripper_tris"]
        tf44 = np.array(view(gripper_geo.transforms()), copy=False).reshape(-1, 4, 4)[0]
        R = tf44[:3, :3]
        t = tf44[:3, 3]
        gripper_pos = gripper_local @ R.T + t
        pairs = self.stitch_vis["pairs"]
        n = pairs.shape[0]
        if n == 0:
            return None
        nodes = np.zeros((2 * n, 3), dtype=np.float64)
        edges = np.zeros((n, 2), dtype=np.int32)
        for i in range(n):
            vid, tid = int(pairs[i, 0]), int(pairs[i, 1])
            pt = string_pos[vid]
            nodes[2 * i] = pt
            tri = gripper_tris[tid]
            nodes[2 * i + 1] = closest_point_on_triangle(
                pt, gripper_pos[tri[0]], gripper_pos[tri[1]], gripper_pos[tri[2]]
            )
            edges[i] = [2 * i, 2 * i + 1]
        return nodes, edges


# ---------------------------------------------------------------------------
# SequenceExporter
# ---------------------------------------------------------------------------


class SequenceExporter:
    """Accumulate per-frame simulation state and write consolidated NPY files."""

    def __init__(
        self,
        joint_names: list[str],
        scene_state: SceneState,
        dt: float = 0.001,
        urdf_rel: str = "",
    ):
        self._joint_names = list(joint_names)
        self._ss = scene_state
        self._dt = dt
        self._urdf_rel = urdf_rel
        self._frame_ids: list[int] = []
        self._rigid_frames: dict[str, list[np.ndarray]] = {}
        self._fem_frames: dict[str, list[np.ndarray]] = {}
        self._joint_frames: list[np.ndarray] = []

    def capture(self, frame: int, joint_state: dict[str, float] | None = None) -> None:
        from uipc import view

        self._frame_ids.append(frame)
        for name, gs in self._ss.geo_slots.items():
            geo = gs.geometry()
            if name in self._ss.fem_objects:
                pos = np.array(view(geo.positions()), copy=True).reshape(-1, 3)
                self._fem_frames.setdefault(name, []).append(pos)
            else:
                tf = np.array(view(geo.transforms()), copy=True).reshape(-1, 4, 4)[0]
                self._rigid_frames.setdefault(name, []).append(tf)
        if joint_state is not None:
            jv = np.array([joint_state.get(n, 0.0) for n in self._joint_names], dtype=np.float64)
            self._joint_frames.append(jv)

    def save(self, seq_dir: Path) -> None:
        seq_dir.mkdir(parents=True, exist_ok=True)
        frame_skip = 1
        if len(self._frame_ids) >= 2:
            frame_skip = max(1, self._frame_ids[1] - self._frame_ids[0])
        meta: dict = {
            "frame_count": len(self._frame_ids),
            "frame_ids": self._frame_ids,
            "frame_skip": frame_skip,
            "dt": self._dt,
            "objects": {},
        }
        for name, frames in self._rigid_frames.items():
            obj_dir = seq_dir / name
            obj_dir.mkdir(parents=True, exist_ok=True)
            arr = np.stack(frames, axis=0)
            npy_name = "transforms.npy"
            np.save(str(obj_dir / npy_name), arr)
            self._copy_mesh_obj(name, obj_dir)
            meta["objects"][name] = {"type": "rigid", "data": f"{name}/{npy_name}"}
        for name, frames in self._fem_frames.items():
            obj_dir = seq_dir / name
            obj_dir.mkdir(parents=True, exist_ok=True)
            arr = np.stack(frames, axis=0)
            npy_name = "positions.npy"
            np.save(str(obj_dir / npy_name), arr)
            self._copy_mesh_obj(name, obj_dir)
            meta["objects"][name] = {"type": "fem", "data": f"{name}/{npy_name}"}
        if self._joint_frames:
            jarr = np.stack(self._joint_frames, axis=0)
            np.save(str(seq_dir / "joints.npy"), jarr)
            meta["joints"] = {"data": "joints.npy", "names": self._joint_names}
        if self._urdf_rel:
            meta["urdf"] = self._urdf_rel
        (seq_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"[seq-export] saved {len(self._frame_ids)} frames to {seq_dir}")

    def _copy_mesh_obj(self, name: str, obj_dir: Path) -> None:
        mesh_dst = obj_dir / "mesh.obj"
        if not mesh_dst.exists() and name in self._ss.obj_sources:
            src = self._ss.obj_sources[name]
            if src.exists():
                shutil.copy2(str(src), str(mesh_dst))


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

LoadSceneFn = Callable  # (scene, world, SceneState) -> None
SetupStitchFn = Callable  # (scene, URDFController, SceneState) -> None


# ---------------------------------------------------------------------------
# URDFGuiApp
# ---------------------------------------------------------------------------


class URDFGuiApp:
    """Configurable URDF inspection / simulation application.

    Args:
        urdf_path: Path to the robot URDF.
        sim_link_patterns: Glob patterns for links to simulate as IPC bodies.
        stc_strength: STC strength array for ``create_ipc_bodies``.
        load_scene_fn: ``(scene, world, SceneState) -> None`` callback that
            populates the scene with user objects and fills *SceneState*.
        setup_stitch_fn: Optional ``(scene, controller, SceneState) -> None``
            callback to create stitch constraints after scene loading.
        record_file: Path to motion keyframes JSON.
        joint_file: Path to saved joint angles JSON.
        config_file: Path to GUI config JSON (ref OBJs, scene transform).
        seq_dir: Directory for sequence export (``--export-recover``).
        entry_file: ``__file__`` of the entry script (for output directory).
    """

    def __init__(
        self,
        urdf_path: Path | str,
        sim_link_patterns: list[str],
        stc_strength: np.ndarray,
        load_scene_fn: LoadSceneFn,
        setup_stitch_fn: SetupStitchFn | None = None,
        record_file: Path | str | None = None,
        joint_file: Path | str | None = None,
        config_file: Path | str | None = None,
        seq_dir: Path | str | None = None,
        entry_file: str | Path = "",
    ):
        self._urdf_path = Path(urdf_path)
        self._sim_link_patterns = list(sim_link_patterns)
        self._stc_strength = stc_strength
        self._load_scene_fn = load_scene_fn
        self._setup_stitch_fn = setup_stitch_fn

        script_dir = Path(entry_file).resolve().parent if entry_file else Path.cwd()
        self._record_file = Path(record_file) if record_file else script_dir / "motion_keyframes.json"
        self._joint_file = Path(joint_file) if joint_file else script_dir / "joint_angles.json"
        self._config_file = Path(config_file) if config_file else script_dir / "urdf_gui_config.json"
        self._seq_dir = Path(seq_dir) if seq_dir else script_dir.parent / "v3" / "seq"
        self._entry_file = str(entry_file) if entry_file else __file__

        self.scene_state = SceneState()

        # Session state (set during run_*)
        self._controller: URDFController | None = None
        self._joint_names: list[str] = []
        self._limits: dict[str, tuple[float, float]] = {}
        self._end_effectors: list[str] = []
        self._left_ee = ""
        self._right_ee = ""

        self._state: dict[str, Any] = {}
        self._rec: dict[str, Any] = {}
        self._ref: dict[str, Any] = {}
        self._user_scene_tf: list[float] = [0.0] * 6
        self._sgui: Any = None
        self._sio: Any = None
        self._sim_ctx: dict[str, Any] = {}
        self._sim_live: dict[str, Any] = {}
        self._stitch_net: Any = None

    # ------------------------------------------------------------------
    # Common setup helpers
    # ------------------------------------------------------------------

    def _init_controller(self) -> URDFController:
        controller = URDFController(self._urdf_path, mesh_source="visual")
        self._controller = controller
        self._joint_names = controller.joint_names
        self._limits = controller.joint_limits
        self._end_effectors = controller.find_end_effectors()
        print(f"[urdf-gui] Actuated joints ({len(self._joint_names)}): {self._joint_names}")
        print(f"[urdf-gui] End effectors: {self._end_effectors}")
        return controller

    def _load_initial_state(self) -> None:
        """Load joints from file, then override with keyframe[0] if present."""
        self._state = {
            "joints": {n: 0.0 for n in self._joint_names},
            "ee_idx": 0,
            "ik_step": 0.01,
            "left_orient_idx": 0,
            "right_orient_idx": 0,
        }
        self._rec = {
            "keyframes": [],
            "interp_time": 1.0,
            "recording": False,
            "playing": False,
            "play_idx": 0,
            "play_t0": 0.0,
            "play_start_joints": {},
            "sel_kf": 0,
        }
        if self._joint_file.exists():
            try:
                loaded = json.loads(self._joint_file.read_text(encoding="utf-8"))
                jd = loaded.get("joints", loaded) if isinstance(loaded, dict) else loaded
                for n in self._joint_names:
                    if n in jd:
                        self._state["joints"][n] = jd[n]
                if isinstance(loaded, dict):
                    lo = loaded.get("left_orient", "")
                    ro = loaded.get("right_orient", "")
                    if lo in PRESET_NAMES:
                        self._state["left_orient_idx"] = PRESET_NAMES.index(lo)
                    if ro in PRESET_NAMES:
                        self._state["right_orient_idx"] = PRESET_NAMES.index(ro)
                print(f"[urdf-gui] Auto-loaded joint angles from {self._joint_file}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[urdf-gui] Failed to auto-load joints: {e}")

        if self._record_file.exists():
            try:
                loaded_kfs = json.loads(self._record_file.read_text(encoding="utf-8"))
                self._rec["keyframes"].clear()
                for entry in loaded_kfs:
                    self._rec["keyframes"].append(
                        {
                            "joints": entry["joints"],
                            "dt": entry.get("dt", 1.0),
                            "left_orient": entry.get("left_orient", ""),
                            "right_orient": entry.get("right_orient", ""),
                        }
                    )
                if self._rec["keyframes"]:
                    kf0 = self._rec["keyframes"][0]
                    for n in self._joint_names:
                        if n in kf0["joints"]:
                            self._state["joints"][n] = kf0["joints"][n]
                    lo = kf0.get("left_orient", "")
                    ro = kf0.get("right_orient", "")
                    if lo in PRESET_NAMES:
                        self._state["left_orient_idx"] = PRESET_NAMES.index(lo)
                    if ro in PRESET_NAMES:
                        self._state["right_orient_idx"] = PRESET_NAMES.index(ro)
                print(f"[urdf-gui] Auto-loaded {len(self._rec['keyframes'])} keyframes")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[urdf-gui] Failed to auto-load keyframes: {e}")

    def _create_engine_scene(self, output_dir: Path):
        """Create engine, world, scene with default config.  Returns (engine, world, scene)."""
        from uipc import Logger, Timer
        from uipc.core import Engine, Scene, World
        from uipc.unit import GPa

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
        return engine, world, scene

    def _setup_robot(self, scene: Any) -> None:
        """Create IPC bodies for the robot and configure contacts."""
        assert self._controller is not None
        self._controller.create_ipc_bodies(
            scene,
            object_prefix="robot_link",
            stc_strength=self._stc_strength,
            include_patterns=self._sim_link_patterns,
        )
        self._controller.configure_contact(scene, enable_self=False, enable_default=True)
        self._controller.apply_to_scene(snap=True)

    def _load_scene_transform(self) -> list[float]:
        """Load user_scene_transform from config file."""
        tf: list[float] = [0.0] * 6
        if self._config_file.exists():
            try:
                cfg = json.loads(self._config_file.read_text(encoding="utf-8"))
                saved = cfg.get("user_scene_transform")
                if saved and len(saved) == 6:
                    tf[:] = saved
            except (json.JSONDecodeError, KeyError):
                pass
        return tf

    def _apply_scene_and_stitch(self, scene: Any, world: Any) -> None:
        """Call load_scene_fn, apply global transform, call setup_stitch_fn."""
        self._user_scene_tf = self._load_scene_transform()
        self._load_scene_fn(scene, world, self.scene_state)
        self.scene_state.apply_global_transform(build_scene_transform(self._user_scene_tf))
        if self._setup_stitch_fn is not None:
            assert self._controller is not None
            self._setup_stitch_fn(scene, self._controller, self.scene_state)

    # ------------------------------------------------------------------
    # Inspector helpers
    # ------------------------------------------------------------------

    def _sync_view(self) -> None:
        assert self._controller is not None
        self._controller.apply_to_scene(snap=True)
        self._controller.update_visual_meshes()
        if self._sgui is not None:
            self._sgui.update()

    def _get_orientation(self, orient_idx: int) -> np.ndarray | None:
        name = PRESET_NAMES[orient_idx]
        R = ORIENTATION_PRESETS[name]
        if np.allclose(R, 0):
            return None
        return R

    def _do_ik(self, ee_name: str, delta: np.ndarray, orient_idx: int) -> None:
        assert self._controller is not None
        cur_tf = self._controller.get_link_transform(ee_name)
        cur_pos = cur_tf[:3, 3].copy()
        arm_joints = self._controller.find_arm_joints(ee_name)
        if len(arm_joints) == 0:
            return
        target_orient = self._get_orientation(orient_idx)
        orient_mode = "all" if target_orient is not None else None
        self._controller.solve_ik(
            ee_name,
            cur_pos + delta,
            target_orientation=target_orient,
            orientation_mode=orient_mode,
            arm_joints=arm_joints,
        )
        self._state["joints"] = self._controller.get_joint_positions()
        self._sync_view()

    def _auto_save(self) -> None:
        data = [
            {
                "joints": kf["joints"],
                "dt": kf["dt"],
                "left_orient": kf.get("left_orient", ""),
                "right_orient": kf.get("right_orient", ""),
            }
            for kf in self._rec["keyframes"]
        ]
        with open(self._record_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _save_config(self) -> None:
        data = {
            "ref_objs": [{"path": e["path"], "transform": e["transform"]} for e in self._ref["entries"]],
            "user_scene_transform": list(self._user_scene_tf),
        }
        with open(self._config_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _apply_ref_transform(self, entry: dict) -> None:
        t = entry["transform"]
        R = make_rotation(t[3], t[4], t[5])
        offset = np.array(t[:3])
        transformed = entry["verts"] @ R.T + offset
        if entry["surf"]:
            ps.get_surface_mesh(entry["surf"]).update_vertex_positions(transformed)
        if entry["curve"]:
            ps.get_curve_network(entry["curve"]).update_node_positions(transformed)

    def _load_ref_obj(self, path_str: str, transform: list[float] | None = None) -> bool:
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
            print(f"[ref] No vertices in {p}")
            return False
        idx = len(self._ref["entries"])
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
            print(f"[ref] Faces -> '{sname}' ({len(faces)} tris)")
        if edges is not None:
            cname = f"ref_line_{p.stem}_{idx}"
            ps.register_curve_network(cname, transformed, edges, color=(1.0, 0.4, 0.2), radius=0.001)
            entry["curve"] = cname
            print(f"[ref] Lines -> '{cname}' ({len(edges)} edges)")
        self._ref["entries"].append(entry)
        return True

    def _unload_ref(self, idx: int) -> None:
        e = self._ref["entries"][idx]
        if e["surf"]:
            ps.remove_surface_mesh(e["surf"], error_if_absent=False)
        if e["curve"]:
            ps.remove_curve_network(e["curve"])
        self._ref["entries"].pop(idx)

    def _unload_all_refs(self) -> None:
        while self._ref["entries"]:
            self._unload_ref(0)

    # ------------------------------------------------------------------
    # Inspector GUI panels
    # ------------------------------------------------------------------

    def _draw_joint_panel(self) -> None:
        if not imgui.TreeNode("Joint Angles"):
            return
        changed = False
        for name in self._joint_names:
            lo, hi = self._limits[name]
            c, v = imgui.SliderFloat(name, self._state["joints"][name], lo, hi)
            if c:
                self._state["joints"][name] = v
                changed = True
        if changed:
            self._controller.set_joint_positions(self._state["joints"])
            self._sync_view()

        if imgui.Button("Reset All Joints"):
            self._state["joints"] = {n: 0.0 for n in self._joint_names}
            self._controller.set_joint_positions(self._state["joints"])
            self._sync_view()

        imgui.SameLine()
        if imgui.Button("Save Joints"):
            data = {
                "joints": self._state["joints"],
                "left_orient": PRESET_NAMES[self._state["left_orient_idx"]],
                "right_orient": PRESET_NAMES[self._state["right_orient_idx"]],
            }
            with open(self._joint_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[urdf-gui] Joint angles saved to {self._joint_file}")

        imgui.SameLine()
        if imgui.Button("Load Joints"):
            if self._joint_file.exists():
                with open(self._joint_file, encoding="utf-8") as f:
                    loaded = json.load(f)
                jd = loaded.get("joints", loaded) if isinstance(loaded, dict) else loaded
                for n in self._joint_names:
                    if n in jd:
                        self._state["joints"][n] = jd[n]
                self._controller.set_joint_positions(self._state["joints"])
                self._sync_view()
                if isinstance(loaded, dict):
                    lo = loaded.get("left_orient", "")
                    ro = loaded.get("right_orient", "")
                    if lo in PRESET_NAMES:
                        self._state["left_orient_idx"] = PRESET_NAMES.index(lo)
                    if ro in PRESET_NAMES:
                        self._state["right_orient_idx"] = PRESET_NAMES.index(ro)
                print(f"[urdf-gui] Joint angles loaded from {self._joint_file}")
            else:
                print(f"[urdf-gui] File not found: {self._joint_file}")
        imgui.TreePop()

    def _draw_ee_panel(self) -> None:
        if not imgui.TreeNode("End Effectors"):
            return
        for ee in self._end_effectors:
            tf = self._controller.get_link_transform(ee)
            pos = tf[:3, 3]
            rot = tf[:3, :3]
            imgui.Text(f"{ee}:")
            imgui.Text(f"  Pos: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            imgui.Text(rot_to_display(rot))
        imgui.TreePop()

    def _draw_ik_panel(self) -> None:
        if not imgui.TreeNode("IK Control"):
            return
        c, step = imgui.SliderFloat("IK Step (m)", self._state["ik_step"], 0.001, 0.05)
        if c:
            self._state["ik_step"] = step
        s = self._state["ik_step"]
        imgui.Separator()

        # Left gripper
        if imgui.TreeNode(f"Left: {self._left_ee}  [WASDQE]"):
            left_tf = self._controller.get_link_transform(self._left_ee)
            left_pos = left_tf[:3, 3]
            imgui.Text(f"Pos: ({left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f})")
            left_arm = self._controller.find_arm_joints(self._left_ee)
            imgui.Text(f"IK joints ({len(left_arm)}): {', '.join(left_arm)}")
            imgui.Text("Orientation:")
            c, oidx = imgui.Combo("##orient_L", self._state["left_orient_idx"], PRESET_NAMES)
            if c:
                self._state["left_orient_idx"] = oidx
            if imgui.Button("Lock current##lock_L"):
                R = left_tf[:3, :3].copy()
                ORIENTATION_PRESETS["Locked L"] = R
                if "Locked L" not in PRESET_NAMES:
                    PRESET_NAMES.append("Locked L")
                self._state["left_orient_idx"] = PRESET_NAMES.index("Locked L")
            for label, delta in [
                ("+X##Lik", [s, 0, 0]),
                ("-X##Lik", [-s, 0, 0]),
                ("+Y##Lik", [0, s, 0]),
                ("-Y##Lik", [0, -s, 0]),
                ("+Z##Lik", [0, 0, s]),
                ("-Z##Lik", [0, 0, -s]),
            ]:
                if label != "+X##Lik":
                    imgui.SameLine()
                if imgui.Button(label):
                    self._do_ik(self._left_ee, np.array(delta), self._state["left_orient_idx"])
            imgui.TreePop()

        imgui.Separator()

        # Right gripper
        if imgui.TreeNode(f"Right: {self._right_ee}  [IJKLUO]"):
            right_tf = self._controller.get_link_transform(self._right_ee)
            right_pos = right_tf[:3, 3]
            imgui.Text(f"Pos: ({right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f})")
            right_arm = self._controller.find_arm_joints(self._right_ee)
            imgui.Text(f"IK joints ({len(right_arm)}): {', '.join(right_arm)}")
            imgui.Text("Orientation:")
            c, oidx = imgui.Combo("##orient_R", self._state["right_orient_idx"], PRESET_NAMES)
            if c:
                self._state["right_orient_idx"] = oidx
            if imgui.Button("Lock current##lock_R"):
                R = right_tf[:3, :3].copy()
                ORIENTATION_PRESETS["Locked R"] = R
                if "Locked R" not in PRESET_NAMES:
                    PRESET_NAMES.append("Locked R")
                self._state["right_orient_idx"] = PRESET_NAMES.index("Locked R")
            for label, delta in [
                ("+X##Rik", [s, 0, 0]),
                ("-X##Rik", [-s, 0, 0]),
                ("+Y##Rik", [0, s, 0]),
                ("-Y##Rik", [0, -s, 0]),
                ("+Z##Rik", [0, 0, s]),
                ("-Z##Rik", [0, 0, -s]),
            ]:
                if label != "+X##Rik":
                    imgui.SameLine()
                if imgui.Button(label):
                    self._do_ik(self._right_ee, np.array(delta), self._state["right_orient_idx"])
            imgui.TreePop()

        imgui.Separator()

        # Keyboard polling
        left_delta = np.zeros(3, dtype=np.float64)
        if try_key_pressed("W"):
            left_delta[0] += s
        if try_key_pressed("S"):
            left_delta[0] -= s
        if try_key_pressed("A"):
            left_delta[1] += s
        if try_key_pressed("D"):
            left_delta[1] -= s
        if try_key_pressed("E"):
            left_delta[2] += s
        if try_key_pressed("Q"):
            left_delta[2] -= s
        if np.any(left_delta != 0):
            self._do_ik(self._left_ee, left_delta, self._state["left_orient_idx"])

        right_delta = np.zeros(3, dtype=np.float64)
        if try_key_pressed("I"):
            right_delta[0] += s
        if try_key_pressed("K"):
            right_delta[0] -= s
        if try_key_pressed("J"):
            right_delta[1] += s
        if try_key_pressed("L"):
            right_delta[1] -= s
        if try_key_pressed("O"):
            right_delta[2] += s
        if try_key_pressed("U"):
            right_delta[2] -= s
        if np.any(right_delta != 0):
            self._do_ik(self._right_ee, right_delta, self._state["right_orient_idx"])

        imgui.Text("Left  keys: W/S=X  A/D=Y  Q/E=Z")
        imgui.Text("Right keys: I/K=X  J/L=Y  U/O=Z")
        imgui.TreePop()

    def _draw_recording_panel(self) -> None:
        if not imgui.TreeNode("Motion Recording"):
            return
        rec = self._rec
        kfs = rec["keyframes"]

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
                "joints": dict(self._state["joints"]),
                "dt": rec["interp_time"],
                "left_orient": PRESET_NAMES[self._state["left_orient_idx"]],
                "right_orient": PRESET_NAMES[self._state["right_orient_idx"]],
            }
            insert_pos = rec["sel_kf"] + 1 if kfs else 0
            kfs.insert(insert_pos, kf)
            rec["sel_kf"] = insert_pos
            self._auto_save()
            print(f"[record] Keyframe inserted at #{insert_pos} (dt={kf['dt']:.2f}s), total={len(kfs)}")

        imgui.SameLine()
        if imgui.Button("Undo Last") and len(kfs) > 0:
            kfs.pop()
            self._auto_save()
            print(f"[record] Last keyframe removed, {len(kfs)} remaining")

        imgui.Text(f"Keyframes: {len(kfs)}")

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
                self._state["joints"] = dict(kfs[idx]["joints"])
                self._controller.set_joint_positions(self._state["joints"])
                self._sync_view()
                lo = kfs[idx].get("left_orient", "")
                ro = kfs[idx].get("right_orient", "")
                if lo in PRESET_NAMES:
                    self._state["left_orient_idx"] = PRESET_NAMES.index(lo)
                if ro in PRESET_NAMES:
                    self._state["right_orient_idx"] = PRESET_NAMES.index(ro)
            imgui.SameLine()
            if imgui.Button("Delete##kf_del"):
                idx = min(rec["sel_kf"], len(kfs) - 1)
                kfs.pop(idx)
                rec["sel_kf"] = max(0, min(rec["sel_kf"], len(kfs) - 1))
                self._auto_save()
                print(f"[record] Keyframe deleted, {len(kfs)} remaining")
            imgui.SameLine()
            if imgui.Button("Update##kf_upd"):
                idx = min(rec["sel_kf"], len(kfs) - 1)
                kfs[idx]["joints"] = dict(self._state["joints"])
                kfs[idx]["dt"] = rec["interp_time"]
                kfs[idx]["left_orient"] = PRESET_NAMES[self._state["left_orient_idx"]]
                kfs[idx]["right_orient"] = PRESET_NAMES[self._state["right_orient_idx"]]
                self._auto_save()
                print(f"[record] Keyframe {idx} updated")

            idx = min(rec["sel_kf"], len(kfs) - 1)
            c, new_dt = imgui.SliderFloat(f"dt (KF {idx})", kfs[idx]["dt"], 0.1, 5.0)
            if c:
                kfs[idx]["dt"] = new_dt
                self._auto_save()

        imgui.Separator()

        # Save / Load / Clear
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
            with open(self._record_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[record] Saved {len(kfs)} keyframes to {self._record_file}")

        imgui.SameLine()
        if imgui.Button("Load Motion"):
            if self._record_file.exists():
                with open(self._record_file, encoding="utf-8") as f:
                    loaded = json.load(f)
                kfs.clear()
                for entry in loaded:
                    kfs.append(
                        {
                            "joints": entry["joints"],
                            "dt": entry.get("dt", 1.0),
                            "left_orient": entry.get("left_orient", ""),
                            "right_orient": entry.get("right_orient", ""),
                        }
                    )
                rec["sel_kf"] = 0
                print(f"[record] Loaded {len(kfs)} keyframes from {self._record_file}")
            else:
                print(f"[record] File not found: {self._record_file}")

        imgui.SameLine()
        if imgui.Button("Clear All"):
            kfs.clear()
            rec["sel_kf"] = 0
            self._auto_save()
            print("[record] All keyframes cleared")

        imgui.Separator()

        # Timeline scrub
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
                for n in self._joint_names:
                    v0 = src.get(n, 0.0)
                    v1 = dst.get(n, 0.0)
                    self._state["joints"][n] = v0 + (v1 - v0) * alpha
                self._controller.set_joint_positions(self._state["joints"])
                self._sync_view()
                rec["sel_kf"] = seg
                lo = kfs[seg].get("left_orient", "")
                ro = kfs[seg].get("right_orient", "")
                if lo in PRESET_NAMES:
                    self._state["left_orient_idx"] = PRESET_NAMES.index(lo)
                if ro in PRESET_NAMES:
                    self._state["right_orient_idx"] = PRESET_NAMES.index(ro)

        imgui.Separator()

        # Step / Playback
        if len(kfs) > 0 and not rec["playing"] and not rec.get("stepping"):
            next_idx = rec["sel_kf"]
            if next_idx < len(kfs) and imgui.Button("Move to Next"):
                rec["stepping"] = True
                rec["step_target"] = next_idx
                rec["play_t0"] = time.monotonic()
                rec["play_start_joints"] = dict(self._state["joints"])
                print(f"[step] Animating to keyframe {next_idx}")

        if not rec["playing"] and not rec.get("stepping"):
            if len(kfs) > 0 and imgui.Button("Play"):
                rec["playing"] = True
                rec["play_idx"] = 0
                rec["play_t0"] = time.monotonic()
                rec["play_start_joints"] = dict(self._state["joints"])
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
            for n in self._joint_names:
                v0 = src.get(n, 0.0)
                v1 = dst.get(n, 0.0)
                self._state["joints"][n] = v0 + (v1 - v0) * alpha
            self._controller.set_joint_positions(self._state["joints"])
            self._sync_view()
            imgui.Text(f"Step -> KF {rec['step_target']}  {alpha * 100:.0f}%")
            if alpha >= 1.0:
                lo = target_kf.get("left_orient", "")
                ro = target_kf.get("right_orient", "")
                if lo in PRESET_NAMES:
                    self._state["left_orient_idx"] = PRESET_NAMES.index(lo)
                if ro in PRESET_NAMES:
                    self._state["right_orient_idx"] = PRESET_NAMES.index(ro)
                rec["sel_kf"] = min(rec["step_target"] + 1, len(kfs) - 1)
                rec["stepping"] = False
                print(f"[step] Arrived at keyframe {rec['step_target']}")
        else:
            if imgui.Button("Stop"):
                rec["playing"] = False
                print("[play] Playback stopped")
            now = time.monotonic()
            target_idx = rec["play_idx"]
            if target_idx < len(kfs):
                target_kf = kfs[target_idx]
                elapsed = now - rec["play_t0"]
                duration = target_kf["dt"]
                alpha = min(elapsed / max(duration, 1e-6), 1.0)
                src = rec["play_start_joints"]
                dst = target_kf["joints"]
                for n in self._joint_names:
                    v0 = src.get(n, 0.0)
                    v1 = dst.get(n, 0.0)
                    self._state["joints"][n] = v0 + (v1 - v0) * alpha
                self._controller.set_joint_positions(self._state["joints"])
                self._sync_view()
                if "timeline_t" in rec:
                    t_acc = sum(kfs[j]["dt"] for j in range(target_idx))
                    rec["timeline_t"] = t_acc + alpha * duration
                if alpha >= 1.0:
                    rec["play_start_joints"] = dict(target_kf["joints"])
                    lo = target_kf.get("left_orient", "")
                    ro = target_kf.get("right_orient", "")
                    if lo in PRESET_NAMES:
                        self._state["left_orient_idx"] = PRESET_NAMES.index(lo)
                    if ro in PRESET_NAMES:
                        self._state["right_orient_idx"] = PRESET_NAMES.index(ro)
                    rec["play_idx"] = target_idx + 1
                    rec["play_t0"] = now
                    if rec["play_idx"] >= len(kfs):
                        rec["playing"] = False
                        print("[play] Playback finished")
            imgui.Text(f"Playing: {rec['play_idx']}/{len(kfs)}")

        imgui.TreePop()

    def _draw_ref_panel(self) -> None:
        if not imgui.TreeNode("Reference OBJ"):
            return
        c, buf = imgui.InputText("OBJ Path", self._ref["input_buf"])
        if c:
            self._ref["input_buf"] = buf
        if imgui.Button("Load##ref_load"):
            p = self._ref["input_buf"].strip()
            if p and self._load_ref_obj(p):
                self._save_config()
        imgui.SameLine()
        if imgui.Button("Clear All##ref_clear"):
            self._unload_all_refs()
            self._save_config()
            print("[ref] All reference meshes removed")

        entries = self._ref["entries"]
        if len(entries) > 0:
            labels = [f"{i}: {Path(e['path']).name}" for i, e in enumerate(entries)]
            c, sel = imgui.Combo("##ref_sel", self._ref["sel_ref"], labels)
            if c:
                self._ref["sel_ref"] = sel
            si = min(self._ref["sel_ref"], len(entries) - 1)
            entry = entries[si]
            if imgui.Button(f"Remove##{si}"):
                self._unload_ref(si)
                self._ref["sel_ref"] = max(0, min(self._ref["sel_ref"], len(entries) - 1))
                self._save_config()
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
                    self._apply_ref_transform(entry)
                    self._save_config()
                if imgui.Button(f"Reset Transform##{si}"):
                    entry["transform"] = [0.0] * 6
                    self._apply_ref_transform(entry)
                    self._save_config()
        imgui.TreePop()

    def _draw_user_scene_panel(self) -> None:
        if not imgui.TreeNode("User Scene"):
            return
        uobj = self.scene_state.get_objects()
        imgui.Text(f"Objects: {', '.join(sorted(uobj.keys())) if uobj else '(none)'}")
        imgui.Text("(transforms applied via instance transforms)")

        ut = self._user_scene_tf
        tf_changed = False
        for label, idx, speed, lo, hi in [
            ("Scene Tx", 0, 0.001, -10.0, 10.0),
            ("Scene Ty", 1, 0.001, -10.0, 10.0),
            ("Scene Tz", 2, 0.001, -10.0, 10.0),
            ("Scene Rx", 3, 0.5, -360.0, 360.0),
            ("Scene Ry", 4, 0.5, -360.0, 360.0),
            ("Scene Rz", 5, 0.5, -360.0, 360.0),
        ]:
            c, v = imgui.DragFloat(label, ut[idx], speed, lo, hi)
            if c:
                ut[idx] = v
                tf_changed = True
        if imgui.Button("Reset Scene Transform"):
            self._user_scene_tf[:] = [0.0] * 6
            tf_changed = True
        if tf_changed:
            self.scene_state.apply_global_transform(build_scene_transform(self._user_scene_tf))
            if self._sgui is not None:
                self._sgui.update()
            self._save_config()
        imgui.TreePop()

    def _draw_sim_panel(self) -> None:
        if not imgui.TreeNode("Simulation"):
            return
        kf_count = len(self._rec["keyframes"])
        imgui.Text(f"Keyframes: {kf_count}")

        world = self._sim_ctx["world"]

        if not self._sim_live["active"]:
            imgui.Text("Scene initialized. Click to start advancing.")
            if imgui.Button("Start Simulation"):
                self._unload_all_refs()
                sim_dt = self._sim_ctx["sim_dt"]
                fresh_schedule = build_frame_schedule(self._rec["keyframes"], sim_dt, 1)
                self._sim_ctx["schedule"] = fresh_schedule
                self._sim_live["active"] = True
                self._sim_live["total_frames"] = len(fresh_schedule)
                self._sim_live["run"] = False
                print(f"[sim] Started: {len(fresh_schedule)} frames")
        else:
            _sgui = self._sim_ctx["sgui"]
            _sio = self._sim_ctx["sio"]

            if imgui.Button("Play / Pause##sim"):
                self._sim_live["run"] = not self._sim_live["run"]
            imgui.SameLine()
            if imgui.Button("Step Once##sim"):
                self._do_sim_step_ctx()
                self._controller.update_visual_meshes()
                _sgui.update()

            c, spd = imgui.SliderInt("Steps/tick##sim", self._sim_live["steps_per_tick"], 1, 8)
            if c:
                self._sim_live["steps_per_tick"] = int(max(1, spd))

            c, v = imgui.Checkbox("Dump state##sim", self._sim_live["dump_enabled"])
            if c:
                self._sim_live["dump_enabled"] = v
            c, v = imgui.Checkbox("Export surface OBJ##sim", self._sim_live["export_surface"])
            if c:
                self._sim_live["export_surface"] = v

            cur_frame = world.frame()
            tf = self._sim_live["total_frames"]
            imgui.Text(f"Frame: {cur_frame} / {tf}")
            if tf > 0:
                imgui.Text(f"Progress: {min(cur_frame / max(tf, 1), 1.0) * 100:.1f}%")

            if self._sim_live["run"]:
                for _ in range(self._sim_live["steps_per_tick"]):
                    self._do_sim_step_ctx()
                    if world.frame() >= tf and tf > 0:
                        self._sim_live["run"] = False
                        print("[sim] Playback complete")
                        break
                self._controller.update_visual_meshes()
                _sgui.update()
                self._sim_ctx["Timer"].report()

            imgui.Separator()
            c, val = imgui.InputInt("Target Frame##sim", self._sim_live["recover_frame"])
            if c:
                self._sim_live["recover_frame"] = max(0, val)
            if imgui.Button("Recover##sim"):
                target = self._sim_live["recover_frame"]
                if world.recover(target):
                    world.retrieve()
                    self._controller.update_visual_meshes()
                    _sgui.update()
                    print(f"[sim] recovered to frame {target}")
                else:
                    print(f"[sim] recover to frame {target} failed")

            imgui.SameLine()
            if imgui.Button("Replay to##sim"):
                target = self._sim_live["recover_frame"]
                world.recover(0)
                world.retrieve()
                self._sim_live["replaying"] = True
                self._sim_live["replay_target"] = target
                print(f"[sim] replaying from 0 to {target} ...")

            if self._sim_live.get("replaying"):
                replay_target = self._sim_live["replay_target"]
                steps_this_tick = min(self._sim_live["steps_per_tick"], replay_target - world.frame())
                for _ in range(max(1, steps_this_tick)):
                    self._do_sim_step_ctx()
                    if world.frame() >= replay_target:
                        self._sim_live["replaying"] = False
                        print(f"[sim] replay reached frame {world.frame()}")
                        break
                self._controller.update_visual_meshes()
                _sgui.update()
                imgui.Text(f"Replaying: {world.frame()} / {replay_target}")

            imgui.Separator()
            if imgui.Button("Rebuild Schedule##sim"):
                sim_dt = self._sim_ctx["sim_dt"]
                fresh = build_frame_schedule(self._rec["keyframes"], sim_dt, 1)
                self._sim_ctx["schedule"] = fresh
                self._sim_live["total_frames"] = len(fresh)
                print(f"[sim] Rebuilt: {len(self._rec['keyframes'])} kfs -> {len(fresh)} frames")

        imgui.TreePop()

    def _do_sim_step_ctx(self) -> None:
        """Advance one simulation step (inspector context)."""
        world = self._sim_ctx["world"]
        world.advance()
        world.retrieve()
        if self._sim_live["dump_enabled"]:
            world.dump()
        if self._sim_live["export_surface"]:
            od = self._sim_ctx["output_dir"]
            self._sim_ctx["sio"].write_surface(f"{od}/surface_{world.frame()}.obj")

    # ------------------------------------------------------------------
    # run_inspector  (was main())
    # ------------------------------------------------------------------

    def run_inspector(self) -> None:
        """FK/IK inspector with fully initialized UIPC scene via SceneGUI."""
        from asset_dir import AssetDir
        from uipc import SceneIO, Timer, view
        from uipc.gui import SceneGUI

        output_dir = Path(AssetDir.output_path(self._entry_file))
        controller = self._init_controller()
        self._load_initial_state()
        controller.set_joint_positions(self._state["joints"])

        engine, world, scene = self._create_engine_scene(output_dir)
        self._setup_robot(scene)
        self._apply_scene_and_stitch(scene, world)

        sim_dt = float(view(scene.config().find("dt"))[0])
        print(f"[urdf-gui] sim_dt = {sim_dt}")

        keyframes = load_keyframes(self._record_file)
        schedule = build_frame_schedule(keyframes, sim_dt, 1)

        self._sim_ctx = {
            "engine": engine,
            "world": world,
            "scene": scene,
            "schedule": schedule,
            "output_dir": output_dir,
            "controller": controller,
            "sim_dt": sim_dt,
            "Timer": Timer,
            "SceneIO": SceneIO,
            "SceneGUI": SceneGUI,
        }

        def on_frame(info: Any, ctrl: URDFController) -> None:
            s = self._sim_ctx["schedule"]
            frame = int(info.frame())
            if frame < len(s):
                ctrl.set_joint_positions(s[frame]["joints"])

        controller.bind_animator(scene, on_frame=on_frame)

        world.init(scene)
        world.retrieve()

        # Polyscope
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("none")

        sgui = SceneGUI(scene, "split")
        sio = SceneIO(scene)
        sgui.register()
        sgui.set_edge_width(1.0)
        self._sgui = sgui
        self._sio = sio
        self._sim_ctx["sgui"] = sgui
        self._sim_ctx["sio"] = sio

        controller.register_visual_only_meshes(transparency=0.45)
        controller.update_visual_meshes()

        # Stitch line visualisation
        self._stitch_net = None
        stitch_result = self.scene_state.build_stitch_line_nodes()
        if stitch_result is not None:
            sn, se = stitch_result
            self._stitch_net = ps.register_curve_network("stitch_line", sn, se, radius=0.001, color=(1.0, 0.2, 0.2))

        # Identify left / right grippers
        ee = self._end_effectors
        self._left_ee = next((e for e in ee if "_L" in e or "left" in e.lower()), ee[0])
        self._right_ee = next((e for e in ee if "_R" in e or "right" in e.lower()), ee[-1])
        print(f"[urdf-gui] Left gripper: {self._left_ee}  |  Right gripper: {self._right_ee}")

        # Reference OBJ state
        self._ref = {"entries": [], "input_buf": "", "sel_ref": 0}
        if self._config_file.exists():
            try:
                cfg = json.loads(self._config_file.read_text(encoding="utf-8"))
                for item in cfg.get("ref_objs", []):
                    self._load_ref_obj(item["path"], item.get("transform"))
                for p in cfg.get("ref_obj_paths", []):
                    if not any(e["path"] == p for e in self._ref["entries"]):
                        self._load_ref_obj(p)
            except (json.JSONDecodeError, KeyError):
                pass

        atexit.register(self._save_config)

        self._sim_live = {
            "active": False,
            "run": False,
            "steps_per_tick": 1,
            "dump_enabled": True,
            "export_surface": True,
            "export_npy": True,
            "recover_frame": 0,
            "total_frames": 0,
        }

        self._sync_view()

        def on_update() -> None:
            imgui.Text("=== URDF Controller Inspector ===")
            imgui.Separator()
            self._draw_joint_panel()
            imgui.Separator()
            self._draw_ee_panel()
            imgui.Separator()
            self._draw_ik_panel()
            imgui.Separator()
            self._draw_recording_panel()
            imgui.Separator()
            self._draw_ref_panel()
            imgui.Separator()
            self._draw_user_scene_panel()
            imgui.Separator()
            self._draw_sim_panel()

        ps.set_user_callback(on_update)
        ps.show()

    # ------------------------------------------------------------------
    # run_simulation  (was run_simulation())
    # ------------------------------------------------------------------

    def run_simulation(self, recover_frame: int = 0) -> None:
        """Launch UIPC simulation directly (``--sim`` CLI mode)."""
        from asset_dir import AssetDir
        from uipc import SceneIO, Timer, view
        from uipc.gui import SceneGUI

        output_dir = Path(AssetDir.output_path(self._entry_file))
        controller = self._init_controller()
        joint_names = self._joint_names

        # Load initial joints
        if self._joint_file.exists():
            try:
                loaded = json.loads(self._joint_file.read_text(encoding="utf-8"))
                jd = loaded.get("joints", loaded) if isinstance(loaded, dict) else loaded
                init_joints = {n: jd.get(n, 0.0) for n in joint_names}
            except (json.JSONDecodeError, KeyError):
                init_joints = {n: 0.0 for n in joint_names}
        else:
            init_joints = {n: 0.0 for n in joint_names}

        sim_keyframes = load_keyframes(self._record_file)
        if sim_keyframes:
            kf0 = sim_keyframes[0]
            for n in joint_names:
                if n in kf0["joints"]:
                    init_joints[n] = kf0["joints"][n]
        controller.set_joint_positions(init_joints)

        engine, world, scene = self._create_engine_scene(output_dir)
        self._setup_robot(scene)
        self._apply_scene_and_stitch(scene, world)

        sim_dt = float(view(scene.config().find("dt"))[0])
        schedule = build_frame_schedule(sim_keyframes, sim_dt, 1)
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
        sim_stitch_result = self.scene_state.build_stitch_line_nodes()
        if sim_stitch_result is not None:
            sn, se = sim_stitch_result
            sim_stitch_net = ps.register_curve_network("stitch_line", sn, se, radius=0.001, color=(1.0, 0.2, 0.2))

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

        def _update_stitch_line() -> None:
            if sim_stitch_net is not None:
                result = self.scene_state.build_stitch_line_nodes()
                if result is not None:
                    sim_stitch_net.update_node_positions(result[0])

        def _rebuild_schedule() -> None:
            nonlocal schedule, total_frames, sim_keyframes
            sim_keyframes = load_keyframes(self._record_file)
            schedule = build_frame_schedule(sim_keyframes, sim_dt, 1)
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

            imgui.Separator()
            imgui.Text("Re-record (appends from current pose)")
            c, dt = imgui.SliderFloat("New KF dt##sim_rec", sim_state["rec_interp_time"], 0.1, 5.0)
            if c:
                sim_state["rec_interp_time"] = dt

            if imgui.Button("Record Current Pose##sim_rec"):
                from uipc import view as _view

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
                with open(self._record_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"[sim-rec] Appended keyframe #{len(sim_keyframes)} (dt={kf['dt']:.2f}s)")

            imgui.SameLine()
            if imgui.Button("Rebuild Schedule##sim_rec"):
                _rebuild_schedule()

        on_sim_update._refs = (engine, world, scene, sgui, sio, controller)
        ps.set_user_callback(on_sim_update)
        ps.show()

    # ------------------------------------------------------------------
    # run_export_recover
    # ------------------------------------------------------------------

    def run_export_recover(self, max_frame: int, frame_skip: int = 10) -> None:
        """No-GUI mode: recover frames, export NPY."""
        from uipc import view

        from asset_dir import AssetDir

        output_dir = Path(AssetDir.output_path(self._entry_file))
        seq_dir = self._seq_dir

        controller = self._init_controller()
        joint_names = self._joint_names

        if self._joint_file.exists():
            try:
                loaded = json.loads(self._joint_file.read_text(encoding="utf-8"))
                jd = loaded.get("joints", loaded) if isinstance(loaded, dict) else loaded
                init_joints = {n: jd.get(n, 0.0) for n in joint_names}
            except (json.JSONDecodeError, KeyError):
                init_joints = {n: 0.0 for n in joint_names}
        else:
            init_joints = {n: 0.0 for n in joint_names}

        sim_keyframes = load_keyframes(self._record_file)
        if sim_keyframes:
            for n in joint_names:
                if n in sim_keyframes[0]["joints"]:
                    init_joints[n] = sim_keyframes[0]["joints"][n]
        controller.set_joint_positions(init_joints)

        engine, world, scene = self._create_engine_scene(output_dir)
        self._setup_robot(scene)
        self._apply_scene_and_stitch(scene, world)

        sim_dt_val = float(view(scene.config().find("dt"))[0])
        repo_root = Path(self._entry_file).resolve().parents[3]
        urdf_rel = str(self._urdf_path.relative_to(repo_root))

        schedule = build_frame_schedule(sim_keyframes, sim_dt_val, 1)
        print(f"[export-recover] {len(sim_keyframes)} keyframes -> {len(schedule)} scheduled frames")

        def _joints_at_frame(f: int) -> dict[str, float]:
            if f < len(schedule):
                return schedule[f]["joints"]
            if schedule:
                return schedule[-1]["joints"]
            return {n: 0.0 for n in joint_names}

        exporter = SequenceExporter(joint_names, self.scene_state, dt=sim_dt_val, urdf_rel=urdf_rel)

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
        print(f"[export-recover] Done: {exported} frames (skip={frame_skip})")

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def main_cli(self) -> None:
        """Parse command-line args and dispatch to the appropriate mode."""
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
        parser.add_argument("--frame-skip", type=int, default=10, help="Export every N-th frame (default 10).")
        parser.add_argument("--seq-dir", type=str, default=None, help="Override sequence export directory.")
        args = parser.parse_args()

        if args.export_recover >= 0:
            if args.seq_dir:
                self._seq_dir = Path(args.seq_dir)
            self.run_export_recover(args.export_recover, frame_skip=max(1, args.frame_skip))
        elif args.sim:
            self.run_simulation(recover_frame=args.recover)
        else:
            self.run_inspector()
