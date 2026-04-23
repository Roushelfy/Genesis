"""replay_traj_polyscope.py - Two-phase cooking-scene viewer.

Phase A (Calibration, "SceneA"):
    Show the marvin_gss robot driven by trajectory.npz qpos, with the pan and
    spatula posed by trajectory.npz rigid transforms.  Use the GUI to fine-tune
    finger joints (and optional small wrist offsets) so the hand visually
    grasps the tool handle.  Press "Capture" to record:
        - finger joint angles (right + left)
        - T_palm_R_rel_pan  = inv(T_pan_world)  @ T_palm_R_world
        - T_palm_L_rel_spat = inv(T_spat_world) @ T_palm_L_world
    These are saved to output/calib.json and re-loaded on startup.

Phase B (Replay, "SceneB"):
    Show the IPC simulation result loaded from seq_v1 (pan, spatula, broccoli,
    tomato, mushroom, noodles).  For each frame, compute the desired palm
    world transform from the simulated tool transform and the captured
    relative offset, then solve 6-DOF IK on the arm to bring the wrist there.
    Finger joints are taken directly from the calibration.

Switch between phases with the toggle button at the top of the GUI panel.

Data sources:
    DemoAssets/cook_with_teleop/trajectory.npz      (Phase A)
    DemoAssets/cook_with_teleop/seq_v1/             (Phase B)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

# -- Paths --------------------------------------------------------------------
_HERE      = Path(__file__).resolve()
_DEMO_ROOT = _HERE.parents[3]
_GSS_URDF  = _DEMO_ROOT / "DemoAssets" / "marvin_gss" / "marvin_gss.urdf"
_TRAJ_NPZ  = _HERE.parents[1] / "trajectory.npz"
_SEQ_DIR   = _HERE.parents[1] / "seq_v1"
_OUTPUT    = _HERE.parents[1] / "output"
_CALIB     = _OUTPUT / "calib.json"

sys.path.insert(0, str(_DEMO_ROOT / "DemoAssets" / "yoyo" / "scripts"))
sys.path.insert(0, str(_HERE.parent))   # (kept for future helpers)
from urdf_controller import URDFController  # noqa: E402

# Robot base (matches examples/IPC_Solver/replay_cooking_traj.py)
_ROBOT_BASE_POS = (0.0, 0.0, 1.08)

# Hand link / arm joint names ---------------------------------------------------
_RIGHT_PALM = "right_palm_link"
_LEFT_PALM  = "left_palm_link"
_RIGHT_ARM_JOINTS = [f"Joint{i}_R" for i in range(1, 8)]
_LEFT_ARM_JOINTS  = [f"Joint{i}_L" for i in range(1, 8)]

# Finger joint name pattern: <side>_finger<fi>_joint<ji>  (fi=1..5, ji=1..4)
_FINGERS = list(range(1, 6))
_KNUCKLES = list(range(1, 5))

def _finger_joint(side: str, fi: int, ji: int) -> str:
    return f"{side}_finger{fi}_joint{ji}"

_RIGHT_FINGER_JOINTS = [_finger_joint("right", fi, ji)
                        for fi in _FINGERS for ji in _KNUCKLES]
_LEFT_FINGER_JOINTS  = [_finger_joint("left",  fi, ji)
                        for fi in _FINGERS for ji in _KNUCKLES]

def _fingertip_link(side: str, fi: int) -> str:
    return f"{side}_finger{fi}_tip_link"

# -- Geometry helpers for gizmo "handle" widgets ------------------------------

def _cube_mesh(half_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Small axis-aligned cube centred at origin for use as a gizmo handle."""
    h = float(half_size)
    V = np.array([
        [-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],
        [-h,-h, h],[h,-h, h],[h,h, h],[-h,h, h],
    ], dtype=np.float64)
    F = np.array([
        [0,1,2],[0,2,3],  [4,6,5],[4,7,6],
        [0,4,5],[0,5,1],  [2,6,7],[2,7,3],
        [1,5,6],[1,6,2],  [0,3,7],[0,7,4],
    ], dtype=np.int32)
    return V, F


# -- Genesis qpos order -------------------------------------------------------
def _build_genesis_qpos_order() -> list[str]:
    names: list[str] = []
    for k in range(1, 8):
        names.append(f"Joint{k}_R")
        names.append(f"Joint{k}_L")
    for k in range(1, 5):
        for f in range(1, 6):
            names.append(f"right_finger{f}_joint{k}")
        for f in range(1, 6):
            names.append(f"left_finger{f}_joint{k}")
    return names

GENESIS_QPOS_ORDER = _build_genesis_qpos_order()
assert len(GENESIS_QPOS_ORDER) == 54


# -- Maths --------------------------------------------------------------------

def _transform_verts(verts: np.ndarray, tf: np.ndarray) -> np.ndarray:
    return (tf[:3, :3] @ verts.T).T + tf[:3, 3]


def _xyzq_to_4x4(xyzq: np.ndarray) -> np.ndarray:
    """Genesis [x,y,z, qw,qx,qy,qz] -> 4x4."""
    x, y, z, qw, qx, qy, qz = (float(v) for v in xyzq)
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    M[0, 1] =     2 * (qx * qy - qz * qw)
    M[0, 2] =     2 * (qx * qz + qy * qw)
    M[1, 0] =     2 * (qx * qy + qz * qw)
    M[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    M[1, 2] =     2 * (qy * qz - qx * qw)
    M[2, 0] =     2 * (qx * qz - qy * qw)
    M[2, 1] =     2 * (qy * qz + qx * qw)
    M[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    M[:3, 3] = (x, y, z)
    return M


def _euler_xyz_deg_to_mat(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx


def _delta_4x4(dxyz: list[float], drpy_deg: list[float]) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = _euler_xyz_deg_to_mat(*drpy_deg)
    M[:3,  3] = dxyz
    return M


# -- Minimal OBJ reader (faces + line edges) ----------------------------------

def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    verts, faces, edges = [], [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("v "):
                verts.append([float(x) for x in raw.split()[1:4]])
            elif raw.startswith("f "):
                toks = [int(t.split("/")[0]) - 1 for t in raw.split()[1:]]
                for k in range(1, len(toks) - 1):
                    faces.append([toks[0], toks[k], toks[k + 1]])
            elif raw.startswith("l "):
                idx = [int(x) - 1 for x in raw.split()[1:]]
                for a, b in zip(idx, idx[1:]):
                    edges.append([a, b])
    return (
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int32) if faces else None,
        np.asarray(edges, dtype=np.int32) if edges else None,
    )


# -- Robot helpers -------------------------------------------------------------

_ROBOT_COLOR_NORMAL = (0.85, 0.85, 0.88)
_ROBOT_COLOR_LOCKED = (0.95, 0.30, 0.30)

def _register_robot(ctrl: URDFController) -> dict[str, ps.SurfaceMesh]:
    handles: dict[str, ps.SurfaceMesh] = {}
    tfs = ctrl.get_mesh_transforms()
    for node in ctrl.mesh_nodes:
        v = _transform_verts(node.local_vertices, tfs[node.node_name])
        sm = ps.register_surface_mesh(
            f"robot/{node.node_name}", v, node.faces, smooth_shade=True
        )
        sm.set_color(_ROBOT_COLOR_NORMAL)
        handles[node.node_name] = sm
    return handles


def _update_robot(ctrl: URDFController, handles: dict[str, ps.SurfaceMesh]) -> None:
    tfs = ctrl.get_mesh_transforms()
    for node in ctrl.mesh_nodes:
        handles[node.node_name].update_vertex_positions(
            _transform_verts(node.local_vertices, tfs[node.node_name])
        )


def _apply_qpos(ctrl: URDFController, qpos: np.ndarray,
                qpos_to_jname: list[str]) -> None:
    actuated = set(ctrl.joint_names)
    joints = {jname: float(qpos[i]) for i, jname in enumerate(qpos_to_jname)
              if jname in actuated}
    ctrl.set_joint_positions(joints)


# -- Sequence loader (Phase B) ------------------------------------------------

_RIGID_COLORS = {"pan": (0.60, 0.60, 0.65), "spatula": (0.50, 0.50, 0.55)}
_BROC_COLOR    = (0.20, 0.60, 0.15)
_TOMATO_COLOR  = (0.85, 0.15, 0.10)
_MUSHROOM_COLOR = (0.85, 0.70, 0.55)
_NOODLE_COLOR  = (0.92, 0.86, 0.55)


class SeqPlayer:
    """Loads meta.json + per-object data from seq_v1/."""

    def __init__(self, seq_dir: Path) -> None:
        self.seq_dir = seq_dir
        self.meta = json.loads((seq_dir / "meta.json").read_text(encoding="utf-8"))
        self.n_frames = int(self.meta["frame_count"])
        self.dt = float(self.meta.get("dt", 0.005))
        self.rigid: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.fem:   dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.rod:   dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for name, info in self.meta["objects"].items():
            npy = seq_dir / info["data"]
            mesh = seq_dir / name / "mesh.obj"
            if not (npy.exists() and mesh.exists()):
                print(f"  [SEQ][warn] missing data for {name}, skipping")
                continue
            arr = np.load(str(npy))
            v, f, e = _read_obj(mesh)
            t = info["type"]
            if t == "rigid":
                self.rigid[name] = (v, f if f is not None else np.zeros((0,3),np.int32),
                                    arr.astype(np.float64))
            elif t == "fem":
                self.fem[name] = (f if f is not None else np.zeros((0,3),np.int32),
                                  arr.astype(np.float64))
            elif t == "rod":
                self.rod[name] = (e if e is not None else np.zeros((0,2),np.int32),
                                  arr.astype(np.float64))


# -- Calibration data ---------------------------------------------------------

class Calibration:
    """Hand-to-tool relative transforms + finger joint angles + robot base pose."""

    def __init__(self) -> None:
        self.T_palm_R_rel_pan:  np.ndarray | None = None   # (4,4)
        self.T_palm_L_rel_spat: np.ndarray | None = None   # (4,4)
        self.finger_angles_R: dict[str, float] = {}
        self.finger_angles_L: dict[str, float] = {}
        # Robot base pose (optional).  If set, the script restores this on
        # startup so the user's chosen base XYZ + RPY survives across runs.
        self.base_pos:     list[float] | None = None   # [x, y, z] in metres
        self.base_rpy_deg: list[float] | None = None   # [roll, pitch, yaw] in deg

    @property
    def is_complete(self) -> bool:
        return (self.T_palm_R_rel_pan is not None
                and self.T_palm_L_rel_spat is not None)

    def to_dict(self) -> dict:
        return {
            "T_palm_R_rel_pan":  None if self.T_palm_R_rel_pan  is None
                                  else self.T_palm_R_rel_pan.tolist(),
            "T_palm_L_rel_spat": None if self.T_palm_L_rel_spat is None
                                  else self.T_palm_L_rel_spat.tolist(),
            "finger_angles_R": self.finger_angles_R,
            "finger_angles_L": self.finger_angles_L,
            "base_pos":        self.base_pos,
            "base_rpy_deg":    self.base_rpy_deg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        c = cls()
        if d.get("T_palm_R_rel_pan")  is not None:
            c.T_palm_R_rel_pan  = np.asarray(d["T_palm_R_rel_pan"],  np.float64)
        if d.get("T_palm_L_rel_spat") is not None:
            c.T_palm_L_rel_spat = np.asarray(d["T_palm_L_rel_spat"], np.float64)
        c.finger_angles_R = dict(d.get("finger_angles_R", {}))
        c.finger_angles_L = dict(d.get("finger_angles_L", {}))
        bp = d.get("base_pos")
        c.base_pos = [float(v) for v in bp] if bp is not None else None
        br = d.get("base_rpy_deg")
        c.base_rpy_deg = [float(v) for v in br] if br is not None else None
        return c

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        print(f"  [CALIB] Saved -> {path}")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [CALIB] Failed to load {path}: {e}")
            return cls()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    # ----- Robot ------------------------------------------------------------
    print(f"Loading URDF : {_GSS_URDF.name}")
    ctrl = URDFController(str(_GSS_URDF), mesh_source="visual")
    print(f"  {len(ctrl.joint_names)} actuated joints, {len(ctrl.mesh_nodes)} mesh nodes")
    def _set_base_pose(x: float, y: float, z: float,
                       rx_deg: float, ry_deg: float, rz_deg: float) -> None:
        """Place the robot's root transform at (x,y,z) with XYZ-Euler rotation."""
        T = np.eye(4)
        T[:3, :3] = _euler_xyz_deg_to_mat(rx_deg, ry_deg, rz_deg)
        T[:3, 3]  = (float(x), float(y), float(z))
        ctrl.set_root_transform(T)

    _set_base_pose(_ROBOT_BASE_POS[0], _ROBOT_BASE_POS[1], _ROBOT_BASE_POS[2],
                   0.0, 0.0, 0.0)

    # ----- Phase A data: trajectory.npz ------------------------------------
    print(f"\nLoading trajectory: {_TRAJ_NPZ.name}")
    traj = np.load(str(_TRAJ_NPZ))
    qpos      = traj["robot_qpos"].astype(np.float64)
    pan_xyzq  = traj["rigid_pan"].astype(np.float64)
    spat_xyzq = traj["rigid_spatula"].astype(np.float64)
    n_traj = len(qpos)
    print(f"  {n_traj} frames")

    # ----- Phase B data: scan all available seq*/ subdirs ----------------
    seq_root = _HERE.parents[1]
    seq_options = sorted(
        p for p in seq_root.iterdir()
        if p.is_dir() and p.name.startswith("seq") and (p / "meta.json").exists()
    )
    if not seq_options:
        raise RuntimeError(f"No seq*/ directories with meta.json found in {seq_root}")
    print(f"\n[SCENES] Available: {[p.name for p in seq_options]}")
    # Use _SEQ_DIR if it's in the list, else the first option.
    initial_seq_dir = _SEQ_DIR if _SEQ_DIR in seq_options else seq_options[0]
    print(f"Loading scene: {initial_seq_dir.name}")
    seq = SeqPlayer(initial_seq_dir)
    print(f"  {seq.n_frames} frames | rigid={len(seq.rigid)}  "
          f"fem={len(seq.fem)}  rod={len(seq.rod)}")
    if "pan" not in seq.rigid or "spatula" not in seq.rigid:
        raise RuntimeError(f"{initial_seq_dir.name} must contain rigid 'pan' and 'spatula'")

    # ----- Calibration ------------------------------------------------------
    calib = Calibration.load(_CALIB)
    if calib.is_complete:
        print(f"  [CALIB] Loaded existing calibration from {_CALIB.name}")
    else:
        print(f"  [CALIB] No existing calibration; record one in Phase A.")

    # ----- Polyscope --------------------------------------------------------
    ps.init()
    ps.set_up_dir("z_up"); ps.set_front_dir("x_front")
    ps.set_window_size(1600, 1000)
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height(0.0)
    ps.set_automatically_compute_scene_extents(True)

    # Initial pose for registration (use Phase A frame 0)
    _apply_qpos(ctrl, qpos[0], GENESIS_QPOS_ORDER)
    robot_handles = _register_robot(ctrl)

    # ----- Phase A meshes (pan/spatula from trajectory) ---------------------
    pan_v_local = np.asarray(seq.rigid["pan"][0],  np.float64)   # local-frame mesh
    spat_v_local = np.asarray(seq.rigid["spatula"][0], np.float64)
    pan_f  = seq.rigid["pan"][1]
    spat_f = seq.rigid["spatula"][1]
    pan_tf0_traj  = _xyzq_to_4x4(pan_xyzq[0])
    spat_tf0_traj = _xyzq_to_4x4(spat_xyzq[0])
    ps_pan_A  = ps.register_surface_mesh("phaseA/pan",
                  _transform_verts(pan_v_local, pan_tf0_traj), pan_f, smooth_shade=True)
    ps_pan_A.set_color(_RIGID_COLORS["pan"])
    ps_spat_A = ps.register_surface_mesh("phaseA/spatula",
                  _transform_verts(spat_v_local, spat_tf0_traj), spat_f, smooth_shade=True)
    ps_spat_A.set_color(_RIGID_COLORS["spatula"])

    # ----- Phase B meshes (pan/spatula + food from seq_v1) ------------------
    rigid_handles_B: dict[str, ps.SurfaceMesh] = {}
    for name, (v_local, faces, tfs) in seq.rigid.items():
        if faces is None or len(faces) == 0:
            continue
        color = _RIGID_COLORS.get(name) or (_BROC_COLOR if name.startswith("broc")
                                            else (0.7, 0.7, 0.7))
        sm = ps.register_surface_mesh(f"phaseB/{name}",
                _transform_verts(v_local, tfs[0]), faces, smooth_shade=True)
        sm.set_color(color)
        sm.set_enabled(False)
        rigid_handles_B[name] = sm

    fem_handles_B: dict[str, ps.SurfaceMesh] = {}
    for name, (faces, positions) in seq.fem.items():
        if len(faces) == 0:
            continue
        color = (_TOMATO_COLOR if name.startswith("tomato")
                 else _MUSHROOM_COLOR if name.startswith("mushroom")
                 else (0.7, 0.7, 0.7))
        sm = ps.register_surface_mesh(f"phaseB/{name}", positions[0], faces,
                                       smooth_shade=True)
        sm.set_color(color)
        sm.set_enabled(False)
        fem_handles_B[name] = sm

    rod_handles_B: dict[str, ps.CurveNetwork] = {}
    for name, (edges, positions) in seq.rod.items():
        if len(edges) == 0:
            continue
        cn = ps.register_curve_network(f"phaseB/{name}", positions[0], edges)
        cn.set_color(_NOODLE_COLOR)
        cn.set_radius(0.0018, relative=False)
        cn.set_enabled(False)
        rod_handles_B[name] = cn

    current_seq_dir = [initial_seq_dir]   # mutable holder for the active dir

    def _reload_scene(new_dir: Path) -> None:
        """Tear down the existing Phase-B meshes, load a different SeqPlayer
        from *new_dir*, and re-register the meshes.  All closures see the new
        SeqPlayer because they read the (rebound) ``seq`` cell."""
        nonlocal seq, pan_v_local, spat_v_local, pan_f, spat_f
        for sm in list(rigid_handles_B.values()): sm.remove()
        for sm in list(fem_handles_B.values()):   sm.remove()
        for cn in list(rod_handles_B.values()):   cn.remove()
        rigid_handles_B.clear()
        fem_handles_B.clear()
        rod_handles_B.clear()

        seq = SeqPlayer(new_dir)
        if "pan" not in seq.rigid or "spatula" not in seq.rigid:
            raise RuntimeError(
                f"{new_dir.name} must contain rigid 'pan' and 'spatula'")
        current_seq_dir[0] = new_dir
        print(f"\n[SCENE] Switched to {new_dir.name}: {seq.n_frames} frames | "
              f"rigid={len(seq.rigid)} fem={len(seq.fem)} rod={len(seq.rod)}",
              flush=True)

        # Refresh rest-pose vertices/faces used by Phase A's pan/spatula tools.
        pan_v_local  = np.asarray(seq.rigid["pan"][0],  np.float64)
        spat_v_local = np.asarray(seq.rigid["spatula"][0], np.float64)
        pan_f        = seq.rigid["pan"][1]
        spat_f       = seq.rigid["spatula"][1]
        ps_pan_A.update_vertex_positions(_transform_verts(pan_v_local, pan_tf0_traj))
        ps_spat_A.update_vertex_positions(_transform_verts(spat_v_local, spat_tf0_traj))

        in_B = state["phase"] == "B"
        for name, (v_local, faces, tfs) in seq.rigid.items():
            if faces is None or len(faces) == 0:
                continue
            color = _RIGID_COLORS.get(name) or (_BROC_COLOR if name.startswith("broc")
                                                else (0.7, 0.7, 0.7))
            sm = ps.register_surface_mesh(f"phaseB/{name}",
                    _transform_verts(v_local, tfs[0]), faces, smooth_shade=True)
            sm.set_color(color)
            sm.set_enabled(in_B)
            rigid_handles_B[name] = sm
        for name, (faces, positions) in seq.fem.items():
            if len(faces) == 0:
                continue
            color = (_TOMATO_COLOR if name.startswith("tomato")
                     else _MUSHROOM_COLOR if name.startswith("mushroom")
                     else (0.7, 0.7, 0.7))
            sm = ps.register_surface_mesh(f"phaseB/{name}", positions[0], faces,
                                           smooth_shade=True)
            sm.set_color(color)
            sm.set_enabled(in_B)
            fem_handles_B[name] = sm
        for name, (edges, positions) in seq.rod.items():
            if len(edges) == 0:
                continue
            cn = ps.register_curve_network(f"phaseB/{name}", positions[0], edges)
            cn.set_color(_NOODLE_COLOR)
            cn.set_radius(0.0018, relative=False)
            cn.set_enabled(in_B)
            rod_handles_B[name] = cn

        # Reset frame index and refresh the visible phase.
        state["frame_B"] = 0
        if in_B:
            update_phase_B(0)
        else:
            update_phase_A(state["frame_A"])

    # ----- Gizmo handles (6-DOF for palms, translate-only for fingertips) ---
    _palm_cube_V, _palm_cube_F = _cube_mesh(0.025)     # 2.5cm cube for palm
    _tip_cube_V,  _tip_cube_F  = _cube_mesh(0.008)     # 0.8cm cube for fingertips

    palm_gizmos: dict[str, ps.SurfaceMesh] = {}
    for arm, link, color in (("R", _RIGHT_PALM, (1.0, 0.9, 0.2)),
                              ("L", _LEFT_PALM,  (0.2, 0.9, 1.0))):
        sm = ps.register_surface_mesh(
            f"gizmo/palm_{arm}", _palm_cube_V, _palm_cube_F, smooth_shade=False)
        sm.set_color(color)
        sm.set_transparency(0.6)
        sm.set_transform(ctrl.get_link_transform(link))
        # Gizmo starts hidden; we'll enable it only when the user selects the
        # corresponding handle mesh.
        sm.set_transform_gizmo_enabled(False)
        g = sm.get_transformation_gizmo()
        g.set_allow_translation(True)
        g.set_allow_rotation(True)
        g.set_allow_scaling(False)
        g.set_interact_in_local_space(False)
        palm_gizmos[arm] = sm

    tip_gizmos: dict[tuple[str, int], ps.SurfaceMesh] = {}
    for arm in ("R", "L"):
        side = "right" if arm == "R" else "left"
        for fi in _FINGERS:
            tip_link = _fingertip_link(side, fi)
            color = ((1.0, 0.6, 0.2) if arm == "R"
                     else (0.2, 0.6, 1.0))
            sm = ps.register_surface_mesh(
                f"gizmo/tip_{arm}{fi}", _tip_cube_V, _tip_cube_F, smooth_shade=False)
            sm.set_color(color)
            sm.set_transparency(0.75)
            T_init = np.eye(4)
            T_init[:3, 3] = ctrl.get_link_transform(tip_link)[:3, 3]
            sm.set_transform(T_init)
            sm.set_transform_gizmo_enabled(False)   # see above
            g = sm.get_transformation_gizmo()
            g.set_allow_translation(True)
            g.set_allow_rotation(False)
            g.set_allow_scaling(False)
            g.set_interact_in_local_space(False)
            tip_gizmos[(arm, fi)] = sm

    # Map from Polyscope structure name -> gizmo identifier.
    gizmo_lookup: dict[str, tuple[str, object]] = {}
    for arm in ("R", "L"):
        gizmo_lookup[palm_gizmos[arm].get_name()] = ("palm", arm)
    for arm in ("R", "L"):
        for fi in _FINGERS:
            gizmo_lookup[tip_gizmos[(arm, fi)].get_name()] = ("tip", (arm, fi))

    # ----- State ------------------------------------------------------------
    state = {
        "phase": "A",                       # "A" or "B"
        "frame_A": 0,
        "frame_B": 0,
        "playing": False,
        "speed": 1.0,
        "_acc": 0.0,
        "_first_render": True,
        # Show/hide all gizmo handle meshes (whether they're clickable).
        "show_palm_gizmos": True,
        "show_tip_gizmos":  True,
        # IK application toggle.  When disabled, gizmo drags only move the
        # handle mesh (for debugging the gizmo + IK separately).
        "apply_ik_on_drag": True,
        # Phase-A behavior: when a calibration is loaded, also run IK so the
        # hand tracks the tool (keeps the grasp visually consistent across
        # phase switches).  Uncheck to see the raw teleop pose.
        "follow_pan_in_A": True,
        # Robot base pose (XYZ + Euler RPY in degrees).  Defaults match
        # replay_cooking_traj.py's `pos=(0, 0, 1.08)`.  If the loaded
        # calibration has a saved base pose we use that instead so the user's
        # previously-chosen base survives across runs.
        "base_x":     float(calib.base_pos[0] if calib.base_pos else _ROBOT_BASE_POS[0]),
        "base_y":     float(calib.base_pos[1] if calib.base_pos else _ROBOT_BASE_POS[1]),
        "base_z":     float(calib.base_pos[2] if calib.base_pos else _ROBOT_BASE_POS[2]),
        "base_rx_deg": float(calib.base_rpy_deg[0] if calib.base_rpy_deg else 0.0),
        "base_ry_deg": float(calib.base_rpy_deg[1] if calib.base_rpy_deg else 0.0),
        "base_rz_deg": float(calib.base_rpy_deg[2] if calib.base_rpy_deg else 0.0),
        # Arm joints that IK should NOT modify.  When a joint is locked it
        # keeps the value set by `_apply_qpos` (i.e. the teleop value).
        # Useful for joints the human anatomy doesn't have (e.g. forearm twist).
        "locked_arm_joints": set(),
        "lock_panel_open":  True,
        # Name of the currently selected gizmo handle (or None).
        "sel_gizmo_name":  None,
        # Last-known transform for each gizmo (used to detect drags)
        "last_palm_R": np.eye(4),
        "last_palm_L": np.eye(4),
    }
    for arm in ("R", "L"):
        for fi in _FINGERS:
            state[f"last_tip_{arm}{fi}"] = np.eye(4)

    # Apply the (possibly calib-loaded) base pose from state to the controller.
    _set_base_pose(state["base_x"], state["base_y"], state["base_z"],
                   state["base_rx_deg"], state["base_ry_deg"],
                   state["base_rz_deg"])

    _FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
    snap_dir = _OUTPUT
    snap_dir.mkdir(parents=True, exist_ok=True)

    def _active_arm_joints(arm: str) -> list[str]:
        """Return the arm joints that IK is allowed to modify (skip locked)."""
        base = _RIGHT_ARM_JOINTS if arm == "R" else _LEFT_ARM_JOINTS
        locked = state["locked_arm_joints"]
        return [j for j in base if j not in locked]

    # ---- Lock-color visualisation ------------------------------------------
    # Build link -> [mesh_node_names] map from the URDFController's mesh
    # nodes, so we can recolour every mesh attached to a "locked" link.
    link_to_meshes: dict[str, list[str]] = {}
    for node in ctrl.mesh_nodes:
        link_to_meshes.setdefault(node.parent_link, []).append(node.node_name)

    def _apply_lock_colors() -> None:
        """Re-tint robot meshes: red for links touching a locked joint."""
        locked_links: set[str] = set()
        for jname in state["locked_arm_joints"]:
            info = ctrl._joints.get(jname)
            if info is None:
                continue
            locked_links.add(info.parent_link)
            locked_links.add(info.child_link)
        for link, mesh_names in link_to_meshes.items():
            color = _ROBOT_COLOR_LOCKED if link in locked_links else _ROBOT_COLOR_NORMAL
            for nname in mesh_names:
                if nname in robot_handles:
                    robot_handles[nname].set_color(color)

    # ---- Drag debug log ----------------------------------------------------
    _drag_log_path = snap_dir / "drag_debug.log"
    _drag_log_path.write_text("", encoding="utf-8")   # truncate on startup
    _drag_log_fh = open(_drag_log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

    def _log_drag(msg: str) -> None:
        _drag_log_fh.write(msg + "\n")
        _drag_log_fh.flush()

    # ---- Gizmo sync: reset all gizmos to current FK ------------------------
    def _sync_gizmos_to_fk() -> None:
        # Palm gizmos: keep full FK transform (position + orientation) so the
        # rotation rings follow the wrist's current orientation.
        for arm, link in (("R", _RIGHT_PALM), ("L", _LEFT_PALM)):
            T = ctrl.get_link_transform(link).copy()
            palm_gizmos[arm].set_transform(T)
            state[f"last_palm_{arm}"] = T.copy()
        # Fingertip gizmos: keep rotation at identity so the translation arrows
        # stay aligned with world X/Y/Z (matches `set_interact_in_local_space=False`).
        # Only the position tracks the fingertip link.
        for arm in ("R", "L"):
            side = "right" if arm == "R" else "left"
            for fi in _FINGERS:
                pos = ctrl.get_link_transform(_fingertip_link(side, fi))[:3, 3]
                T = np.eye(4); T[:3, 3] = pos
                tip_gizmos[(arm, fi)].set_transform(T)
                state[f"last_tip_{arm}{fi}"] = T.copy()

    # ---- Update Phase A ----
    def update_phase_A(fi: int) -> None:
        fi = int(np.clip(fi, 0, n_traj - 1))
        state["frame_A"] = fi

        # Teleop qpos (includes fingers) sets the baseline pose.
        _apply_qpos(ctrl, qpos[fi], GENESIS_QPOS_ORDER)

        pan_tf  = _xyzq_to_4x4(pan_xyzq[fi])
        spat_tf = _xyzq_to_4x4(spat_xyzq[fi])

        # If we already have a calibration, drive the arms via IK exactly
        # like Phase B so the hand stays glued to the tool handles.  This
        # matches what the user saw in Phase B and keeps hand<->tool in sync
        # across phase switches.  Without calibration we show the raw teleop
        # pose (which typically has a handle offset because Genesis records
        # the tool transform separately from the arm qpos).
        if calib.is_complete and state["follow_pan_in_A"]:
            palm_R_target = pan_tf  @ calib.T_palm_R_rel_pan
            palm_L_target = spat_tf @ calib.T_palm_L_rel_spat
            r_active = _active_arm_joints("R")
            l_active = _active_arm_joints("L")
            if r_active:
                ctrl.solve_ik(_RIGHT_PALM, palm_R_target[:3, 3],
                              target_orientation=palm_R_target[:3, :3],
                              orientation_mode="all", arm_joints=r_active)
            if l_active:
                ctrl.solve_ik(_LEFT_PALM,  palm_L_target[:3, 3],
                              target_orientation=palm_L_target[:3, :3],
                              orientation_mode="all", arm_joints=l_active)

        # Apply calibrated finger angles on top (overrides teleop fingers).
        if calib.finger_angles_R:
            ctrl.set_joint_positions(calib.finger_angles_R)
        if calib.finger_angles_L:
            ctrl.set_joint_positions(calib.finger_angles_L)
        _update_robot(ctrl, robot_handles)

        ps_pan_A.update_vertex_positions(_transform_verts(pan_v_local, pan_tf))
        ps_spat_A.update_vertex_positions(_transform_verts(spat_v_local, spat_tf))

        _sync_gizmos_to_fk()

    # ---- Update Phase B ----
    def update_phase_B(fi: int) -> None:
        fi = int(np.clip(fi, 0, seq.n_frames - 1))
        state["frame_B"] = fi

        # Tool transforms from sim
        pan_tf  = seq.rigid["pan"][2][fi]
        spat_tf = seq.rigid["spatula"][2][fi]

        # Compute palm targets from calibration
        if calib.is_complete:
            palm_R_target = pan_tf  @ calib.T_palm_R_rel_pan
            palm_L_target = spat_tf @ calib.T_palm_L_rel_spat
            r_active = _active_arm_joints("R")
            l_active = _active_arm_joints("L")
            # IK each arm (6-DOF: position + full orientation)
            if r_active:
                ctrl.solve_ik(_RIGHT_PALM, palm_R_target[:3, 3],
                              target_orientation=palm_R_target[:3, :3],
                              orientation_mode="all", arm_joints=r_active)
            if l_active:
                ctrl.solve_ik(_LEFT_PALM,  palm_L_target[:3, 3],
                              target_orientation=palm_L_target[:3, :3],
                              orientation_mode="all", arm_joints=l_active)

        # Phase B always uses the captured finger angles (sim has no fingers).
        if calib.finger_angles_R:
            ctrl.set_joint_positions(calib.finger_angles_R)
        if calib.finger_angles_L:
            ctrl.set_joint_positions(calib.finger_angles_L)
        _update_robot(ctrl, robot_handles)
        _sync_gizmos_to_fk()

        # Update all seq_v1 meshes
        for name, sm in rigid_handles_B.items():
            v_local, _faces, tfs = seq.rigid[name]
            sm.update_vertex_positions(_transform_verts(v_local, tfs[fi]))
        for name, sm in fem_handles_B.items():
            _faces, positions = seq.fem[name]
            sm.update_vertex_positions(positions[fi])
        for name, cn in rod_handles_B.items():
            _edges, positions = seq.rod[name]
            cn.update_node_positions(positions[fi])

    # ---- Phase visibility ----
    def _set_phase(phase: str) -> None:
        state["phase"] = phase
        state["playing"] = False
        is_A = (phase == "A")
        ps_pan_A.set_enabled(is_A)
        ps_spat_A.set_enabled(is_A)
        for sm in rigid_handles_B.values(): sm.set_enabled(not is_A)
        for sm in fem_handles_B.values():   sm.set_enabled(not is_A)
        for cn in rod_handles_B.values():   cn.set_enabled(not is_A)
        # Gizmo handles only make sense in Phase A.  Also clear any active
        # gizmo when leaving Phase A so nothing floats around in Phase B.
        for sm in palm_gizmos.values():
            sm.set_enabled(is_A and state["show_palm_gizmos"])
            sm.set_transform_gizmo_enabled(False)
        for sm in tip_gizmos.values():
            sm.set_enabled(is_A and state["show_tip_gizmos"])
            sm.set_transform_gizmo_enabled(False)
        if not is_A:
            state["sel_gizmo_name"] = None
        if is_A:
            update_phase_A(state["frame_A"])
        else:
            update_phase_B(state["frame_B"])

    _set_phase("A")

    # ---- Gizmo selection tracking -------------------------------------------
    def _current_selection_name() -> str | None:
        """Return the Polyscope name of the currently selected structure, or None."""
        if not ps.have_selection():
            return None
        sel = ps.get_selection()
        # PickResult may expose the structure name directly or via .structure.
        if hasattr(sel, "structure_name"):
            return sel.structure_name
        s = getattr(sel, "structure", None)
        if s is not None:
            if hasattr(s, "get_name"):
                return s.get_name()
            if hasattr(s, "name"):
                return s.name
        return None

    def _update_selection_gizmos() -> None:
        """Enable the gizmo of the selected handle; disable all others."""
        sel_name = _current_selection_name()
        # Only track selections on *our* gizmo handles; otherwise keep the
        # previous selection (avoids losing gizmo when user clicks the robot).
        if sel_name in gizmo_lookup:
            state["sel_gizmo_name"] = sel_name
        target = state["sel_gizmo_name"]
        for name, sm in [(palm_gizmos[a].get_name(), palm_gizmos[a])
                          for a in ("R", "L")] + \
                         [(tip_gizmos[(a, fi)].get_name(), tip_gizmos[(a, fi)])
                          for a in ("R", "L") for fi in _FINGERS]:
            sm.set_transform_gizmo_enabled(name == target)

    # ---- Gizmo drag polling -------------------------------------------------
    _GIZMO_EPS = 1e-4   # ignore floating-point noise (<0.1 mm / <0.006 deg)

    def _mat_changed(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.any(np.abs(a - b) > _GIZMO_EPS))

    def _sync_other_gizmos(skip: str) -> None:
        """Sync all gizmos except the one whose name matches *skip* to FK.
        This keeps fingertip gizmos attached to the hand when the palm moves,
        but leaves the gizmo the user is currently dragging untouched."""
        for arm, link in (("R", _RIGHT_PALM), ("L", _LEFT_PALM)):
            name = palm_gizmos[arm].get_name()
            if name == skip:
                continue
            T = ctrl.get_link_transform(link).copy()
            palm_gizmos[arm].set_transform(T)
            state[f"last_palm_{arm}"] = T.copy()
        for arm in ("R", "L"):
            side = "right" if arm == "R" else "left"
            for fi in _FINGERS:
                name = tip_gizmos[(arm, fi)].get_name()
                if name == skip:
                    continue
                pos = ctrl.get_link_transform(_fingertip_link(side, fi))[:3, 3]
                T = np.eye(4); T[:3, 3] = pos
                tip_gizmos[(arm, fi)].set_transform(T)
                state[f"last_tip_{arm}{fi}"] = T.copy()

    # Incremental step sizes: each frame the IK target advances at most this
    # much toward the gizmo.  Keeps IK in the linear regime and avoids
    # jumping to a bad local minimum (which was the reason pulling the
    # fingertip red axis made the finger shoot up instead).
    _TIP_STEP_M    = 0.005    # 5 mm per frame for fingertips
    _PALM_STEP_M   = 0.020    # 20 mm per frame for palm position
    _AT_TARGET_TOL = 1e-4     # stop running IK when within 0.1 mm

    def _poll_gizmo_drags() -> None:
        """Pull the currently selected gizmo's associated link toward the
        gizmo's world transform, advancing one small step per frame so IK
        stays in the well-behaved linear regime."""
        target = state["sel_gizmo_name"]
        if target is None or target not in gizmo_lookup:
            return
        kind, key = gizmo_lookup[target]

        if kind == "palm":
            arm = key
            gz   = palm_gizmos[arm]
            cur_T = np.asarray(gz.get_transform(), dtype=np.float64)
            state[f"last_palm_{arm}"] = cur_T.copy()
            if not state["apply_ik_on_drag"]:
                return
            palm_link  = _RIGHT_PALM if arm == "R" else _LEFT_PALM
            arm_joints = _active_arm_joints(arm)
            if not arm_joints:
                return  # nothing to solve - all joints are locked
            fk         = ctrl.get_link_transform(palm_link)
            dir_vec    = cur_T[:3, 3] - fk[:3, 3]
            dist       = float(np.linalg.norm(dir_vec))
            rot_close  = bool(np.allclose(fk[:3, :3], cur_T[:3, :3], atol=1e-3))
            if dist < _AT_TARGET_TOL and rot_close:
                return
            step_pos = (fk[:3, 3] + dir_vec * (_PALM_STEP_M / dist)
                        if dist > _PALM_STEP_M else cur_T[:3, 3])
            ctrl.solve_ik(palm_link, step_pos,
                          target_orientation=cur_T[:3, :3],
                          orientation_mode="all",
                          arm_joints=arm_joints)
            _update_robot(ctrl, robot_handles)
            _sync_other_gizmos(target)

        elif kind == "tip":
            arm, fi = key
            gz = tip_gizmos[(arm, fi)]
            cur_pos = np.asarray(gz.get_position(), dtype=np.float64)
            state[f"last_tip_{arm}{fi}"] = np.asarray(gz.get_transform()).copy()
            if not state["apply_ik_on_drag"]:
                return
            side = "right" if arm == "R" else "left"
            finger_joints = [_finger_joint(side, fi, ji) for ji in _KNUCKLES]
            tip_link = _fingertip_link(side, fi)
            cur_tip  = ctrl.get_link_transform(tip_link)[:3, 3]
            dir_vec  = cur_pos - cur_tip
            dist     = float(np.linalg.norm(dir_vec))
            if dist < _AT_TARGET_TOL:
                return
            step_target = (cur_tip + dir_vec * (_TIP_STEP_M / dist)
                           if dist > _TIP_STEP_M else cur_pos)
            joints_before = {j: float(ctrl.get_joint_positions().get(j, 0.0))
                              for j in finger_joints}
            ctrl.solve_ik(tip_link, step_target, arm_joints=finger_joints)
            _update_robot(ctrl, robot_handles)
            new_tip = ctrl.get_link_transform(tip_link)[:3, 3]
            joints_after = {j: float(ctrl.get_joint_positions().get(j, 0.0))
                             for j in finger_joints}
            dj = {j: joints_after[j] - joints_before[j] for j in finger_joints}
            _log_drag(
                f"[TIP {arm}{fi}] "
                f"gizmo={cur_pos.round(4).tolist()}  "
                f"cur_fk={cur_tip.round(4).tolist()}  "
                f"dir={dir_vec.round(4).tolist()}  dist={dist:.4f}  "
                f"step_tgt={np.asarray(step_target).round(4).tolist()}"
            )
            _log_drag(
                f"          after IK: new_fk={new_tip.round(4).tolist()}  "
                f"joint_deltas={[(j, round(d, 4)) for j, d in dj.items()]}"
            )

    # ---- Capture handler ----
    def do_capture() -> None:
        # The current palm transforms already include any wrist deltas (they
        # were applied via IK in update_phase_A), so we can read them directly.
        palm_R_w = ctrl.get_link_transform(_RIGHT_PALM)
        palm_L_w = ctrl.get_link_transform(_LEFT_PALM)
        pan_w  = _xyzq_to_4x4(pan_xyzq[state["frame_A"]])
        spat_w = _xyzq_to_4x4(spat_xyzq[state["frame_A"]])

        calib.T_palm_R_rel_pan  = np.linalg.inv(pan_w)  @ palm_R_w
        calib.T_palm_L_rel_spat = np.linalg.inv(spat_w) @ palm_L_w
        # Snapshot all finger angles (current state of the controller)
        cur = ctrl.get_joint_positions()
        calib.finger_angles_R = {j: float(cur[j]) for j in _RIGHT_FINGER_JOINTS
                                 if j in cur}
        calib.finger_angles_L = {j: float(cur[j]) for j in _LEFT_FINGER_JOINTS
                                 if j in cur}
        # Record current robot base pose so the next launch reproduces it.
        calib.base_pos     = [state["base_x"], state["base_y"], state["base_z"]]
        calib.base_rpy_deg = [state["base_rx_deg"], state["base_ry_deg"],
                              state["base_rz_deg"]]
        calib.save(_CALIB)

    # ---- Export robot trajectory for ipc_robot_cook.py ----
    def do_export_robot_seq() -> None:
        """Run Phase B IK for every frame and dump the resulting qpos
        sequence (+ base pose) to <seq_dir>/robot.npz so ipc_robot_cook.py
        can replay the arm motion alongside the cooking scene."""
        if not calib.is_complete:
            print("  [EXPORT] No calibration available; cannot export robot seq.")
            return

        n = seq.n_frames
        qposes = np.zeros((n, 54), dtype=np.float32)
        saved_frame = state["frame_B"]
        print(f"  [EXPORT] Computing IK for {n} frames ...", flush=True)
        for fi in range(n):
            update_phase_B(fi)
            cur = ctrl.get_joint_positions()
            for i, jname in enumerate(GENESIS_QPOS_ORDER):
                qposes[fi, i] = float(cur.get(jname, 0.0))
            if (fi + 1) % 50 == 0 or fi == n - 1:
                print(f"    frame {fi + 1}/{n}", flush=True)
        update_phase_B(saved_frame)

        out_path = current_seq_dir[0] / "robot.npz"
        np.savez(
            str(out_path),
            qpos          = qposes,
            urdf          = str(_GSS_URDF.relative_to(_DEMO_ROOT)),
            base_pos      = np.asarray(
                [state["base_x"], state["base_y"], state["base_z"]],
                dtype=np.float32),
            base_rpy_deg  = np.asarray(
                [state["base_rx_deg"], state["base_ry_deg"],
                 state["base_rz_deg"]], dtype=np.float32),
            frame_count   = np.int32(n),
            qpos_order    = np.asarray(GENESIS_QPOS_ORDER, dtype=object),
        )
        print(f"  [EXPORT] Saved {n} frames + base pose -> {out_path}",
              flush=True)

    # ---- GUI ----
    def gui() -> None:
        if state["_first_render"]:
            state["_first_render"] = False
            ps.look_at(camera_location=(1.4, -1.0, 1.3),
                       target=(0.35, 0.0, 0.9))
            p = snap_dir / "replay_frame0.png"
            ps.screenshot(str(p))
            print(f"  [SNAP] frame-0 -> {p}")

        # ---- Phase toggle ----
        cur_label = f"[Phase {state['phase']}]"
        other_phase = "B" if state["phase"] == "A" else "A"
        if imgui.Button(f"Switch to Phase {other_phase}"):
            _set_phase(other_phase)
        imgui.SameLine(); imgui.TextUnformatted(cur_label)
        imgui.SameLine();
        imgui.TextUnformatted(
            "(A = Calibration on teleop)" if state["phase"] == "A"
            else "(B = Replay sim with IK)"
        )
        # "Load frame on switch" - lets the user pick the target frame for the
        # OTHER phase so the moment they press Switch they jump there.
        if other_phase == "B":
            ch, v = imgui.SliderInt(
                f"Phase B start frame##switch", state["frame_B"], 0, seq.n_frames - 1)
            if ch:
                state["frame_B"] = int(v)
        else:
            ch, v = imgui.SliderInt(
                f"Phase A start frame##switch", state["frame_A"], 0, n_traj - 1)
            if ch:
                state["frame_A"] = int(v)
        imgui.Separator()

        # ---- Frame slider ----
        is_A = state["phase"] == "A"
        n_max = (n_traj if is_A else seq.n_frames) - 1
        cur_frame = state["frame_A"] if is_A else state["frame_B"]
        imgui.TextUnformatted(
            f"Frame {cur_frame}/{n_max}  "
            f"({'PLAY' if state['playing'] else 'PAUSE'}  x{state['speed']:.2f})"
        )
        if imgui.Button("Play / Pause"):
            state["playing"] = not state["playing"]; state["_acc"] = 0.0
        imgui.SameLine()
        if imgui.Button("|< Reset"):
            state["playing"] = False
            (update_phase_A if is_A else update_phase_B)(0)
        imgui.SameLine()
        if imgui.Button(">| Last"):
            state["playing"] = False
            (update_phase_A if is_A else update_phase_B)(n_max)
        imgui.SameLine()
        if imgui.Button("Snap"):
            p = snap_dir / f"replay_phase{state['phase']}_f{cur_frame:04d}.png"
            ps.screenshot(str(p)); print(f"  [SNAP] -> {p}")

        changed, v = imgui.SliderInt("Frame", cur_frame, 0, n_max)
        if changed:
            state["playing"] = False
            (update_phase_A if is_A else update_phase_B)(v)
        changed, v = imgui.SliderFloat("Speed", state["speed"], 0.1, 4.0)
        if changed: state["speed"] = float(v)

        # ---- Selection-gated gizmos + drag polling (Phase A only) ----
        if is_A:
            _update_selection_gizmos()
            _poll_gizmo_drags()

        # ---- Phase-A controls ----
        if is_A:
            imgui.Separator()
            # Gizmo handle visibility (whether the small coloured cubes show at all)
            ch, v = imgui.Checkbox("Show palm handles", state["show_palm_gizmos"])
            if ch:
                state["show_palm_gizmos"] = bool(v)
                for sm in palm_gizmos.values():
                    sm.set_enabled(state["show_palm_gizmos"])
            imgui.SameLine()
            ch, v = imgui.Checkbox("Show fingertip handles", state["show_tip_gizmos"])
            if ch:
                state["show_tip_gizmos"] = bool(v)
                for sm in tip_gizmos.values():
                    sm.set_enabled(state["show_tip_gizmos"])

            ch, v = imgui.Checkbox("Apply IK on drag  (uncheck to debug gizmo only)",
                                    state["apply_ik_on_drag"])
            if ch:
                state["apply_ik_on_drag"] = bool(v)

            ch, v = imgui.Checkbox(
                "Follow tool with calib IK  (hand tracks pan/spatula)",
                state["follow_pan_in_A"])
            if ch:
                state["follow_pan_in_A"] = bool(v)
                update_phase_A(state["frame_A"])

            # Current selection readout + live debug of what the gizmo reports
            sel = state["sel_gizmo_name"]
            imgui.TextUnformatted(
                f"Selected: {sel or '(none — click a handle to activate its gizmo)'}"
            )
            if imgui.Button("Clear selection"):
                state["sel_gizmo_name"] = None
                ps.reset_selection()

            if sel is not None and sel in gizmo_lookup:
                kind, key = gizmo_lookup[sel]
                if kind == "palm":
                    gz = palm_gizmos[key]
                    link = _RIGHT_PALM if key == "R" else _LEFT_PALM
                else:
                    gz = tip_gizmos[key]
                    arm, fi = key
                    side = "right" if arm == "R" else "left"
                    link = _fingertip_link(side, fi)
                try:
                    gp = np.asarray(gz.get_position(), dtype=np.float64)
                except Exception:
                    gp = np.asarray(gz.get_transform(), dtype=np.float64)[:3, 3]
                fk_p = ctrl.get_link_transform(link)[:3, 3]
                imgui.TextUnformatted(
                    f"  gizmo.get_position() = "
                    f"[{gp[0]:+.4f}, {gp[1]:+.4f}, {gp[2]:+.4f}]"
                )
                imgui.TextUnformatted(
                    f"  FK({link}) = "
                    f"[{fk_p[0]:+.4f}, {fk_p[1]:+.4f}, {fk_p[2]:+.4f}]   "
                    f"diff = {np.linalg.norm(gp - fk_p):.4f} m"
                )

            # Palm <-> tool distance readout
            palm_R_w = ctrl.get_link_transform(_RIGHT_PALM)[:3, 3]
            palm_L_w = ctrl.get_link_transform(_LEFT_PALM)[:3, 3]
            pan_w_xyz  = pan_xyzq[state["frame_A"]][:3]
            spat_w_xyz = spat_xyzq[state["frame_A"]][:3]
            imgui.TextUnformatted(
                f"R palm<->pan  = {np.linalg.norm(palm_R_w - pan_w_xyz):.3f} m   "
                f"L palm<->spat = {np.linalg.norm(palm_L_w - spat_w_xyz):.3f} m"
            )
            imgui.TextUnformatted(
                "Click a coloured handle to show its gizmo (palm = 6-DOF, tip = translate)."
            )

            imgui.Separator()
            if imgui.Button("Capture & Save Calibration"):
                do_capture()
            imgui.SameLine()
            if imgui.Button("Reset to teleop frame"):
                update_phase_A(state["frame_A"])    # re-apply qpos, re-sync gizmos

        # ---- Phase-B status ----
        if not is_A:
            imgui.Separator()
            # Scene-source switcher: lets the user pick between any seq*/
            # subdirectory found at startup.
            imgui.TextUnformatted(f"Scene source: [{current_seq_dir[0].name}]")
            for opt_dir in seq_options:
                is_active = opt_dir == current_seq_dir[0]
                if imgui.RadioButton(opt_dir.name + "##scene_src", is_active):
                    if not is_active:
                        _reload_scene(opt_dir)
                imgui.SameLine()
            imgui.NewLine()

            if calib.is_complete:
                imgui.TextUnformatted("Calibration loaded.  IK is driving the arms.")
            else:
                imgui.TextUnformatted(
                    "[!] No calibration yet.  Switch to Phase A and press Capture."
                )
            if imgui.Button(f"Export robot seq -> {current_seq_dir[0].name}/robot.npz"):
                do_export_robot_seq()
            # Robot base pose (6-DOF): XYZ translation + RPY Euler rotation.
            imgui.TextUnformatted("Robot base pose:")
            base_changed = False
            for lbl, key, rng in (
                ("Base X",   "base_x",     (-1.0, 1.0)),
                ("Base Y",   "base_y",     (-1.0, 1.0)),
                ("Base Z",   "base_z",     ( 0.3, 1.6)),
                ("Base roll  (deg)", "base_rx_deg", (-180.0, 180.0)),
                ("Base pitch (deg)", "base_ry_deg", (-180.0, 180.0)),
                ("Base yaw   (deg)", "base_rz_deg", (-180.0, 180.0)),
            ):
                ch, v = imgui.SliderFloat(lbl, state[key], rng[0], rng[1])
                if ch:
                    state[key] = float(v)
                    base_changed = True
            if imgui.Button("Reset base pose"):
                state["base_x"]      = float(_ROBOT_BASE_POS[0])
                state["base_y"]      = float(_ROBOT_BASE_POS[1])
                state["base_z"]      = float(_ROBOT_BASE_POS[2])
                state["base_rx_deg"] = 0.0
                state["base_ry_deg"] = 0.0
                state["base_rz_deg"] = 0.0
                base_changed = True
            if base_changed:
                _set_base_pose(state["base_x"], state["base_y"], state["base_z"],
                               state["base_rx_deg"], state["base_ry_deg"],
                               state["base_rz_deg"])
                update_phase_B(state["frame_B"])

            # Lock arm joints from IK (keep teleop value).  Useful for joints
            # the human anatomy doesn't have, e.g. forearm twist.
            _, state["lock_panel_open"] = imgui.CollapsingHeader(
                "Lock arm joints (skip in IK)##B", state["lock_panel_open"])
            if state["lock_panel_open"]:
                cur_q = ctrl.get_joint_positions()
                changed_lock = False
                for arm_label, joints in (("Right arm", _RIGHT_ARM_JOINTS),
                                          ("Left arm",  _LEFT_ARM_JOINTS)):
                    imgui.TextUnformatted(arm_label)
                    for jname in joints:
                        deg = float(np.degrees(cur_q.get(jname, 0.0)))
                        locked = jname in state["locked_arm_joints"]
                        ch, v = imgui.Checkbox(
                            f"{jname}  ({deg:+.0f} deg)##lock",
                            locked,
                        )
                        if ch:
                            if v:
                                state["locked_arm_joints"].add(jname)
                            else:
                                state["locked_arm_joints"].discard(jname)
                            changed_lock = True
                imgui.TextUnformatted(
                    f"  Currently locked: {sorted(state['locked_arm_joints'])}"
                )
                if imgui.Button("Unlock all##B"):
                    state["locked_arm_joints"].clear()
                    changed_lock = True
                if changed_lock:
                    _apply_lock_colors()
                    update_phase_B(state["frame_B"])

        # ---- Auto-play step ----
        if state["playing"]:
            state["_acc"] += state["speed"]
            step = int(state["_acc"])
            if step >= 1:
                state["_acc"] -= step
                nxt = cur_frame + step
                if nxt >= n_max:
                    (update_phase_A if is_A else update_phase_B)(n_max)
                    state["playing"] = False
                else:
                    (update_phase_A if is_A else update_phase_B)(nxt)

    ps.set_user_callback(gui)
    print("\nPolyscope window ready.")
    print("  Phase A: tune Grip + small wrist offsets, then Capture.")
    print("  Phase B: with calibration loaded, IK drives the arms over the sim.")
    ps.show()


if __name__ == "__main__":
    main()
