"""
Replay exported yoyo simulation sequences.

Usage:
    python examples/IPC_Solver/replay_yoyo_traj.py
    python examples/IPC_Solver/replay_yoyo_traj.py --trajectory long_sleep
    python examples/IPC_Solver/replay_yoyo_traj.py --render --nyx # v4
    python examples/IPC_Solver/replay_yoyo_traj.py --render --nyx --camera-traj surround
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from _replay_common import (
    CameraTrajectory,
    SurroundCamera,
    TrajectoryReplay,
    marvin_urdf,
)
from _yoyo_common import (
    INTERNAL_OPACITY_TRANSPARENT,
    REPLAY_CAMERA_FOV,
    REPLAY_CAMERA_LOOKAT,
    REPLAY_CAMERA_POS,
    REPLAY_CAMERA_UP,
    SHELL_OPACITY_TRANSPARENT,
    TRAJECTORIES,
    YOYO_ASSETS_DIR,
    load_seq_data,
    resolve_mesh_path,
    tf_to_pos_quat,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Camera orbit parameters (v3) ──
ORBIT_CENTER = np.array([0.20, 0.0, -0.25])  # shifted down 10cm more to keep yoyo in frame
ORBIT_RADIUS = 0.80  # 10cm further (was 0.70)
ORBIT_HEIGHT = 0.15  # 5cm higher (was 0.10)
ORBIT_ANGLE_START = math.radians(-60)
ORBIT_ANGLE_END = math.radians(60)
CLOSEUP_DISTANCE = 0.02
CLOSEUP_FOV = 40


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _find_asset(name):
    p = YOYO_ASSETS_DIR / f"{name}.glb"
    return str(p) if p.exists() else None


# ── Custom camera trajectories ──────────────────────────────────────────────


class YoyoOrbitCamera(CameraTrajectory):
    """Half-circle orbit around the robot (v3 trajectory)."""

    def __init__(self, freeze_frac=None):
        self._freeze_frac = freeze_frac  # if set, clamp frac at this value

    def get_pose(self, frame_idx, n_frames):
        frac = frame_idx / max(n_frames - 1, 1)
        if self._freeze_frac is not None:
            frac = min(frac, self._freeze_frac)
        angle = ORBIT_ANGLE_START + (ORBIT_ANGLE_END - ORBIT_ANGLE_START) * _smooth(frac)
        cam_pos = (
            ORBIT_CENTER[0] + ORBIT_RADIUS * math.cos(angle),
            ORBIT_CENTER[1] + ORBIT_RADIUS * math.sin(angle),
            ORBIT_HEIGHT,
        )
        return cam_pos, tuple(ORBIT_CENTER)


class YoyoCloseupCamera(CameraTrajectory):
    """Static close-up tracking the yoyo ball (long_sleep trajectory)."""

    def __init__(self, rigid_data):
        self._rigid_data = rigid_data

    def get_pose(self, frame_idx, n_frames):
        if "yoyo_ball" in self._rigid_data and frame_idx < self._rigid_data["yoyo_ball"].shape[0]:
            ball_pos = self._rigid_data["yoyo_ball"][frame_idx][:3].copy()
        else:
            ball_pos = np.array([0.256, 0.008, -0.056])
        cam_pos = ball_pos + np.array([0.0, -CLOSEUP_DISTANCE, 0.0])
        return tuple(cam_pos), tuple(ball_pos)


# ── YoyoReplay ──────────────────────────────────────────────────────────────


class YoyoReplay(TrajectoryReplay):
    name = "yoyo"
    cam_pos = REPLAY_CAMERA_POS
    cam_lookat = REPLAY_CAMERA_LOOKAT
    cam_fov = REPLAY_CAMERA_FOV

    def add_args(self, parser):
        parser.add_argument(
            "--trajectory",
            type=str,
            default="v3",
            choices=list(TRAJECTORIES.keys()),
            help="Trajectory to replay.",
        )
        parser.add_argument("--seq-dir", type=str, default=None, help="Override sequence directory.")

    def load_trajectory(self):
        args = self.args
        if args.seq_dir:
            self._seq_dir = Path(args.seq_dir)
        else:
            self._seq_dir = TRAJECTORIES[args.trajectory]

        meta_path = self._seq_dir / "meta.json"
        assert meta_path.exists(), f"meta.json not found in {self._seq_dir}"
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self._is_long_sleep = "long_sleep" in str(self._seq_dir)
        self._force_closeup_camera = False
        if self._is_long_sleep or self._force_closeup_camera:
            self.cam_fov = CLOSEUP_FOV

        robot_pos = self._meta.get("robot_base_pos")
        if robot_pos and robot_pos[2] > 0.5:
            rz = robot_pos[2]
            self.cam_pos = (1.2, -0.8, rz + 0.6)
            self.cam_lookat = (0.2, 0.0, rz - 0.1)
            self.cam_fov = 45

        # Joint data (remapped to qpos in post_build via _remap_joint_data)
        joint_meta = self._meta.get("joints", {})
        self._joint_names = joint_meta.get("names", [])
        self._raw_joint_data = None
        if joint_meta.get("data"):
            joints_path = self._seq_dir / joint_meta["data"]
            if joints_path.exists():
                self._raw_joint_data = np.load(str(joints_path))

        # Rigid + FEM data
        rigid_raw, self._fem_data = load_seq_data(self._seq_dir, self._meta)

        # Convert rigid transforms from (N,4,4) to (N,7) pos+quat
        from scipy.spatial.transform import Rotation

        self._rigid_data = {}
        for name, data in rigid_raw.items():
            if data.ndim == 3 and data.shape[1:] == (4, 4):
                n = data.shape[0]
                pos = data[:, :3, 3]
                xyzw = Rotation.from_matrix(data[:, :3, :3]).as_quat()
                # Genesis uses wxyz
                wxyz = np.column_stack([xyzw[:, 3], xyzw[:, :3]])
                self._rigid_data[name] = np.column_stack([pos, wxyz]).astype(np.float32)
            else:
                self._rigid_data[name] = data

        n_frames = self._meta["frame_count"]
        dt = self._meta.get("dt", 0.001)
        frame_skip = self._meta.get("frame_skip", 10)
        self.fps = min(int(1.0 / dt / frame_skip), 60)

        # Default camera trajectory: orbit for v3, close-up for long_sleep
        if not self.args.camera_traj:
            self.args.camera_traj = "_yoyo_default"

        return n_frames

    def build_scene(self, scene):
        import genesis as gs

        use_nyx = self.args.nyx
        is_long_sleep = self._is_long_sleep or getattr(self, "_force_closeup_camera", False)

        # Robot — marvin URDFs always come from the internal_assets HF dataset
        # (marvin_description/marvin_robots/), regardless of any meta.urdf path.
        urdf_rel = self._meta.get("urdf", "").replace("\\", "/")
        assert urdf_rel, "meta.json must specify 'urdf'"
        if "marvin_sharpa" in urdf_rel:
            urdf_path = Path(marvin_urdf("marvin_sharpa"))
        elif "marvin_pika" in urdf_rel:
            urdf_path = Path(marvin_urdf("marvin_pika"))
        else:
            urdf_path = Path(urdf_rel)
            if not urdf_path.is_absolute():
                urdf_path = _REPO_ROOT / urdf_rel
        assert urdf_path.exists(), f"Robot URDF not found: {urdf_path}"
        robot_pos = self._meta.get("robot_base_pos", [0, 0, 0])
        self._robot = scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf_path),
                fixed=True,
                collision=False,
                pos=tuple(robot_pos),
            ),
            material=gs.materials.Rigid(coup_type="external_articulation"),
            vis_mode="visual",
            name="robot",
        )

        # Rigid yoyo parts
        self._rigid_entities: dict[str, list] = {}

        for name in self._rigid_data:
            if name == "yoyo_ball":
                shell_alpha = SHELL_OPACITY_TRANSPARENT if is_long_sleep else None
                internal_alpha = INTERNAL_OPACITY_TRANSPARENT if is_long_sleep else None
                ball_parts = []
                for asset_name, opacity in [
                    ("yoyo-top_shell", shell_alpha),
                    ("yoyo-bottom_shell", shell_alpha),
                    ("yoyo-top_ring", shell_alpha),
                    ("yoyo-bottom_ring", shell_alpha),
                    ("yoyo-axle", internal_alpha),
                    ("yoyo-hub", internal_alpha),
                ]:
                    ent = self._add_ball_part(scene, asset_name, opacity, use_nyx)
                    if ent is not None:
                        ball_parts.append(ent)
                self._rigid_entities["yoyo_ball"] = ball_parts
            else:
                mesh_path = resolve_mesh_path(name, self._seq_dir)
                if mesh_path is None:
                    continue
                surface = None
                if is_long_sleep and name == "bearing_outer":
                    surface = gs.surfaces.BSDF(
                        opacity_texture=gs.textures.ColorTexture(color=(INTERNAL_OPACITY_TRANSPARENT,)),
                    )
                ent = scene.add_entity(
                    morph=gs.morphs.Mesh(file=str(mesh_path), fixed=True, collision=False),
                    material=gs.materials.Rigid(),
                    surface=surface,
                    name=name,
                )
                self._rigid_entities[name] = [ent]

        # String (FEM rope)
        self._fem_entities: dict[str, object] = {}
        for name in self._fem_data:
            mesh_path = self._seq_dir / name / "mesh.obj"
            if not mesh_path.exists():
                continue
            ent = scene.add_entity(
                morph=gs.morphs.Mesh(file=str(mesh_path)),
                material=gs.materials.FEM.Rope(E=1e6, rho=100.0, thickness=0.0004),
                surface=gs.surfaces.Default(color=(0.9, 0.87, 0.8, 1.0)),
                name=name,
            )
            self._fem_entities[name] = ent
            break

    def _add_ball_part(self, scene, asset_name, opacity, use_nyx):
        import genesis as gs

        if opacity is not None and use_nyx:
            p = _find_asset(asset_name + "_transparent")
            if p is None:
                p = _find_asset(asset_name)
            surface = None
        elif opacity is not None:
            p = _find_asset(asset_name)
            surface = gs.surfaces.BSDF(
                opacity_texture=gs.textures.ColorTexture(color=(opacity,)),
            )
        else:
            p = _find_asset(asset_name)
            surface = None
        if p is None:
            return None
        return scene.add_entity(
            morph=gs.morphs.Mesh(file=p, fixed=True, collision=False),
            material=gs.materials.Rigid(),
            surface=surface,
            name=asset_name,
        )

    def apply_frame(self, scene, frame_idx):
        super().apply_frame(scene, frame_idx)
        if frame_idx in (30, 31, 32, 33):
            self._save_rope_tube_mesh(frame_idx)

    def _save_rope_tube_mesh(self, frame_idx):
        import trimesh
        from genesis.engine.mesh import LineMesh

        out_dir = Path("data/ipc_demo/ipc_yoyo/rope_tube_meshes")
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, entity in self._fem_entities.items():
            for rm_idx, rmesh in enumerate(entity.render_meshes):
                if not isinstance(rmesh, LineMesh) or rmesh.tube_faces is None:
                    continue
                # Get current sim vertex positions
                sim_verts = self._fem_data[name][frame_idx]
                tube_verts = rmesh.build_tube_verts(sim_verts.astype(np.float32))
                tube_faces = rmesh.tube_faces
                mesh = trimesh.Trimesh(vertices=tube_verts, faces=tube_faces, process=False)
                path = out_dir / f"{name}_frame{frame_idx}.obj"
                mesh.export(str(path))
                print(f"[save] Tube mesh: {path} ({len(tube_verts)} verts, {len(tube_faces)} faces)")

    def make_camera_traj(self, name):
        if name == "_yoyo_default":
            if self._is_long_sleep or getattr(self, "_force_closeup_camera", False):
                return YoyoCloseupCamera(self._rigid_data)
            return YoyoOrbitCamera(freeze_frac=0.606)  # stop rotating at sub10 f296
        if name == "surround":
            return SurroundCamera(
                center=tuple(ORBIT_CENTER),
                radius=ORBIT_RADIUS,
                height=ORBIT_HEIGHT,
                angle_start=math.degrees(ORBIT_ANGLE_START),
                angle_end=math.degrees(ORBIT_ANGLE_END),
            )
        return super().make_camera_traj(name)


if __name__ == "__main__":
    YoyoReplay().run()
