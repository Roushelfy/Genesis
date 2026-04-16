"""
Replay exported yoyo v5 simulation sequences.

Uses OBJ meshes directly from the v5/seq directory (no GLB assets).

Usage:
    python examples/IPC_Solver/replay_yoyo_v5_traj.py
    python examples/IPC_Solver/replay_yoyo_v5_traj.py --seq-dir path/to/seq
    python examples/IPC_Solver/replay_yoyo_v5_traj.py --render
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs

from _replay_common import TrajectoryReplay

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SEQ_DIR = _REPO_ROOT / "DemoAssets" / "yoyo" / "v7" / "seq"


class YoyoV5Replay(TrajectoryReplay):
    name = "yoyo_v5"
    cam_pos = (1.2, -0.8, 1.7)
    cam_lookat = (0.2, 0.0, 1.0)
    cam_fov = 45

    def add_args(self, parser):
        parser.add_argument(
            "--seq-dir", type=str, default=str(_DEFAULT_SEQ_DIR),
            help="Sequence directory containing meta.json",
        )

    def load_trajectory(self) -> int:
        self._seq_dir = Path(self.args.seq_dir)
        meta_path = self._seq_dir / "meta.json"
        assert meta_path.exists(), f"meta.json not found in {self._seq_dir}"
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self._joint_names = self._meta.get("joints", {}).get("names", [])
        self._raw_joint_data = None
        joint_data_path = self._meta.get("joints", {}).get("data")
        if joint_data_path:
            p = self._seq_dir / joint_data_path
            if p.exists():
                self._raw_joint_data = np.load(str(p))

        rigid_raw = {}
        self._fem_data = {}
        for name, info in self._meta["objects"].items():
            npy_path = self._seq_dir / info["data"]
            if not npy_path.exists():
                print(f"[warn] {npy_path} not found, skipping {name}")
                continue
            arr = np.load(str(npy_path))
            if info["type"] == "rigid":
                rigid_raw[name] = arr
            else:
                self._fem_data[name] = arr

        self._rigid_data = {}
        for name, data in rigid_raw.items():
            if data.ndim == 3 and data.shape[1:] == (4, 4):
                n = data.shape[0]
                pos = data[:, :3, 3]
                xyzw = Rotation.from_matrix(data[:, :3, :3]).as_quat()
                wxyz = np.column_stack([xyzw[:, 3], xyzw[:, :3]])
                self._rigid_data[name] = np.column_stack([pos, wxyz]).astype(np.float32)
            else:
                self._rigid_data[name] = data

        n_frames = self._meta["frame_count"]
        dt = self._meta.get("dt", 0.001)
        frame_skip = self._meta.get("frame_skip", 10)
        self.fps = min(int(1.0 / dt / frame_skip), 60)
        return n_frames

    def build_scene(self, scene):
        import genesis as gs

        urdf_rel = self._meta.get("urdf", "")
        assert urdf_rel, "meta.json must specify 'urdf'"
        urdf_path = Path(urdf_rel)
        if not urdf_path.is_absolute():
            urdf_path = _REPO_ROOT / urdf_rel
        if not urdf_path.exists() and "marvin_sharpa" in urdf_rel:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                repo_id="Genesis-Intelligence/internal_assets",
                repo_type="dataset",
                allow_patterns="marvin_sharpa_description/**",
            )
            urdf_path = Path(local_dir) / "marvin_sharpa_description" / "marvin_sharpa.urdf"
        assert urdf_path.exists(), f"Robot URDF not found: {urdf_path}"

        robot_pos = self._meta.get("robot_base_pos", [0, 0, 0])
        self._robot = scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf_path), fixed=True, collision=False,
                pos=tuple(robot_pos),
            ),
            material=gs.materials.Rigid(coup_type="external_articulation"),
            vis_mode="visual",
            name="robot",
        )

        self._rigid_entities = {}
        for name in self._rigid_data:
            mesh_path = self._seq_dir / name / "mesh.obj"
            if not mesh_path.exists():
                print(f"[warn] {mesh_path} not found, skipping {name}")
                continue
            ent = scene.add_entity(
                morph=gs.morphs.Mesh(file=str(mesh_path), fixed=True, collision=False),
                material=gs.materials.Rigid(),
                name=name,
            )
            self._rigid_entities[name] = [ent]

        self._fem_entities = {}
        for name in self._fem_data:
            mesh_path = self._seq_dir / name / "mesh.obj"
            if not mesh_path.exists():
                print(f"[warn] {mesh_path} not found, skipping {name}")
                continue
            ent = scene.add_entity(
                morph=gs.morphs.Mesh(file=str(mesh_path)),
                material=gs.materials.FEM.Rope(E=1e6, rho=100.0, thickness=0.0004),
                surface=gs.surfaces.Default(color=(0.9, 0.87, 0.8, 1.0)),
                name=name,
            )
            self._fem_entities[name] = ent


    def post_build(self):
        super().post_build()
        for name, ent in self._fem_entities.items():
            data = self._fem_data.get(name)
            if data is not None:
                print(f"[v5] FEM '{name}': entity n_vertices={ent.n_vertices}, "
                      f"seq data shape={data.shape}")

    def apply_frame(self, scene, frame_idx):
        if self._joint_qpos is not None and frame_idx < len(self._joint_qpos):
            self._robot.set_qpos(self._joint_qpos[frame_idx])

        for name, entities in self._rigid_entities.items():
            if name in self._rigid_data and frame_idx < len(self._rigid_data[name]):
                pose = self._rigid_data[name][frame_idx]
                if not isinstance(entities, (list, tuple)):
                    entities = [entities]
                for ent in entities:
                    ent.set_pos(pose[:3])
                    ent.set_quat(pose[3:])

        for name, entity in self._fem_entities.items():
            if name in self._fem_data and frame_idx < len(self._fem_data[name]):
                pos = self._fem_data[name][frame_idx]
                n_ent = entity.n_vertices
                n_data = pos.shape[0]
                if n_data == n_ent:
                    entity.set_position(pos)
                elif n_data < n_ent:
                    padded = np.zeros((n_ent, 3), dtype=pos.dtype)
                    padded[:n_data] = pos
                    padded[n_data:] = pos[-1]
                    entity.set_position(padded)
                else:
                    entity.set_position(pos[:n_ent])

    def run(self) -> None:
        gs.init(backend=gs.gpu, logging_level="warning")

        self._n_frames = self.load_trajectory()
        print(f"[{self.name}] {self._n_frames} frames, {self.fps} fps")

        self._make_scene()
        self.build_scene(self._scene)

        self._cam = None
        if self.args.render:
            self._add_camera()

        self._scene.build()
        self.post_build()

        self._camera_traj = None
        if self.args.camera_traj:
            self._camera_traj = self.make_camera_traj(self.args.camera_traj)

        if self.args.render:
            self._run_render()
        else:
            self._run_interactive()


if __name__ == "__main__":
    YoyoV5Replay().run()
