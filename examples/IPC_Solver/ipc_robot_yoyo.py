"""
Replay exported yoyo simulation sequences using Genesis.

Loads ``meta.json`` from the sequence directory, sets up:
- Marvin robot (URDF) with per-frame joint positions
- Yoyo rigid parts (ball, bearings) with per-frame transforms
- Yoyo string (FEM rope) with per-frame vertex positions

All collisions are disabled — this is purely visual replay.

Usage:
    python ipc_robot_yoyo.py                # GUI playback
    python ipc_robot_yoyo.py --no-gui       # headless verification
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SEQ_DIR = _REPO_ROOT / "IPC-Samples" / "python" / "Yoyo" / "results" / "v3" / "seq"


def _tf_to_pos_quat(tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position (3,) and quaternion (w,x,y,z) from a 4x4 transform."""
    from scipy.spatial.transform import Rotation

    pos = tf[:3, 3].copy()
    R = tf[:3, :3]
    r = Rotation.from_matrix(R)
    xyzw = r.as_quat()
    quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return pos, quat


def _parse_obj_edges(path: Path) -> np.ndarray | None:
    """Parse line elements from OBJ for string visualization."""
    edges: list[list[int]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if parts and parts[0] == "l":
                idx = [int(p) - 1 for p in parts[1:]]
                for k in range(len(idx) - 1):
                    edges.append([idx[k], idx[k + 1]])
    return np.array(edges, dtype=np.int32) if edges else None


def run_gui(seq_dir: Path, meta: dict) -> None:
    import genesis as gs

    gs.init(backend=gs.cpu, logging_level="warning")

    frame_count = meta["frame_count"]
    objects = meta["objects"]
    joint_meta = meta.get("joints", {})
    joint_names = joint_meta.get("names", [])
    urdf_rel = meta.get("urdf", "")

    joints_data = None
    if joint_meta.get("data"):
        joints_path = seq_dir / joint_meta["data"]
        if joints_path.exists():
            joints_data = np.load(str(joints_path))

    rigid_data: dict[str, np.ndarray] = {}
    fem_data: dict[str, np.ndarray] = {}
    for name, info in objects.items():
        npy_path = seq_dir / info["data"]
        if not npy_path.exists():
            print(f"[warn] {npy_path} not found, skipping {name}")
            continue
        arr = np.load(str(npy_path))
        if info["type"] == "rigid":
            rigid_data[name] = arr
        else:
            fem_data[name] = arr

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=meta.get("dt", 0.001),
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.5, 0.5),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=45,
        ),
        show_viewer=True,
    )

    robot = None
    if urdf_rel:
        urdf_path = _REPO_ROOT / urdf_rel
        if urdf_path.exists():
            robot = scene.add_entity(
                gs.morphs.URDF(
                    file=str(urdf_path),
                    fixed=True,
                    collision=False,
                ),
                material=gs.materials.Rigid(
                    coup_type="ipc_only",
                ),
                name="robot",
            )

    rigid_entities: dict[str, object] = {}
    for name in rigid_data:
        mesh_path = seq_dir / name / "mesh.obj"
        if not mesh_path.exists():
            print(f"[warn] {mesh_path} not found, skipping {name}")
            continue
        ent = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(mesh_path),
                fixed=True,
                collision=False,
            ),
            material=gs.materials.Rigid(),
            name=name,
        )
        rigid_entities[name] = ent

    string_entity = None
    string_name = None
    for name in fem_data:
        mesh_path = seq_dir / name / "mesh.obj"
        if not mesh_path.exists():
            print(f"[warn] {mesh_path} not found, skipping {name}")
            continue
        string_name = name
        string_entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(mesh_path),
            ),
            material=gs.materials.FEM.Rope(
                E=1e6,
                rho=100.0,
                thickness=0.0004,
            ),
            name=name,
        )
        break

    scene.build()

    qs_idx_map: list[int] = []
    if robot is not None:
        for jname in joint_names:
            try:
                qs_idx_map.append(robot.get_joint(jname).qs_idx_local[0])
            except Exception:
                qs_idx_map.append(-1)
        matched = sum(1 for x in qs_idx_map if x >= 0)
        print(f"[replay] Joint mapping: {matched}/{len(joint_names)} matched")
        if matched == 0:
            all_joints = [j.name for j in robot.joints]
            print(f"[replay] Robot joints: {all_joints}")
            print(f"[replay] Seq joints: {joint_names}")

    def _apply_frame(i: int) -> None:
        if i < 0 or i >= frame_count:
            return

        if robot is not None and joints_data is not None and i < joints_data.shape[0]:
            qpos = robot.get_qpos()
            for j, qi in enumerate(qs_idx_map):
                if qi >= 0:
                    qpos[qi] = joints_data[i, j]
            robot.set_qpos(qpos, zero_velocity=True)

        for name, ent in rigid_entities.items():
            if name in rigid_data and i < rigid_data[name].shape[0]:
                tf = rigid_data[name][i]
                pos, quat = _tf_to_pos_quat(tf)
                ent.set_pos(pos)
                ent.set_quat(quat)

        if string_entity is not None and string_name in fem_data:
            if i < fem_data[string_name].shape[0]:
                string_entity.set_position(fem_data[string_name][i])

        scene.step()

    _apply_frame(0)

    for i in range(1, frame_count):
        _apply_frame(i)

    print(f"[replay] finished {frame_count} frames")


def run_no_gui(seq_dir: Path, meta: dict) -> None:
    frame_count = meta["frame_count"]
    objects = meta["objects"]
    joint_meta = meta.get("joints", {})

    print(f"[no-gui] {frame_count} frames, {len(objects)} objects")

    for name, info in objects.items():
        npy_path = seq_dir / info["data"]
        mesh_path = seq_dir / name / "mesh.obj"
        npy_ok = npy_path.exists()
        mesh_ok = mesh_path.exists()
        if npy_ok:
            arr = np.load(str(npy_path))
            print(f"  {name}: type={info['type']}  shape={arr.shape}  mesh={'OK' if mesh_ok else 'MISS'}")
        else:
            print(f"  {name}: type={info['type']}  data=MISSING  mesh={'OK' if mesh_ok else 'MISS'}")

    if joint_meta.get("data"):
        jp = seq_dir / joint_meta["data"]
        if jp.exists():
            jarr = np.load(str(jp))
            print(f"  joints: shape={jarr.shape}  names={len(joint_meta.get('names', []))}")
        else:
            print(f"  joints: MISSING")

    print(f"  urdf: {meta.get('urdf', '(none)')}")
    print("[no-gui] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay exported yoyo simulation (Genesis).")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--seq-dir", type=str, default=str(_DEFAULT_SEQ_DIR))
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    meta_path = seq_dir / "meta.json"

    if not meta_path.exists():
        print(f"[error] meta.json not found in {seq_dir}")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"[replay] {meta['frame_count']} frames from {seq_dir}")

    if args.no_gui:
        run_no_gui(seq_dir, meta)
    else:
        run_gui(seq_dir, meta)


if __name__ == "__main__":
    main()
