"""
Replay exported cooking simulation sequences using Genesis.

Loads ``meta.json`` from a sequence directory exported by
``cook_with_teleop/scripts/replay_cook.py --export-seq``, then sets up:

- Rigid parts (pan, spatula, broccoli) with per-frame 4x4 transforms
- FEM cloth (tomato slices) with per-frame vertex positions
- FEM rope (noodles, split per strand) with per-frame vertex positions

Usage:
    python ipc_robot_cook.py --seq-dir PATH/TO/seq        # GUI playback
    python ipc_robot_cook.py --seq-dir PATH/TO/seq --no-gui
    python ipc_robot_cook.py --seq-dir PATH/TO/seq --render  # record video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SEQ = _REPO_ROOT / "DemoAssets" / "cook_with_teleop" / "seq"

VERTS_PER_NOODLE = 51


def tf_to_pos_quat(tf: np.ndarray):
    """Extract (pos, quat_wxyz) from a 4x4 homogeneous transform."""
    pos = tf[:3, 3].copy()
    R = tf[:3, :3]
    r = Rotation.from_matrix(R)
    xyzw = r.as_quat()
    quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return pos, quat


def _apply_tf_to_verts(verts: np.ndarray, tf: np.ndarray) -> np.ndarray:
    R = tf[:3, :3]
    t = tf[:3, 3]
    return (R @ verts.T).T + t


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray | None = None,
               edges: np.ndarray | None = None):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        if faces is not None:
            for tri in faces:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        if edges is not None:
            for e in edges:
                f.write(f"l {e[0]+1} {e[1]+1}\n")


def _read_obj(path: Path):
    verts, faces, edges = [], [], []
    with open(path) as f:
        for raw in f:
            if raw.startswith("v "):
                verts.append([float(x) for x in raw.split()[1:4]])
            elif raw.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in raw.split()[1:4]])
            elif raw.startswith("l "):
                idx = [int(x) - 1 for x in raw.split()[1:]]
                for a, b in zip(idx, idx[1:]):
                    edges.append([a, b])
    return (np.array(verts, dtype=np.float64),
            np.array(faces, dtype=np.int32) if faces else None,
            np.array(edges, dtype=np.int32) if edges else None)


def load_seq_data(seq_dir: Path, meta: dict):
    """Load trajectory data split by type: rigid, fem, rod."""
    rigid_data: dict[str, np.ndarray] = {}
    fem_data: dict[str, np.ndarray] = {}
    rod_data: dict[str, np.ndarray] = {}
    for name, info in meta["objects"].items():
        npy_path = seq_dir / info["data"]
        if not npy_path.exists():
            print(f"[warn] {npy_path} not found, skipping {name}")
            continue
        arr = np.load(str(npy_path))
        obj_type = info["type"]
        if obj_type == "rigid":
            rigid_data[name] = arr
        elif obj_type == "rod":
            rod_data[name] = arr
        else:
            fem_data[name] = arr
    return rigid_data, fem_data, rod_data


ENTITY_COLORS = {
    "pan": (0.6, 0.6, 0.65, 1.0),
    "spatula": (0.5, 0.5, 0.55, 1.0),
}
BROC_COLOR = (0.2, 0.6, 0.15, 1.0)
TOMATO_COLOR = (0.85, 0.15, 0.1, 1.0)
NOODLE_COLOR = (0.92, 0.86, 0.55, 1.0)


def _prepare_rigid_world_mesh(seq_dir: Path, name: str, tf0: np.ndarray) -> Path:
    """Transform a local-space mesh.obj into frame-0 world positions."""
    world_path = seq_dir / name / "mesh_world.obj"
    if world_path.exists():
        return world_path
    local_verts, local_faces, _ = _read_obj(seq_dir / name / "mesh.obj")
    world_verts = _apply_tf_to_verts(local_verts, tf0)
    _write_obj(world_path, world_verts, faces=local_faces)
    print(f"[mesh] {name}: wrote world-space mesh ({len(local_verts)} verts)")
    return world_path


def _split_noodle_meshes(seq_dir: Path, n_noodles: int) -> list[Path]:
    """Split the combined noodle line mesh into individual OBJ files."""
    noodle_dir = seq_dir / "noodles"
    first_path = noodle_dir / "noodle_0.obj"
    if first_path.exists():
        return [noodle_dir / f"noodle_{i}.obj" for i in range(n_noodles)]

    verts, _, edges = _read_obj(noodle_dir / "mesh.obj")
    vpn = VERTS_PER_NOODLE
    paths = []
    for i in range(n_noodles):
        v_start = i * vpn
        v_end = v_start + vpn
        nv = verts[v_start:v_end]
        ne = np.array([[j, j + 1] for j in range(vpn - 1)], dtype=np.int32)
        p = noodle_dir / f"noodle_{i}.obj"
        _write_obj(p, nv, edges=ne)
        paths.append(p)
    print(f"[mesh] split {n_noodles} noodle meshes ({vpn} verts each)")
    return paths


def run_gui(seq_dir: Path, meta: dict, render_output: str | None = None) -> None:
    import genesis as gs

    use_render = render_output is not None
    gs.init(backend=gs.gpu if use_render else gs.cpu, logging_level="warning")

    frame_count = meta["frame_count"]
    rigid_data, fem_data, rod_data = load_seq_data(seq_dir, meta)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=meta.get("dt", 0.005),
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -0.6, 1.3),
            camera_lookat=(0.5, 0.2, 0.85),
            camera_fov=45,
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.4, 0.4, 0.45),
        ),
        show_viewer=not use_render,
    )

    # ---- Rigid entities (pan, spatula, broccoli) ----
    # Mesh pre-transformed to frame-0 world space; playback uses T_i @ T_0^{-1}.
    rigid_entities: dict[str, object] = {}
    rigid_T0_inv: dict[str, np.ndarray] = {}
    for name in rigid_data:
        mesh_path = seq_dir / name / "mesh.obj"
        if not mesh_path.exists():
            continue
        T0 = rigid_data[name][0]
        world_mesh = _prepare_rigid_world_mesh(seq_dir, name, T0)
        rigid_T0_inv[name] = np.linalg.inv(T0)

        color = ENTITY_COLORS.get(name)
        if color is None:
            color = BROC_COLOR if name.startswith("broc") else (0.7, 0.7, 0.7, 1.0)
        ent = scene.add_entity(
            morph=gs.morphs.Mesh(file=str(world_mesh), fixed=True, collision=False),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=color),
            name=name,
        )
        rigid_entities[name] = ent

    # ---- FEM cloth entities (tomato slices) ----
    # Mesh uses frame-0 vertex positions; set_position() per frame.
    fem_entities: dict[str, object] = {}
    for name in fem_data:
        mesh_path = seq_dir / name / "mesh.obj"
        if not mesh_path.exists():
            continue
        color = TOMATO_COLOR if name.startswith("tomato") else (0.7, 0.7, 0.7, 1.0)
        ent = scene.add_entity(
            morph=gs.morphs.Mesh(file=str(mesh_path)),
            material=gs.materials.FEM.Cloth(E=1e4, nu=0.45, rho=1000.0),
            surface=gs.surfaces.Default(color=color),
            name=name,
        )
        fem_entities[name] = ent

    # ---- FEM rope entities (noodles, split per strand) ----
    noodle_entities: list[object] = []
    noodle_positions: np.ndarray | None = None
    n_noodles = 0
    for name in rod_data:
        noodle_positions = rod_data[name]
        n_total_verts = noodle_positions.shape[1]
        n_noodles = n_total_verts // VERTS_PER_NOODLE
        noodle_meshes = _split_noodle_meshes(seq_dir, n_noodles)
        for i, mp in enumerate(noodle_meshes):
            ent = scene.add_entity(
                morph=gs.morphs.Mesh(file=str(mp)),
                material=gs.materials.FEM.Rope(E=1e6, rho=100.0, thickness=0.001),
                surface=gs.surfaces.Default(color=NOODLE_COLOR),
                name=f"noodle_{i}",
            )
            noodle_entities.append(ent)
        print(f"[scene] {n_noodles} noodle FEM.Rope entities")
        break  # only one "noodles" rod entry

    cam = None
    if use_render:
        cam = scene.add_camera(
            res=(1920, 1080),
            pos=(0.8, -0.6, 1.2),
            lookat=(0.5, 0.0, 0.85),
            up=(0.0, 0.0, 1.0),
            fov=45,
            spp=128,
        )

    scene.build()

    vpn = VERTS_PER_NOODLE

    def _apply_frame(i: int) -> None:
        if i < 0 or i >= frame_count:
            return
        for name, ent in rigid_entities.items():
            if name in rigid_data and i < rigid_data[name].shape[0]:
                T_rel = rigid_data[name][i] @ rigid_T0_inv[name]
                pos, quat = tf_to_pos_quat(T_rel)
                ent.set_pos(pos)
                ent.set_quat(quat)
        for name, ent in fem_entities.items():
            if name in fem_data and i < fem_data[name].shape[0]:
                ent.set_position(fem_data[name][i])
        if noodle_positions is not None and i < noodle_positions.shape[0]:
            frame_pos = noodle_positions[i]
            for ni, ent in enumerate(noodle_entities):
                v_start = ni * vpn
                ent.set_position(frame_pos[v_start: v_start + vpn])

    if use_render:
        import imageio

        output_path = Path(render_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cam.start_recording()
        frames_rgb = []
        for i in range(frame_count):
            _apply_frame(i)
            scene._visualizer.update_visual_states(force_render=True)
            rgb_result = cam.render(rgb=True, force_render=True)
            rgb_tensor = rgb_result[0]
            rgb = rgb_tensor.cpu().numpy() if hasattr(rgb_tensor, "cpu") else np.array(rgb_tensor)
            frames_rgb.append(rgb)
            if i % 50 == 0:
                print(f"[render] Frame {i}/{frame_count}")
        replay_fps = int(1.0 / meta.get("dt", 0.005) / 10)
        fps = min(replay_fps, 60)
        writer = imageio.get_writer(str(output_path), fps=fps)
        for rgb in frames_rgb:
            writer.append_data(rgb)
        writer.close()
        cam.stop_recording()
        print(f"[render] Saved {output_path} ({frame_count} frames)")
        return

    import time as _time

    print(f"[replay] playing {frame_count} frames (loop), close viewer to exit")
    try:
        while True:
            for i in range(frame_count):
                t0 = _time.perf_counter()
                _apply_frame(i)
                t1 = _time.perf_counter()
                scene._visualizer.update_visual_states(force_render=True)
                t2 = _time.perf_counter()
                scene._visualizer.update(force=True)
                t3 = _time.perf_counter()
                if i % 50 == 0:
                    print(f"  frame {i}: apply={1000*(t1-t0):.1f}ms  "
                          f"vis_state={1000*(t2-t1):.1f}ms  "
                          f"update={1000*(t3-t2):.1f}ms  "
                          f"total={1000*(t3-t0):.1f}ms")
            print(f"[replay] loop done, looping...")
    except Exception:
        print("[replay] viewer closed, exiting.")


def run_no_gui(seq_dir: Path, meta: dict) -> None:
    frame_count = meta["frame_count"]
    objects = meta["objects"]
    print(f"[no-gui] {frame_count} frames, {len(objects)} objects")
    for name, info in objects.items():
        npy_path = seq_dir / info["data"]
        mesh_path = seq_dir / name / "mesh.obj"
        if npy_path.exists():
            arr = np.load(str(npy_path))
            print(f"  {name}: type={info['type']}  shape={arr.shape}  "
                  f"mesh={'OK' if mesh_path.exists() else 'MISS'}")
        else:
            print(f"  {name}: type={info['type']}  data=MISSING")
    print("[no-gui] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay exported cooking simulation (Genesis)."
    )
    parser.add_argument("--seq-dir", type=str, default=str(_DEFAULT_SEQ),
                        help="Sequence directory with meta.json")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--render", action="store_true",
                        help="Record video (LuisaRender)")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    meta_path = seq_dir / "meta.json"
    if not meta_path.exists():
        print(f"[error] meta.json not found in {seq_dir}")
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"[replay] {meta['frame_count']} frames from {seq_dir}")

    render_output = None
    if args.render:
        render_output = str(Path("data/ipc_demo/ipc_cook") / "cook_replay.mp4")

    if args.no_gui:
        run_no_gui(seq_dir, meta)
    else:
        run_gui(seq_dir, meta, render_output=render_output)


if __name__ == "__main__":
    main()
