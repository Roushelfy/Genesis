"""
Replay exported yoyo simulation sequences using Genesis.

Loads ``meta.json`` from the sequence directory, sets up:
- Marvin robot (URDF) with per-frame joint positions
- Yoyo rigid parts (ball shells, bearings) with per-frame transforms
- Yoyo string (FEM rope) with per-frame vertex positions

One shell is transparent to reveal the internal structure. During the throw,
the camera tracks the yoyo and orbits to view the transparent side, then
pulls back to the full view.

Usage:
    python ipc_robot_yoyo.py                                        # GUI playback
    python ipc_robot_yoyo.py --no-gui                               # headless verification
    python ipc_robot_yoyo.py --render data/ipc_demo/ipc_yoyo/yoyo_replay.mp4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from _yoyo_common import (
    DEFAULT_SEQ_DIR,
    REPLAY_CAMERA_FOV,
    REPLAY_CAMERA_LOOKAT,
    REPLAY_CAMERA_POS,
    REPLAY_CAMERA_UP,
    END_CAMERA_POS,
    SHELL_OPACITY_TRANSPARENT,
    YOYO_ASSETS_DIR,
    load_seq_data,
    make_raytracer,
    resolve_mesh_path,
    tf_to_pos_quat,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Camera tracking parameters ──

# Camera starts at REPLAY_CAMERA_POS (shared with ipc_show_yoyo).
# During the throw (mid-frames), it moves close to the yoyo and orbits
# to view the transparent shell, then pulls back to the original pose.
TRACK_START_FRAC = 0.00  # fraction of frames where tracking begins
TRACK_PEAK_FRAC = 0.45  # fraction where camera is closest (behind yoyo)
TRACK_END_FRAC = 0.75  # fraction where camera returns to original pose
TRACK_DISTANCE = 0.10  # close-up distance from yoyo center
TRACK_ELEVATION = math.radians(20)  # elevation above yoyo during tracking


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _find_asset(name):
    p = YOYO_ASSETS_DIR / f"{name}.glb"
    return str(p) if p.exists() else None


def run_gui(seq_dir: Path, meta: dict, render_output: str | None = None) -> None:
    import genesis as gs

    use_raytracer = render_output is not None
    gs.init(backend=gs.gpu if use_raytracer else gs.cpu, logging_level="warning")

    frame_count = meta["frame_count"]
    joint_meta = meta.get("joints", {})
    joint_names = joint_meta.get("names", [])
    urdf_rel = meta.get("urdf", "")

    joints_data = None
    if joint_meta.get("data"):
        joints_path = seq_dir / joint_meta["data"]
        if joints_path.exists():
            joints_data = np.load(str(joints_path))

    rigid_data, fem_data = load_seq_data(seq_dir, meta)

    renderer_kwargs = {}
    if use_raytracer:
        renderer_kwargs["renderer"] = make_raytracer()

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
        vis_options=gs.options.VisOptions(
            ambient_light=(0.3, 0.3, 0.35),
        ),
        show_viewer=not use_raytracer,
        **renderer_kwargs,
    )

    # Robot
    robot = None
    if urdf_rel:
        urdf_path = _REPO_ROOT / urdf_rel
        if urdf_path.exists():
            robot = scene.add_entity(
                gs.morphs.URDF(file=str(urdf_path), fixed=True, collision=False),
                material=gs.materials.Rigid(coup_type="ipc_only"),
                name="robot",
            )

    # Metallic/roughness are baked into the GLB assets.
    # Only opacity is set at runtime for the transparent side.
    transparent_srf = gs.surfaces.BSDF(
        opacity_texture=gs.textures.ColorTexture(color=(SHELL_OPACITY_TRANSPARENT,)),
    )

    # Rigid yoyo parts — split yoyo_ball into shells, rings, internals
    rigid_entities: dict[str, object] = {}
    ball_entities: list[object] = []  # entities driven by yoyo_ball transform

    def _add_ball_part(asset_name, entity_name, surface=None):
        p = _find_asset(asset_name)
        if p is None:
            return None
        ent = scene.add_entity(
            morph=gs.morphs.Mesh(file=p, fixed=True, collision=False),
            material=gs.materials.Rigid(),
            surface=surface,
            name=entity_name,
        )
        ball_entities.append(ent)
        rigid_entities[entity_name] = ent
        return ent

    for name in rigid_data:
        if name == "yoyo_ball":
            # Spin axis is world -X. Camera starts on +X side, orbits to +Y side.
            # Top shell faces -X (away from camera initially) — make it transparent
            # so camera can see internals when it orbits to the +Y viewing angle.
            _add_ball_part("yoyo-top_shell", "yoyo_ball_top")
            _add_ball_part("yoyo-bottom_shell", "yoyo_ball_bottom", transparent_srf)  # opaque (GLB material)
            # Top ring transparent (same side as transparent shell)
            _add_ball_part("yoyo-top_ring", "yoyo_ball_top_ring")
            _add_ball_part("yoyo-bottom_ring", "yoyo_ball_bottom_ring", transparent_srf)  # opaque
            # Axle + hub: opaque, use GLB materials
            _add_ball_part("yoyo-axle", "yoyo_ball_axle")
            _add_ball_part("yoyo-hub", "yoyo_ball_hub")
        else:
            mesh_path = resolve_mesh_path(name, seq_dir)
            if mesh_path is None:
                continue
            ent = scene.add_entity(
                morph=gs.morphs.Mesh(file=str(mesh_path), fixed=True, collision=False),
                material=gs.materials.Rigid(),
                name=name,
            )
            rigid_entities[name] = ent

    # String (FEM rope)
    string_entity = None
    string_name = None
    for name in fem_data:
        mesh_path = seq_dir / name / "mesh.obj"
        if not mesh_path.exists():
            continue
        string_name = name
        string_entity = scene.add_entity(
            morph=gs.morphs.Mesh(file=str(mesh_path)),
            material=gs.materials.FEM.Rope(E=1e6, rho=100.0, thickness=0.0004),
            surface=gs.surfaces.Default(color=(0.9, 0.87, 0.8, 1.0)),
            name=name,
        )
        break

    cam = None
    if use_raytracer:
        cam = scene.add_camera(
            res=(1920, 1080),
            pos=REPLAY_CAMERA_POS,
            lookat=REPLAY_CAMERA_LOOKAT,
            up=REPLAY_CAMERA_UP,
            fov=REPLAY_CAMERA_FOV,
            spp=256,
        )

    scene.build()

    # Save initial camera transform for seam verification
    if use_raytracer:
        initial_transform = cam.transform
        seam_path = Path(render_output).parent / "_replay_initial_cam_transform.npy"
        np.save(str(seam_path), initial_transform)
        print(f"[replay] Saved camera transform to {seam_path.name}")

    # Joint mapping for robot
    qs_idx_map: list[int] = []
    if robot is not None:
        for jname in joint_names:
            try:
                qs_idx_map.append(robot.get_joint(jname).qs_idx_local[0])
            except Exception:
                qs_idx_map.append(-1)
        matched = sum(1 for x in qs_idx_map if x >= 0)
        print(f"[replay] Joint mapping: {matched}/{len(joint_names)} matched")

    _replay_pos = np.array(REPLAY_CAMERA_POS)
    _replay_lookat = np.array(REPLAY_CAMERA_LOOKAT)
    _replay_up = np.array(REPLAY_CAMERA_UP)
    _end_pos = np.array(END_CAMERA_POS)
    _end_lookat = np.array(REPLAY_CAMERA_LOOKAT)

    def _get_ball_pos(i):
        """Get yoyo ball center at frame i."""
        if "yoyo_ball" in rigid_data and i < rigid_data["yoyo_ball"].shape[0]:
            return rigid_data["yoyo_ball"][i][:3, 3].copy()
        return np.array([0.256, 0.008, -0.056])

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
            # Ball-derived entities use the yoyo_ball transform
            data_name = "yoyo_ball" if name.startswith("yoyo_ball_") or name.startswith("yoyo-") else name
            if data_name in rigid_data and i < rigid_data[data_name].shape[0]:
                pos, quat = tf_to_pos_quat(rigid_data[data_name][i])
                ent.set_pos(pos)
                ent.set_quat(quat)
        if string_entity is not None and string_name in fem_data:
            if i < fem_data[string_name].shape[0]:
                string_entity.set_position(fem_data[string_name][i])
        scene.step()

    # Render mode: record video with camera tracking
    if use_raytracer:
        output_path = Path(render_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cam.start_recording()

        i_track_start = int(frame_count * TRACK_START_FRAC)
        i_track_peak = int(frame_count * TRACK_PEAK_FRAC)
        i_track_end = int(frame_count * TRACK_END_FRAC)

        for i in range(frame_count):
            _apply_frame(i)

            # Camera animation:
            # Phase 1 (start → TRACK_START): static at replay pose
            # Phase 2 (TRACK_START → TRACK_PEAK): move close, orbit to +Y side
            #          (looking from +Y toward -Y through transparent top shell)
            # Phase 3 (TRACK_PEAK → TRACK_END): pull out to end pose
            # Phase 4 (TRACK_END → end): static at end pose
            frac = i / frame_count
            ball_pos = _get_ball_pos(i)

            # Close-up position: on the +Y side of yoyo, looking through
            # transparent top shell (faces -X) from the +Y direction
            close_pos = ball_pos + np.array(
                [
                    0.02,
                    TRACK_DISTANCE,
                    TRACK_DISTANCE * math.sin(TRACK_ELEVATION),
                ]
            )

            if frac < TRACK_START_FRAC:
                cam_pos = _replay_pos.copy()
                cam_lookat = _replay_lookat.copy()
            elif frac < TRACK_PEAK_FRAC:
                # Move in: replay pose → close-up at +Y
                t = _smooth((frac - TRACK_START_FRAC) / (TRACK_PEAK_FRAC - TRACK_START_FRAC))
                cam_pos = _lerp(_replay_pos, close_pos, t)
                cam_lookat = _lerp(_replay_lookat, ball_pos, t)
            elif frac < TRACK_END_FRAC:
                # Pull out: close-up → end pose
                t = _smooth((frac - TRACK_PEAK_FRAC) / (TRACK_END_FRAC - TRACK_PEAK_FRAC))
                cam_pos = _lerp(close_pos, _end_pos, t)
                cam_lookat = _lerp(ball_pos, _end_lookat, t)
            else:
                cam_pos = _end_pos.copy()
                cam_lookat = _end_lookat.copy()

            cam.set_pose(pos=tuple(cam_pos), lookat=tuple(cam_lookat), up=REPLAY_CAMERA_UP)

            cam.render(rgb=True)
            if i % 50 == 0:
                print(f"[render] Frame {i}/{frame_count}")

        replay_fps = int(1.0 / meta.get("dt", 0.001) / 10)
        cam.stop_recording(save_to_filename=str(output_path), fps=min(replay_fps, 60))
        print(f"[render] Saved video {output_path} ({frame_count} frames)")
        return

    # Interactive replay
    for i in range(frame_count):
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
        if npy_path.exists():
            arr = np.load(str(npy_path))
            print(f"  {name}: type={info['type']}  shape={arr.shape}  mesh={'OK' if mesh_path.exists() else 'MISS'}")
        else:
            print(f"  {name}: type={info['type']}  data=MISSING")

    if joint_meta.get("data"):
        jp = seq_dir / joint_meta["data"]
        if jp.exists():
            jarr = np.load(str(jp))
            print(f"  joints: shape={jarr.shape}  names={len(joint_meta.get('names', []))}")
    print(f"  urdf: {meta.get('urdf', '(none)')}")
    print("[no-gui] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay exported yoyo simulation (Genesis).")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument(
        "--render",
        type=str,
        default=None,
        metavar="FILE",
        help="Record video with LuisaRender (e.g. data/ipc_demo/ipc_yoyo/yoyo_replay.mp4)",
    )
    parser.add_argument("--seq-dir", type=str, default=str(DEFAULT_SEQ_DIR))
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
        run_gui(seq_dir, meta, render_output=args.render)


if __name__ == "__main__":
    main()
