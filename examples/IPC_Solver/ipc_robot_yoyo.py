"""
Replay exported yoyo simulation sequences using Genesis.

Loads ``meta.json`` from the sequence directory, sets up:
- Marvin robot (URDF) with per-frame joint positions
- Yoyo rigid parts (ball shells, bearings) with per-frame transforms
- Yoyo string (FEM rope) with per-frame vertex positions

Camera behaviour depends on the trajectory:
- **v3**: half-circle orbit around the robot to show the full teleop throw.
- **long_sleep**: static close-up facing the transparent shell so the bearing
  balls are visible spinning inside.

Usage:
    python ipc_robot_yoyo.py                                        # GUI playback (v3 trajectory)
    python ipc_robot_yoyo.py --trajectory long_sleep                # long sleeping yoyo
    python ipc_robot_yoyo.py --full                                 # full-rate long sleeping (2003 frames)
    python ipc_robot_yoyo.py --render                               # render video (LuisaRender)
    python ipc_robot_yoyo.py --render --nyx                         # render video (Nyx)
    python ipc_robot_yoyo.py --render --trajectory long_sleep       # render long sleep trajectory
    python ipc_robot_yoyo.py --render --start-frame 500             # continue render from frame 500
    python ipc_robot_yoyo.py --no-gui                               # headless verification
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from _yoyo_common import (
    DEFAULT_SEQ_DIR,
    INTERNAL_OPACITY_TRANSPARENT,
    REPLAY_CAMERA_FOV,
    REPLAY_CAMERA_LOOKAT,
    REPLAY_CAMERA_POS,
    REPLAY_CAMERA_UP,
    SHELL_OPACITY_TRANSPARENT,
    TRAJECTORIES,
    YOYO_ASSETS_DIR,
    load_seq_data,
    make_raytracer,
    resolve_mesh_path,
    tf_to_pos_quat,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Camera orbit parameters (v3) ──
# 120° orbit around the robot to observe the full teleop throw.
ORBIT_CENTER = np.array([0.20, 0.0, -0.05])
ORBIT_RADIUS = 0.85
ORBIT_HEIGHT = 0.30
ORBIT_ANGLE_START = math.radians(-60)  # right-front of robot
ORBIT_ANGLE_END = math.radians(60)  # left-front of robot

# ── Close-up parameters (long_sleep) ──
# Static side-on view facing the transparent shell.
# The yoyo spin axis is ≈ (0, -1, 0) in world frame, so the transparent
# bottom shell faces -Y.  Camera placed on the -Y side looking +Y.
CLOSEUP_DISTANCE = 0.02  # distance from yoyo centre
CLOSEUP_FOV = 40


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _find_asset(name):
    p = YOYO_ASSETS_DIR / f"{name}.glb"
    return str(p) if p.exists() else None


def run_gui(
    seq_dir: Path,
    meta: dict,
    render_output: str | None = None,
    use_nyx: bool = False,
    save_frames: bool = False,
    start_frame: int = 0,
) -> None:
    import genesis as gs

    use_render = render_output is not None
    is_long_sleep = "long_sleep" in str(seq_dir)
    # long_sleep always uses LuisaRender (Nyx transparency doesn't work well)
    if is_long_sleep and use_nyx:
        print("[render] long_sleep: forcing LuisaRender (Nyx alpha unsupported)")
        use_nyx = False
    gs.init(backend=gs.gpu if use_render else gs.cpu, logging_level="warning")

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
    if use_render and not use_nyx:
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
        show_viewer=not use_render,
        **renderer_kwargs,
    )

    # Robot
    assert urdf_rel, "meta.json must specify 'urdf'"
    urdf_path = _REPO_ROOT / urdf_rel
    assert urdf_path.exists(), f"Robot URDF not found: {urdf_path}"
    robot = scene.add_entity(
        gs.morphs.URDF(file=str(urdf_path), fixed=True, collision=False),
        material=gs.materials.Rigid(coup_type="ipc_only"),
        name="robot",
    )

    # Rigid yoyo parts — split yoyo_ball into shells, rings, internals
    rigid_entities: dict[str, object] = {}
    ball_entities: list[object] = []  # entities driven by yoyo_ball transform

    def _add_ball_part(asset_name, entity_name, opacity=None):
        """Add a yoyo ball sub-part. opacity=None means fully opaque."""
        # For Nyx: use *_transparent.glb (alphaMode=BLEND baked in), no surface
        # override so Nyx's GLTF importer reads the alpha material directly.
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
            # v3: fully opaque yoyo (overview orbit).
            # long_sleep: bottom shell + ring transparent to see bearings spinning.
            shell_alpha = SHELL_OPACITY_TRANSPARENT if is_long_sleep else None
            internal_alpha = INTERNAL_OPACITY_TRANSPARENT if is_long_sleep else None
            _add_ball_part("yoyo-top_shell", "yoyo_ball_top", opacity=shell_alpha)
            _add_ball_part("yoyo-bottom_shell", "yoyo_ball_bottom", opacity=shell_alpha)
            _add_ball_part("yoyo-top_ring", "yoyo_ball_top_ring", opacity=shell_alpha)
            _add_ball_part("yoyo-bottom_ring", "yoyo_ball_bottom_ring", opacity=shell_alpha)
            _add_ball_part("yoyo-axle", "yoyo_ball_axle", opacity=internal_alpha)
            _add_ball_part("yoyo-hub", "yoyo_ball_hub", opacity=internal_alpha)
        else:
            mesh_path = resolve_mesh_path(name, seq_dir)
            if mesh_path is None:
                continue
            # bearing_outer uses internal opacity in long_sleep
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

    # Initial camera pose (used for scene creation; overridden per-frame)
    init_cam_pos = REPLAY_CAMERA_POS
    init_cam_lookat = REPLAY_CAMERA_LOOKAT
    init_cam_fov = CLOSEUP_FOV if is_long_sleep else REPLAY_CAMERA_FOV

    cam = None
    if use_render and use_nyx:
        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
        import gs_nyx.nyx_py_renderer as npr
        import gs_nyx.nyx_py_sdk as ap

        env_map = ap.EnvironmentMapAsset()
        env_map.texture = str((_REPO_ROOT / "DemoAssets/textures/san_carlos_left_marvin_modified.exr").resolve())
        env_map.rotation = 0.0
        env_map.multiplier = 1.0

        cam = scene.add_sensor(
            NyxCameraOptions(
                res=(1920, 1080),
                pos=init_cam_pos,
                lookat=init_cam_lookat,
                up=REPLAY_CAMERA_UP,
                fov=init_cam_fov,
                near=0.005,
                far=100.0,
                spp=256,
                denoise=True,
                render_mode=npr.ERenderMode.RefPathTracer,
                env_maps=(env_map,),
            )
        )
    elif use_render:
        cam = scene.add_camera(
            res=(1920, 1080),
            pos=init_cam_pos,
            lookat=init_cam_lookat,
            up=REPLAY_CAMERA_UP,
            fov=init_cam_fov,
            spp=256,
        )

    scene.build(n_envs=1 if use_nyx else 0)

    # Save initial camera transform for seam verification (LuisaRender only)
    if use_render and not use_nyx:
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

    def _get_ball_pos(i):
        """Get yoyo ball center at frame i."""
        if "yoyo_ball" in rigid_data and i < rigid_data["yoyo_ball"].shape[0]:
            return rigid_data["yoyo_ball"][i][:3, 3].copy()
        return np.array([0.256, 0.008, -0.056])

    def _camera_pose_v3(frac):
        """Half-circle orbit around the robot."""
        angle = ORBIT_ANGLE_START + (ORBIT_ANGLE_END - ORBIT_ANGLE_START) * _smooth(frac)
        cam_pos = np.array(
            [
                ORBIT_CENTER[0] + ORBIT_RADIUS * math.cos(angle),
                ORBIT_CENTER[1] + ORBIT_RADIUS * math.sin(angle),
                ORBIT_HEIGHT,
            ]
        )
        return cam_pos, ORBIT_CENTER.copy()

    def _camera_pose_long_sleep(i):
        """Static close-up facing the transparent shell (−Y side)."""
        ball_pos = _get_ball_pos(i)
        cam_pos = ball_pos + np.array([0.0, -CLOSEUP_DISTANCE, 0.0])
        return cam_pos, ball_pos.copy()

    def _apply_frame(i: int) -> None:
        if i < 0 or i >= frame_count:
            return
        if robot is not None and joints_data is not None and i < joints_data.shape[0]:
            qpos = robot.get_qpos()
            if qpos.ndim == 2:
                # Batched (n_envs > 0): index with [env, qs]
                for j, qi in enumerate(qs_idx_map):
                    if qi >= 0:
                        qpos[0, qi] = joints_data[i, j]
            else:
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
        scene._visualizer.update_visual_states(force_render=True)

    # Render mode: record video with camera tracking
    if use_render:
        import imageio

        output_path = Path(render_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = None
        if save_frames:
            frames_dir = output_path.parent / (output_path.stem + "_frames")
            frames_dir.mkdir(parents=True, exist_ok=True)

        if not use_nyx:
            cam.start_recording()

        frames_rgb = []

        if start_frame > 0:
            print(f"[render] Skipping to frame {start_frame}")
            _apply_frame(start_frame - 1)

        for i in range(start_frame, frame_count):
            _apply_frame(i)

            # Compute camera pose for this frame
            if is_long_sleep:
                cam_pos, cam_lookat = _camera_pose_long_sleep(i)
            else:
                cam_pos, cam_lookat = _camera_pose_v3(i / max(frame_count - 1, 1))

            if use_nyx:
                cam.update_camera_pose(pos=tuple(cam_pos), lookat=tuple(cam_lookat), up=REPLAY_CAMERA_UP)
                data = cam.read()
                rgb = data.rgb[0].cpu().numpy()
            else:
                cam.set_pose(pos=tuple(cam_pos), lookat=tuple(cam_lookat), up=REPLAY_CAMERA_UP)
                rgb_result = cam.render(rgb=True, force_render=True)
                # cam.render returns (rgb_arr, depth_arr, seg_arr, normal_arr)
                rgb_tensor = rgb_result[0]
                if hasattr(rgb_tensor, "cpu"):
                    rgb = rgb_tensor.cpu().numpy()
                else:
                    rgb = np.array(rgb_tensor)

            if save_frames:
                imageio.imwrite(str(frames_dir / f"{i:05d}.png"), rgb)
            frames_rgb.append(rgb)

            if i % 50 == 0:
                print(f"[render] Frame {i}/{frame_count}")

        # Assemble video
        replay_fps = int(1.0 / meta.get("dt", 0.001) / 10)
        fps = min(replay_fps, 60)
        writer = imageio.get_writer(str(output_path), fps=fps)
        for rgb in frames_rgb:
            writer.append_data(rgb)
        writer.close()

        if not use_nyx:
            cam.stop_recording()

        print(f"[render] Saved {output_path} ({frame_count} frames)")
        if save_frames:
            print(f"[render] Individual frames in {frames_dir}/")
        return

    # Interactive replay
    if start_frame > 0:
        print(f"[replay] Skipping to frame {start_frame}")
        _apply_frame(start_frame - 1)
    for i in range(start_frame, frame_count):
        _apply_frame(i)
        scene._visualizer.update_visual_states(force_render=True)
    print(f"[replay] finished {frame_count - start_frame} frames (from {start_frame})")


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
    _OUTPUT_DIR = Path("data/ipc_demo/ipc_yoyo")

    parser = argparse.ArgumentParser(description="Replay exported yoyo simulation (Genesis).")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--render", action="store_true", help="Record video (LuisaRender or Nyx)")
    parser.add_argument("--nyx", action="store_true", help="Use Nyx renderer instead of LuisaRender")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="v3",
        choices=list(TRAJECTORIES.keys()),
        help=f"Trajectory to replay. Available: {list(TRAJECTORIES.keys())}. Default: v3.",
    )
    parser.add_argument("--seq-dir", type=str, default=None, help="Override sequence directory.")
    parser.add_argument("--save-frames", action="store_true", help="Save each rendered frame as PNG.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start replay/render from this frame index.")
    args = parser.parse_args()

    if args.seq_dir:
        seq_dir = Path(args.seq_dir)
    else:
        seq_dir = TRAJECTORIES[args.trajectory]
    meta_path = seq_dir / "meta.json"
    if not meta_path.exists():
        print(f"[error] meta.json not found in {seq_dir}")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"[replay] {meta['frame_count']} frames from {seq_dir}")

    render_output = None
    if args.render:
        renderer_name = "nyx" if args.nyx else "luisa"
        render_output = str(_OUTPUT_DIR / f"yoyo_{args.trajectory}_{renderer_name}.mp4")

    if args.no_gui:
        run_no_gui(seq_dir, meta)
    else:
        run_gui(
            seq_dir,
            meta,
            render_output=render_output,
            use_nyx=args.nyx,
            save_frames=args.save_frames,
            start_frame=args.start_frame,
        )


if __name__ == "__main__":
    main()
