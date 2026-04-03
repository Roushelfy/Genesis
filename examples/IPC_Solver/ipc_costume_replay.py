"""
Replay exported G1 kimono dancing simulations using Genesis.

Loads USD (``.usdc``) trajectories containing:
- Rigid robot link transforms (42 links, per-frame 4x4 matrices)
- Deformable cloth vertex positions (kimono pieces, per-frame)

The robot is loaded as a URDF entity.  Per-frame joint angles (``qpos``)
are computed from the USD world transforms via inverse kinematics on the
URDF kinematic chain.

Available trajectories:
  d0_lower, d0_no_lower, d1_no_lower, d2_lower, d2_no_lower,
  d4_lower, d4_no_lower, d5_lower, d5_no_lower

Usage:
    python ipc_costume_replay.py                              # GUI (d0_no_lower)
    python ipc_costume_replay.py --trajectory d5_lower        # different trajectory
    python ipc_costume_replay.py --render                     # render video (LuisaRender)
    python ipc_costume_replay.py --render --nyx               # render video (Nyx)
    python ipc_costume_replay.py --start-frame 100            # start from frame 100
    python ipc_costume_replay.py --no-gui                     # headless verification
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACK_DIR = _REPO_ROOT / "DemoAssets" / "costume" / "results" / "dancing_export_pack"
_KIMONO_DIR = _REPO_ROOT / "DemoAssets" / "kimono"

AVAILABLE_TRAJECTORIES = [
    "d0_lower",
    "d0_no_lower",
    "d1_no_lower",
    "d2_lower",
    "d2_no_lower",
    "d4_lower",
    "d4_no_lower",
    "d5_lower",
    "d5_no_lower",
]

URDF_PATH = str(_REPO_ROOT / "DemoAssets" / "g1_robot" / "assets" / "g1_29dof_rev_1_0.urdf")

# Cloth PBR materials from the Sketchfab kimono GLTF.
# (base_color_rgb, roughness)
CLOTH_MATERIALS = {
    "kimono_outer": ((0.020, 0.031, 0.040), 0.82),
    "kimono_inner_upper": ((0.080, 0.112, 0.137), 0.98),
    "kimono_inner_lower": ((0.020, 0.031, 0.040), 0.98),
    "kimono_belt": ((0.014, 0.017, 0.019), 0.35),
}

# Map from USD link names (from GLB filenames) to Genesis URDF link names.
# Most are identical; a few have _rev_1_0 suffixes in USD.
_USD_TO_URDF_LINK = {
    "waist_yaw_link_rev_1_0": "waist_yaw_link",
    "waist_roll_link_rev_1_0": "waist_roll_link",
    "torso_link_rev_1_0": "torso_link",
}


# ── USD loading ──────────────────────────────────────────────────────────────


def _usd_name_to_link(usd_name: str) -> str:
    """``pelvis_glb`` -> ``pelvis``, ``torso_link_rev_1_0_glb`` -> ``torso_link``."""
    stripped = re.sub(r"_glb(_\d+)?$", "", usd_name)
    return _USD_TO_URDF_LINK.get(stripped, stripped)


def _usd_cloth_to_piece_name(usd_name: str) -> str:
    """``wearing_kimono_belt_0`` -> ``kimono_belt``."""
    return re.sub(r"^wearing_", "", re.sub(r"_\d+$", "", usd_name))


def load_usd_trajectory(usdc_path: Path) -> dict:
    """Load rigid transforms and cloth positions from a USD file.

    Returns dict with:
        frame_count: int
        link_transforms: dict[urdf_link_name -> (T, 4, 4) ndarray]  (column-vector)
        cloth: dict[usd_name -> { "positions": (T, N, 3), "faces": (M, 3) }]
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usdc_path))
    root_name = usdc_path.stem
    root_prim = stage.GetPrimAtPath(f"/{root_name}")

    link_transforms = {}
    cloth_data = {}
    frame_count = 0

    for child in root_prim.GetChildren():
        if child.GetName() == "rigid":
            for link_prim in child.GetChildren():
                xf = UsdGeom.Xformable(link_prim)
                ops = xf.GetOrderedXformOps()
                if not ops:
                    continue
                ts = ops[0].GetTimeSamples()
                frame_count = max(frame_count, len(ts))

                link_name = _usd_name_to_link(link_prim.GetName())
                # Skip duplicate sub-meshes (e.g. left_knee_link_glb_1)
                if link_name in link_transforms:
                    continue

                # USD is row-vector convention, transpose to column-vector
                transforms = np.zeros((len(ts), 4, 4), dtype=np.float64)
                for i, t in enumerate(ts):
                    m = np.array(ops[0].Get(t))
                    transforms[i] = m.T
                link_transforms[link_name] = transforms

        elif child.GetName() == "deformable":
            for cloth_prim in child.GetChildren():
                mesh = UsdGeom.Mesh(cloth_prim)
                pts_attr = mesh.GetPointsAttr()
                ts = pts_attr.GetTimeSamples()
                frame_count = max(frame_count, len(ts))

                face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
                faces = face_indices.reshape(-1, 3)

                first_pts = np.array(pts_attr.Get(ts[0]), dtype=np.float64)
                positions = np.zeros((len(ts), len(first_pts), 3), dtype=np.float64)
                for i, t in enumerate(ts):
                    positions[i] = np.array(pts_attr.Get(t), dtype=np.float64)

                cloth_data[cloth_prim.GetName()] = {
                    "positions": positions,
                    "faces": faces,
                }

    return {
        "frame_count": frame_count,
        "link_transforms": link_transforms,
        "cloth": cloth_data,
    }


# ── IK: compute qpos from world transforms ──────────────────────────────────


def _extract_joint_angle(R_rel: np.ndarray, axis: np.ndarray) -> float:
    """Extract rotation angle about a given axis from a rotation matrix."""
    rot = Rotation.from_matrix(R_rel)
    rotvec = rot.as_rotvec()
    # Project onto joint axis
    angle = np.dot(rotvec, axis)
    return angle


def build_ik_chain(urdf_path: str) -> list[dict]:
    """Parse URDF to build the IK chain for computing qpos.

    Returns a list of joint dicts, one per revolute joint, ordered by Genesis
    qs_idx.  Each dict has: name, parent_link, child_link, origin (4x4), axis.
    """
    from genesis.ext.urdfpy.urdf import URDF

    urdf = URDF.load(urdf_path)
    chain = []
    for j in urdf.joints:
        if j.joint_type == "fixed":
            continue
        origin = j.origin if j.origin is not None else np.eye(4)
        chain.append(
            {
                "name": j.name,
                "parent_link": j.parent,
                "child_link": j.child,
                "origin": origin.copy(),
                "axis": np.array(j.axis, dtype=np.float64),
            }
        )
    return chain


def _build_fixed_joint_map(urdf_path: str) -> dict[str, tuple[str, np.ndarray]]:
    """Map fixed-joint children to their parent link and static offset.

    Returns {child_link_name: (parent_link_name, T_parent_to_child)}.
    """
    from genesis.ext.urdfpy.urdf import URDF

    urdf = URDF.load(urdf_path)
    fixed_map = {}
    for j in urdf.joints:
        if j.joint_type == "fixed":
            origin = j.origin if j.origin is not None else np.eye(4)
            fixed_map[j.child] = (j.parent, origin.copy())
    return fixed_map


def compute_qpos_from_transforms(
    link_transforms: dict[str, np.ndarray],
    ik_chain: list[dict],
    fixed_joint_map: dict[str, tuple[str, np.ndarray]],
    frame: int,
) -> np.ndarray:
    """Compute joint angles for one frame from link world transforms.

    For each revolute joint:
      T_child = T_parent @ T_joint_origin @ R(q, axis)
      => R(q, axis) = (T_parent @ T_joint_origin)^-1 @ T_child
      => q = project rotation onto joint axis
    """
    qpos = np.zeros(len(ik_chain), dtype=np.float64)

    def _get_link_tf(link_name: str) -> np.ndarray | None:
        """Get link world transform, resolving fixed-joint chains."""
        if link_name in link_transforms:
            tf = link_transforms[link_name]
            return tf[frame] if frame < len(tf) else tf[-1]
        # Fixed joint child — compute from parent
        if link_name in fixed_joint_map:
            parent_name, offset = fixed_joint_map[link_name]
            parent_tf = _get_link_tf(parent_name)
            if parent_tf is not None:
                return parent_tf @ offset
        return None

    for i, jinfo in enumerate(ik_chain):
        T_parent = _get_link_tf(jinfo["parent_link"])
        T_child = _get_link_tf(jinfo["child_link"])
        if T_parent is None or T_child is None:
            continue

        # T_child = T_parent @ T_joint_origin @ R(q, axis)
        T_joint_frame = T_parent @ jinfo["origin"]
        T_rel = np.linalg.inv(T_joint_frame) @ T_child
        R_rel = T_rel[:3, :3]
        qpos[i] = _extract_joint_angle(R_rel, jinfo["axis"])

    return qpos


# ── Main ─────────────────────────────────────────────────────────────────────


def run_gui(
    traj: dict,
    trajectory_name: str,
    render_output: str | None = None,
    use_nyx: bool = False,
    start_frame: int = 0,
    save_frames: bool = False,
) -> None:
    import genesis as gs

    use_render = render_output is not None
    gs.init(backend=gs.gpu if use_render else gs.cpu, logging_level="warning")

    renderer_kwargs = {}
    if use_render and not use_nyx:
        renderer_kwargs["renderer"] = gs.renderers.RayTracer(
            env_radius=15.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (3.0, 2.0, 4.0), "radius": 1.0, "color": (10.0, 10.0, 10.0)},
                {"pos": (-2.0, 3.0, 3.0), "radius": 1.5, "color": (5.0, 5.0, 5.0)},
                {"pos": (-1.0, -2.0, 3.0), "radius": 0.8, "color": (6.0, 6.0, 6.0)},
            ],
        )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.5, 1.2),
            camera_lookat=(0.0, 0.0, 0.6),
            camera_fov=45,
        ),
        vis_options=gs.options.VisOptions(ambient_light=(0.3, 0.3, 0.35)),
        show_viewer=not use_render,
        **renderer_kwargs,
    )

    frame_count = traj["frame_count"]
    link_transforms = traj["link_transforms"]
    cloth_data = traj["cloth"]

    # ── Ground plane ──
    scene.add_entity(gs.morphs.Plane())

    # ── Robot (URDF) ──
    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, pos=(0.0, 0.0, 0.0), fixed=True, collision=False),
        surface=gs.surfaces.BSDF(),
    )

    # ── IK chain ──
    ik_chain = build_ik_chain(URDF_PATH)
    fixed_joint_map = _build_fixed_joint_map(URDF_PATH)

    # Build Genesis joint name -> qs_idx mapping to reorder qpos
    # ik_chain is in URDF order; need to map to Genesis qs_idx order
    joint_name_to_qs_idx: dict[str, int] = {}
    for joint in robot.joints:
        if joint.n_qs > 0:
            joint_name_to_qs_idx[joint.name] = joint.qs_idx_local[0]

    # ── Cloth entities ──
    cloth_entities: dict[str, object] = {}
    for usd_name, cdata in cloth_data.items():
        piece_name = _usd_cloth_to_piece_name(usd_name)
        obj_path = _KIMONO_DIR / f"{piece_name}.obj"
        if not obj_path.exists():
            print(f"[warn] cloth mesh not found: {obj_path.name}")
            continue
        # Prefer GLB (has PBR material baked in), fall back to OBJ
        glb_path = _KIMONO_DIR / f"{piece_name}.glb"
        mesh_path = glb_path if glb_path.exists() else obj_path
        ent = scene.add_entity(
            morph=gs.morphs.Mesh(file=str(mesh_path)),
            material=gs.materials.FEM.Cloth(E=1e4, rho=200.0, thickness=1e-4),
            surface=gs.surfaces.BSDF(),
            name=usd_name,
        )
        cloth_entities[usd_name] = ent

    # ── Camera (tracks pelvis) ──
    # Camera offset relative to pelvis: front-left, slightly above
    CAM_OFFSET = np.array([1.5, -1.5, 0.4])
    CAM_LOOKAT_OFFSET = np.array([0.0, 0.0, -0.1])
    cam_up = (0.0, 0.0, 1.0)
    cam_fov = 40

    def _get_pelvis_pos(frame_idx: int) -> np.ndarray:
        if "pelvis" in link_transforms and frame_idx < link_transforms["pelvis"].shape[0]:
            return link_transforms["pelvis"][frame_idx][:3, 3].copy()
        return np.array([0.0, 0.0, 0.78])

    init_pelvis = _get_pelvis_pos(0)
    cam_pos = tuple(init_pelvis + CAM_OFFSET)
    cam_lookat = tuple(init_pelvis + CAM_LOOKAT_OFFSET)

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
                pos=cam_pos,
                lookat=cam_lookat,
                up=cam_up,
                fov=cam_fov,
                spp=256,
                denoise=True,
                render_mode=npr.ERenderMode.RefPathTracer,
                env_maps=(env_map,),
            )
        )
    elif use_render:
        cam = scene.add_camera(
            res=(1920, 1080),
            pos=cam_pos,
            lookat=cam_lookat,
            up=cam_up,
            fov=cam_fov,
            spp=256,
        )

    scene.build(n_envs=1 if (use_render and use_nyx) else 0)

    def _apply_frame(i: int) -> None:
        if i < 0 or i >= frame_count:
            return

        # Compute qpos from USD transforms
        qpos_urdf_order = compute_qpos_from_transforms(
            link_transforms,
            ik_chain,
            fixed_joint_map,
            i,
        )

        # Reorder from URDF joint order to Genesis qs_idx order
        qpos = robot.get_qpos()
        for j_idx, jinfo in enumerate(ik_chain):
            gs_idx = joint_name_to_qs_idx.get(jinfo["name"])
            if gs_idx is not None:
                qpos[gs_idx] = qpos_urdf_order[j_idx]

        # Set pelvis (base) position from USD
        if "pelvis" in link_transforms and i < link_transforms["pelvis"].shape[0]:
            tf = link_transforms["pelvis"][i]
            pos = tf[:3, 3]
            robot.set_pos(pos)
            rot = Rotation.from_matrix(tf[:3, :3])
            xyzw = rot.as_quat()
            robot.set_quat(np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]))

        robot.set_qpos(qpos, zero_velocity=True)

        # Cloth
        for usd_name, ent in cloth_entities.items():
            cdata = cloth_data[usd_name]
            if i < cdata["positions"].shape[0]:
                ent.set_position(cdata["positions"][i])

        scene._visualizer.update_visual_states(force_render=True)

    # ── Render mode ──
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

            # Track pelvis with camera
            pelvis_pos = _get_pelvis_pos(i)
            cur_cam_pos = tuple(pelvis_pos + CAM_OFFSET)
            cur_cam_lookat = tuple(pelvis_pos + CAM_LOOKAT_OFFSET)

            if use_nyx:
                cam.update_camera_pose(pos=cur_cam_pos, lookat=cur_cam_lookat, up=cam_up)
                data = cam.read()
                rgb = data.rgb.cpu().numpy()
            else:
                cam.set_pose(pos=cur_cam_pos, lookat=cur_cam_lookat, up=cam_up)
                rgb_result = cam.render(rgb=True, force_render=True)
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

        fps = min(int(1.0 / 0.01 / 2), 60)
        writer = imageio.get_writer(str(output_path), fps=fps)
        for rgb in frames_rgb:
            writer.append_data(rgb)
        writer.close()

        if not use_nyx:
            cam.stop_recording()

        print(f"[render] Saved {output_path} ({len(frames_rgb)} frames)")
        return

    # ── Interactive replay ──
    if start_frame > 0:
        print(f"[replay] Skipping to frame {start_frame}")
        _apply_frame(start_frame - 1)
    for i in range(start_frame, frame_count):
        _apply_frame(i)
    print(f"[replay] finished {frame_count - start_frame} frames (from {start_frame})")


def run_no_gui(traj: dict, trajectory_name: str) -> None:
    frame_count = traj["frame_count"]
    link_transforms = traj["link_transforms"]
    cloth_data = traj["cloth"]

    print(f"[no-gui] {trajectory_name}: {frame_count} frames")
    print(f"  rigid links: {len(link_transforms)}")
    for usd_name, cdata in cloth_data.items():
        piece = _usd_cloth_to_piece_name(usd_name)
        print(f"  cloth: {piece}  verts={cdata['positions'].shape[1]}  faces={cdata['faces'].shape[0]}")
    print("[no-gui] Done.")


def main() -> None:
    _OUTPUT_DIR = Path("data/ipc_demo/ipc_costume")

    parser = argparse.ArgumentParser(description="Replay G1 kimono dancing simulation (Genesis).")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--render", action="store_true", help="Record video")
    parser.add_argument("--nyx", action="store_true", help="Use Nyx renderer")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="d0_no_lower",
        choices=AVAILABLE_TRAJECTORIES,
        help="Trajectory to replay.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Start from this frame.")
    parser.add_argument("--save-frames", action="store_true", help="Save each frame as PNG.")
    args = parser.parse_args()

    usdc_path = _PACK_DIR / f"{args.trajectory}.usdc"
    if not usdc_path.exists():
        print(f"[error] USD file not found: {usdc_path}")
        return

    print(f"[replay] Loading {usdc_path.name} ...")
    traj = load_usd_trajectory(usdc_path)
    print(
        f"[replay] {traj['frame_count']} frames, {len(traj['link_transforms'])} rigid links, {len(traj['cloth'])} cloth"
    )

    render_output = None
    if args.render:
        renderer_name = "nyx" if args.nyx else "luisa"
        render_output = str(_OUTPUT_DIR / f"costume_{args.trajectory}_{renderer_name}.mp4")

    if args.no_gui:
        run_no_gui(traj, args.trajectory)
    else:
        run_gui(
            traj,
            args.trajectory,
            render_output=render_output,
            use_nyx=args.nyx,
            start_frame=args.start_frame,
            save_frames=args.save_frames,
        )


if __name__ == "__main__":
    main()
