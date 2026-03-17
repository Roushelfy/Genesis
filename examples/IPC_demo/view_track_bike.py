"""
Visualize the articulated track bike URDF with chain on sprockets.

Bike uses two_way_soft_constraint coupling; chain parts use ipc_only.

Usage:
    LD_PRELOAD=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12 \
        python examples/IPC_demo/view_track_bike.py [--no-viewer] [--no-chain]
"""

import argparse
import os

import imageio
import numpy as np
from scipy.optimize import fsolve

import genesis as gs
import genesis.utils.geom as gu

from load_rigid_ipc_scene import (
    euler_xyz_deg_to_quat,
    yup_to_zup_position,
    yup_to_zup_quat,
)

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_bike", "track_bike.urdf")

# Chain mesh paths (same mesh family as the sprockets)
CHAIN_MESH_DIR = os.path.expanduser("~/Desktop/hz/rigid-ipc/meshes/507-movements/227-chain-pully")
LINK_MESH = os.path.join(CHAIN_MESH_DIR, "link.obj")
PIN_MESH = os.path.join(CHAIN_MESH_DIR, "pin.obj")
BARRING_MESH = os.path.join(CHAIN_MESH_DIR, "barring.obj")

# Chain geometry constants (from chain_pully.py, in mesh units)
LINK_HOLE_CENTER = 2.45905
LINK_WIDTH = 2 * LINK_HOLE_CENTER
LINK_Z_OFFSETS = [0.763387, 0.940965]

# Scale: same as the sprocket scale in build_track_bike_urdf.py
CHAIN_SCALE = 0.005597

# Sprocket positions in Y-up mesh coords (from build_track_bike_urdf.py)
FRONT_CENTER_YUP = np.array([-0.10925, 0.31893, 0.04541])
REAR_CENTER_YUP = np.array([-0.53711, 0.36663, 0.04541])


def _solve_chain_radii(front_xy, rear_xy, link_w, n_front_arc, n_rear_arc, n_top, n_bot):
    """Solve for chain path radii that give exact link_w spacing everywhere.

    The chain path rides slightly off the nominal pitch circles so that
    both straight sections are exact integer multiples of link_w.  This
    eliminates per-link spacing error (< 1 µm vs 70 µm tolerance).
    """
    d_vec = rear_xy - front_xy
    d_dist = np.linalg.norm(d_vec)
    theta = np.arctan2(d_vec[1], d_vec[0])

    # Nominal pitch-circle radii as initial guess
    r_front_nom = LINK_WIDTH / (2 * np.sin(np.pi / 20)) * CHAIN_SCALE
    r_rear_nom = LINK_WIDTH / (2 * np.sin(np.pi / 8)) * CHAIN_SCALE

    def equations(x):
        rf, rr = x
        if rf <= 0 or rr <= 0 or abs((rf - rr) / d_dist) >= 1:
            return [1e6, 1e6]
        hf = link_w / (2 * rf)
        hr = link_w / (2 * rr)
        if abs(hf) >= 1 or abs(hr) >= 1:
            return [1e6, 1e6]

        arc_step_f = 2 * np.arcsin(hf)
        arc_step_r = 2 * np.arcsin(hr)
        beta = np.arcsin((rf - rr) / d_dist)
        tangent_top = theta - np.pi / 2 + beta
        tangent_bot = theta + np.pi / 2 - beta

        # Arc endpoints
        front_end_angle = tangent_bot + n_front_arc * arc_step_f
        rear_start_angle = tangent_top
        rear_end_angle = rear_start_angle + n_rear_arc * arc_step_r

        front_end_pt = front_xy + rf * np.array([np.cos(front_end_angle), np.sin(front_end_angle)])
        rear_start_pt = rear_xy + rr * np.array([np.cos(rear_start_angle), np.sin(rear_start_angle)])
        rear_end_pt = rear_xy + rr * np.array([np.cos(rear_end_angle), np.sin(rear_end_angle)])
        front_start_pt = front_xy + rf * np.array([np.cos(tangent_bot), np.sin(tangent_bot)])

        top_dist = np.linalg.norm(rear_start_pt - front_end_pt)
        bot_dist = np.linalg.norm(front_start_pt - rear_end_pt)
        return [top_dist - n_top * link_w, bot_dist - n_bot * link_w]

    sol, info, ier, msg = fsolve(equations, [r_front_nom, r_rear_nom], full_output=True)
    if ier != 1 or np.max(np.abs(info["fvec"])) > 1e-10:
        raise RuntimeError(f"Chain radius solve failed: {msg}")
    return sol[0], sol[1]


def generate_chain_path():
    """Generate chain joint positions in the Y-up XY plane (world units).

    Returns Nx2 array of chain joint positions forming a closed loop.

    The chain path radii are optimized so that every segment (arc and
    straight) has exactly link_w chord length.  The radii differ from
    the nominal pitch circles by ~2 mm; IPC contact forces pull the
    chain into proper mesh during simulation.
    """
    link_w = LINK_WIDTH * CHAIN_SCALE
    front_xy = FRONT_CENTER_YUP[:2]
    rear_xy = REAR_CENTER_YUP[:2]

    # Target: 46 segments (even), split as 14 front arc + 13 top + 3 rear arc + 16 bottom
    n_front_arc = 12
    n_rear_arc = 4
    n_top = 14
    n_bot = 16

    rf = 0.08798
    rr = 0.03597
    # rf, rr = _solve_chain_radii(front_xy, rear_xy, link_w, n_front_arc, n_rear_arc, n_top, n_bot)

    # Arc angular steps (chord = link_w at solved radius)
    # arc_step_f = 2 * np.arcsin(link_w / (2 * rf))
    # arc_step_r = 2 * np.arcsin(link_w / (2 * rr))
    arc_step_f = 2 * np.pi / 20
    arc_step_r = 2 * np.pi / 8

    # d_vec = rear_xy - front_xy
    # d_dist = np.linalg.norm(d_vec)
    # theta = np.arctan2(d_vec[1], d_vec[0])
    # beta = np.arcsin((rf - rr) / d_dist)
    # tangent_top = theta - np.pi / 2 + beta
    # tangent_bot = theta + np.pi / 2 - beta

    # Front arc: tangent_bot CCW, n_front_arc+1 points
    front_angles = np.arange(n_front_arc + 1) * arc_step_f - np.pi / 2
    front_pts = np.column_stack(
        [
            front_xy[0] + rf * np.cos(front_angles),
            front_xy[1] + rf * np.sin(front_angles),
        ]
    )

    rear_angles = np.arange(n_rear_arc + 1) * arc_step_r + np.pi / 2
    rear_pts = np.column_stack(
        [
            rear_xy[0] + rr * np.cos(rear_angles),
            rear_xy[1] + rr * np.sin(rear_angles),
        ]
    )

    # Top straight: exact link_w steps from front arc end toward rear arc start
    # rear_start_pt = rear_xy + rr * np.array([np.cos(tangent_top), np.sin(tangent_top)])
    top_dir = rear_pts[0] - front_pts[-1]
    top_dir = top_dir / np.linalg.norm(top_dir)
    top_pts = front_pts[-1] + np.array([i * link_w * top_dir for i in range(1, n_top - 1)])
    top_last_seg = rear_pts[0] - top_pts[-1]
    top_last_n = np.linalg.norm(top_last_seg)
    assert link_w < top_last_n <= link_w * 2, f"{link_w} vs {top_last_n} vs {link_w * 2}"

    top_last_normed = top_last_seg / top_last_n
    top_vert_n = np.sqrt(link_w**2 - (top_last_n / 2) ** 2)
    top_last_pt = (
        top_pts[-1]
        + top_last_normed * (top_last_n / 2)
        + np.array((top_last_normed[1], -top_last_normed[0])) * top_vert_n
    )

    bot_dir = front_pts[0] - rear_pts[-1]
    bot_dir = bot_dir / np.linalg.norm(bot_dir)
    bot_pts = rear_pts[-1] + np.array([i * link_w * bot_dir for i in range(1, n_bot - 1)])
    bot_last_seg = front_pts[0] - bot_pts[-1]
    bot_last_n = np.linalg.norm(bot_last_seg)
    assert link_w < bot_last_n <= link_w * 2, f"{link_w} vs {bot_last_n} vs {link_w * 2}"

    bot_last_normed = bot_last_seg / bot_last_n
    bot_vert_n = np.sqrt(link_w**2 - (bot_last_n / 2) ** 2)
    bot_last_pt = (
        bot_pts[-1]
        + bot_last_normed * (bot_last_n / 2)
        + np.array([bot_last_normed[1], -bot_last_normed[0]]) * bot_vert_n
    )

    # Combine: front_arc + top_straight + top_kink + rear_arc + bot_straight + bot_kink
    points = np.vstack(
        [
            front_pts,
            top_pts,
            top_last_pt[np.newaxis],
            rear_pts,
            bot_pts,
            bot_last_pt[np.newaxis],
        ]
    )

    # Verify spacing
    max_err = 0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        d = np.linalg.norm(points[j] - points[i])
        max_err = max(max_err, abs(d - link_w))

    r_front_nom = LINK_WIDTH / (2 * np.sin(np.pi / 20)) * CHAIN_SCALE
    r_rear_nom = LINK_WIDTH / (2 * np.sin(np.pi / 8)) * CHAIN_SCALE

    print(
        f"  Chain path: {len(points)} joints "
        f"(front arc {n_front_arc + 1} + top {n_top - 1} + rear arc {n_rear_arc + 1}"
        f" + bottom {n_bot - 1})"
    )
    print(
        f"  Radii: rf={rf * 1000:.2f}mm (δ={((rf - r_front_nom) * 1000):+.2f}mm) "
        f"rr={rr * 1000:.2f}mm (δ={((rr - r_rear_nom) * 1000):+.2f}mm)"
    )
    print(f"  Max spacing error: {max_err * 1e6:.2f} µm (tolerance: 70 µm)")
    return points


def add_chain(scene, bike_pos_zup=(0, 0, 0)):
    """Add chain entities as ipc_only bodies.

    Each segment gets 2 link plates (±Z offset), 1 barring at the start
    joint, and 1 pin at the end joint — matching the original chain_pully.py
    layout.  Pins and barrings are placed at the path points (joint
    positions) so that adjacent segments share co-located pin+barring at
    each joint, so adjacent segments share co-located pin+barring.

    bike_pos_zup: world-frame offset of the bike entity (Z-up),
        applied to all chain part positions so they align with the sprockets.
    """
    bike_offset_zup = np.array(bike_pos_zup, dtype=float)
    points = generate_chain_path()
    n = len(points)

    chain_z = FRONT_CENTER_YUP[2]

    chain_surface = gs.surfaces.Iron(color=(0.15, 0.15, 0.15))
    chain_material = gs.materials.Rigid(
        coup_type="ipc_only",
        coup_friction=0.0,
        friction=0.01,
        enable_coup_collision=True,
    )
    # Base quat for barring/pin: 90° around X (Y-up mesh convention), no yaw
    joint_quat = yup_to_zup_quat(euler_xyz_deg_to_quat(90, 0, 0))
    entities = []

    # scene.add_entity(
    #     gs.morphs.Sphere(
    #         pos=yup_to_zup_position(FRONT_CENTER_YUP),
    #         radius=0.02,
    #         collision=False,
    #     ),
    #     surface=gs.surfaces.Plastic(color=(0.0, 0.0, 1.0)),
    # )
    # scene.add_entity(
    #     gs.morphs.Sphere(
    #         pos=yup_to_zup_position(REAR_CENTER_YUP),
    #         radius=0.02,
    #         collision=False,
    #     ),
    #     surface=gs.surfaces.Plastic(color=(1.0, 0.0, 0.0)),
    # )

    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        mid = (p0 + p1) / 2

        dp = p1 - p0
        seg_len = np.linalg.norm(dp)
        angle_deg = np.degrees(np.arctan2(dp[1], dp[0]))
        link_quat = yup_to_zup_quat(euler_xyz_deg_to_quat(90, 0, angle_deg))

        z_off = LINK_Z_OFFSETS[i % 2] * CHAIN_SCALE

        # Two link plates (±Z offset in Y-up)
        for sign in (+1, -1):
            pos_yup = (mid[0], mid[1], chain_z + sign * z_off)
            pos_zup = np.array(yup_to_zup_position(pos_yup)) + bike_offset_zup
            entities.append(
                scene.add_entity(
                    gs.morphs.Mesh(
                        file=LINK_MESH,
                        pos=tuple(pos_zup),
                        quat=link_quat,
                        scale=CHAIN_SCALE,
                        fixed=False,
                        convexify=False,
                        decimate=False,
                    ),
                    material=chain_material,
                    surface=chain_surface,
                )
            )

        # Barring at start joint (path point p0)
        barring_pos_yup = (p0[0], p0[1], chain_z)
        barring_pos_zup = np.array(yup_to_zup_position(barring_pos_yup)) + bike_offset_zup
        entities.append(
            scene.add_entity(
                gs.morphs.Mesh(
                    file=BARRING_MESH,
                    pos=tuple(barring_pos_zup),
                    quat=joint_quat,
                    scale=CHAIN_SCALE,
                    fixed=False,
                    convexify=False,
                    decimate=False,
                ),
                material=chain_material,
                surface=chain_surface,
            )
        )

        # Pin at end joint (path point p1)
        pin_pos_yup = (p1[0], p1[1], chain_z)
        pin_pos_zup = np.array(yup_to_zup_position(pin_pos_yup)) + bike_offset_zup
        entities.append(
            scene.add_entity(
                gs.morphs.Mesh(
                    file=PIN_MESH,
                    pos=tuple(pin_pos_zup),
                    quat=joint_quat,
                    scale=CHAIN_SCALE,
                    fixed=False,
                    convexify=False,
                    decimate=False,
                ),
                material=chain_material,
                surface=chain_surface,
            )
        )

    print(f"  Added {len(entities)} chain bodies ({n} segments: {2 * n} links + {n} barrings + {n} pins)")
    return entities


def main():
    parser = argparse.ArgumentParser(description="Track bike with chain")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-chain", action="store_true", help="Load bike only, no chain")
    parser.add_argument("--no-bike", action="store_true", help="Load chain only, no bike")
    parser.add_argument("--no-franka", action="store_true", help="Skip Franka robot")
    parser.add_argument("--no-plane", action="store_true", help="No ground plane")
    parser.add_argument("--fix-bike", action="store_true", help="Fix bike base (no free joint)")
    parser.add_argument("--no-gravity", action="store_true")
    parser.add_argument(
        "--motor", choices=["front", "rear", "none"], default="rear", help="Which sprocket to apply motor to"
    )
    parser.add_argument("--steps", type=int, default=600, help="Number of sim steps")
    parser.add_argument("--video", type=str, default="./data/track_bike.mp4", help="Video output path")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=not args.no_viewer,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -1.5, 0.7),
            camera_lookat=(-0.2, 0.0, 0.4),
            camera_fov=45,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            enable_collision=True,
            constraint_solver=gs.constraint_solver.CG,
            gravity=(0, 0, 0 if args.no_gravity else -9.81),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=2e-4,
            newton_semi_implicit_enable=True,
            linear_system_tolerance=1e-4,
        ),
    )

    if not args.no_plane:
        # Ground plane (rigid collision only, no IPC coupling)
        scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(needs_coup=False),
        )

    bike_pos_zup = (0, 0, 0.002)

    if not args.no_chain:
        print("Adding chain...")
        add_chain(scene, bike_pos_zup=bike_pos_zup)

    if not args.no_bike:
        # Mesh is Y-up (Blender convention); rotate 90 deg around X for Genesis Z-up.
        # Bike uses two_way coupling so IPC and Genesis share control.
        print("Adding bike...")
        bike_material = gs.materials.Rigid(
            coup_type="two_way_soft_constraint",
            coup_friction=0.3,
            coup_links=["front_sprocket", "rear_sprocket"],
            enable_coup_collision=True,
        )

        bike = scene.add_entity(
            gs.morphs.URDF(
                file=URDF_PATH,
                fixed=args.fix_bike,
                pos=bike_pos_zup,
                euler=(90, 0, 0),
                convexify=False,
                merge_fixed_links=False,
            ),
            material=bike_material,
            vis_mode="collision",
        )

        print(f"Links: {[l.name for l in bike.links]}")
        print(f"Joints: {[(j.name, str(j.type).split('.')[-1]) for j in bike.joints]}")
        print(f"DOFs: {bike.n_dofs}")
    else:
        bike = None

    # Franka robot to grip the crank pedal from the front
    franka = None
    if not args.no_franka:
        # Front sprocket center in Z-up: (~-0.109, ~-0.045, ~0.319)
        sprocket_zup = yup_to_zup_position(FRONT_CENTER_YUP)
        print(f"Adding Franka... (sprocket center Z-up: {sprocket_zup})")
        # Place Franka in front of the bike (negative Y), reaching forward (+Y)
        franka = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda_non_overlap.xml",
                pos=(sprocket_zup[0], sprocket_zup[1] - 0.6, 0.0),
            ),
            material=gs.materials.Rigid(
                coup_type="two_way_soft_constraint",
                coup_links=("left_finger", "right_finger"),
            ),
        )

    # Camera for recording
    cam = scene.add_camera(
        res=(1280, 960),
        pos=(0.5, -1.2, 0.6),
        lookat=(-0.2, 0.0, 0.35),
        fov=45,
    )

    scene.build()

    # Set up motor to spin a sprocket
    if bike is not None and args.motor != "none":
        has_chain = not args.no_chain
        # With chain: slightly stronger motor to overcome chain friction/inertia
        motor_kv = 10.0 if has_chain else 5.0
        motor_vel = -3.0

        if args.motor == "front":
            motor_dof = bike.get_joint("front_sprocket_joint").dof_idx_local
        else:
            motor_dof = bike.get_joint("rear_sprocket_joint").dof_idx_local
        bike.set_dofs_kp(0.0, dofs_idx_local=motor_dof)
        bike.set_dofs_kv(motor_kv, dofs_idx_local=motor_dof)
        bike.control_dofs_velocity(motor_vel, dofs_idx_local=motor_dof)

    # Set up Franka multi-phase grasp
    if franka is not None:
        motor_dofs_idx = slice(0, 7)
        finger_dofs_idx = slice(7, 9)
        ee_link = franka.get_link("hand")

        sprocket_zup = yup_to_zup_position(FRONT_CENTER_YUP)
        # Crank geometry in Z-up:
        #   Arms extend ±0.187 in Z from sprocket center → bottom pedal at Z ≈ 0.132
        #   Pedal width along Y: sprocket_y+0.201 to sprocket_y-0.109
        #   Grab from -Y side near Y ≈ sprocket_y - 0.10
        pedal_radius = 0.187
        pedal_z = sprocket_zup[2] - pedal_radius
        pedal_x = sprocket_zup[0]
        # Where the pedal is in Y (crank mesh Z range -0.201..+0.109 maps to Y offset +0.201..-0.109)
        pedal_y = sprocket_zup[1] - 0.10

        # Hand faces forward (-Y) to approach from front
        grip_quat = gu.xyz_to_quat(np.array([-90.0, 0.0, 0.0], dtype=gs.np_float), degrees=True)

        # Phase waypoints
        # 1) Start: offset in -Y from pedal (open grip, no contact)
        approach_pos = np.array([pedal_x, pedal_y - 0.15, pedal_z], dtype=gs.np_float)
        # 2) Grasp: at the pedal (close grip)
        grasp_pos = np.array([pedal_x, pedal_y, pedal_z], dtype=gs.np_float)

        # Solve IK for approach position, set as initial pose
        qpos = franka.inverse_kinematics(
            link=ee_link,
            pos=approach_pos,
            quat=grip_quat,
            dofs_idx_local=motor_dofs_idx,
        )
        franka.set_qpos(qpos)
        franka.control_dofs_position(qpos[motor_dofs_idx], motor_dofs_idx)

        # PD gains
        franka.set_dofs_kp(
            [4500, 4500, 3500, 3500, 2000, 2000, 2000],
            dofs_idx_local=motor_dofs_idx,
        )
        franka.set_dofs_kv(
            [450, 450, 350, 350, 200, 200, 200],
            dofs_idx_local=motor_dofs_idx,
        )
        franka.set_dofs_kp(500.0, dofs_idx_local=finger_dofs_idx)
        franka.set_dofs_kv(50.0, dofs_idx_local=finger_dofs_idx)
        # Open grip initially
        franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs_idx)

    first_frame_saved = False
    video_frames = []

    # Phase durations (in steps)
    N_SETTLE = 30
    N_APPROACH = 100
    N_CLOSE = 50
    n_rotate = args.steps - N_SETTLE - N_APPROACH - N_CLOSE

    for step_i in range(args.steps):
        if franka is not None:
            if step_i < N_SETTLE:
                # Phase 1: Settle with open grip at approach position
                pass

            elif step_i < N_SETTLE + N_APPROACH:
                # Phase 2: Move in +Y toward the crank pedal (grip open)
                t = (step_i - N_SETTLE) / N_APPROACH
                pos = approach_pos * (1 - t) + grasp_pos * t
                qpos = franka.inverse_kinematics(
                    link=ee_link,
                    pos=pos,
                    quat=grip_quat,
                    dofs_idx_local=motor_dofs_idx,
                )
                franka.control_dofs_position(qpos[motor_dofs_idx], motor_dofs_idx)
                franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs_idx)

            elif step_i < N_SETTLE + N_APPROACH + N_CLOSE:
                # Phase 3: Close grip around the pedal
                franka.control_dofs_position(0.0, dofs_idx_local=finger_dofs_idx)

            else:
                # Phase 4: Rotate the crank by tracing a circular arc in XZ plane
                rot_step = step_i - N_SETTLE - N_APPROACH - N_CLOSE
                # Start angle at bottom (-π/2), rotate CCW
                angle = -np.pi / 2 + rot_step * 0.01 * 1.0
                rot_pos = np.array(
                    [
                        sprocket_zup[0] + pedal_radius * np.cos(angle),
                        pedal_y,
                        sprocket_zup[2] + pedal_radius * np.sin(angle),
                    ],
                    dtype=gs.np_float,
                )
                qpos = franka.inverse_kinematics(
                    link=ee_link,
                    pos=rot_pos,
                    quat=grip_quat,
                    dofs_idx_local=motor_dofs_idx,
                )
                franka.control_dofs_position(qpos[motor_dofs_idx], motor_dofs_idx)
                franka.control_dofs_position(0.0, dofs_idx_local=finger_dofs_idx)

        scene.step()

        # Log bike position and update camera to follow
        if bike is not None and step_i % 50 == 0:
            pos = bike.get_pos()
            vel = bike.get_vel()
            print(f"  Step {step_i:4d}: bike pos = {pos}  vel = {vel}")

        if bike is not None:
            bike_pos = bike.get_pos().cpu().numpy()
            cam.set_pose(
                pos=(bike_pos[0] + 0.5, bike_pos[1] - 1.2, 0.6),
                lookat=(bike_pos[0], bike_pos[1], 0.35),
            )

        # Render and collect frame
        rgb, _, _, _ = cam.render(rgb=True)
        video_frames.append(rgb)
        if not first_frame_saved:
            imageio.imwrite("track_bike_first_frame.png", rgb)
            print("Saved first frame to track_bike_first_frame.png")
            first_frame_saved = True

        # Flush video every 10 steps so we can watch progress mid-simulation
        if (step_i + 1) % 10 == 0 or step_i == args.steps - 1:
            imageio.mimwrite(args.video, video_frames, fps=60)
            print(f"  Saved video ({step_i + 1}/{args.steps} frames) to {args.video}")


if __name__ == "__main__":
    main()
