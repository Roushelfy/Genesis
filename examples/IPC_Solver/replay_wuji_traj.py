"""
Replay a recorded Wuji hand trajectory in Genesis viewer.

Usage:
    python examples/IPC_Solver/replay_wuji_traj.py
    python examples/IPC_Solver/replay_wuji_traj.py --traj /path/to/wuji_qpos_trajectory.npy
    python examples/IPC_Solver/replay_wuji_traj.py --no-ipc   # pure rigid (no IPC coupler)
    python examples/IPC_Solver/replay_wuji_traj.py --loop      # loop the trajectory
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import genesis as gs

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRAJ = Path("/home/zhehuan/Desktop/hz/gs-core/tmp/teleop/20260403_210121_174578/wuji_qpos_trajectory.npy")


def main():
    parser = argparse.ArgumentParser(description="Replay Wuji hand trajectory")
    parser.add_argument("--traj", type=str, default=str(_DEFAULT_TRAJ), help="Path to qpos trajectory .npy")
    parser.add_argument(
        "--timestamps", type=str, default=None, help="Path to timestamps .npy (optional, for real-time playback)"
    )
    parser.add_argument("--no-ipc", action="store_true", help="Pure rigid, no IPC coupler")
    parser.add_argument("--loop", action="store_true", help="Loop the trajectory")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument(
        "--pd", action="store_true", help="Use control_dofs_position (PD control) instead of set_qpos (teleport)"
    )
    parser.add_argument("--kp", type=float, default=2000.0, help="PD position gain (with --pd)")
    parser.add_argument("--kv", type=float, default=100.0, help="PD velocity gain (with --pd)")
    args = parser.parse_args()

    # Load trajectory
    traj = np.load(args.traj)
    print(f"Loaded trajectory: {traj.shape} ({traj.shape[0]} frames, {traj.shape[1]} DOFs)")

    timestamps = None
    if args.timestamps:
        timestamps = np.load(args.timestamps)
        print(f"Loaded timestamps: {timestamps.shape}")

    gs.init(backend=gs.gpu)

    scene_kwargs = dict(
        sim_options=gs.options.SimOptions(dt=1 / 60),
        rigid_options=gs.options.RigidOptions(constraint_solver=gs.constraint_solver.CG),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -0.8, 1.5),
            camera_lookat=(0.4, 0.0, 1.0),
            camera_fov=45,
        ),
        show_viewer=True,
    )

    if not args.no_ipc:
        scene_kwargs["coupler_options"] = gs.options.IPCCouplerOptions(
            contact_d_hat=0.001,
            contact_resistance=1e7,
        )

    scene = gs.Scene(**scene_kwargs)

    # Ground plane
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(coup_type="ipc_only") if not args.no_ipc else None,
    )

    # Wuji robot
    urdf_path = str(
        Path.home()
        / ".cache/huggingface/hub/datasets--Genesis-Intelligence--internal_assets/snapshots/0978856d9bd81e63422af041364c5963e15f467e/marvin_wuji_description/robot_capsule.urdf"
    )

    mat_kwargs = {
        "rho": 100.0,
    }
    if not args.no_ipc:
        mat_kwargs["coup_type"] = "external_articulation"
        mat_kwargs["coup_friction"] = 0.9
        mat_kwargs["friction"] = 0.5

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0.0, 0.0, 1.08),
            fixed=True,
        ),
        material=gs.materials.Rigid(**mat_kwargs),
    )

    scene.build()
    scene.step()

    print(f"Robot: {robot.n_dofs} DOFs, {robot.n_links} links")
    print(f"Trajectory DOFs: {traj.shape[1]}")

    # Get qpos0 offset (difference between absolute qpos and relative dofs_position)
    # qpos0 = qpos - dofs_pos (at build time, before any movement)
    build_qpos = robot.get_qpos()
    build_dofs = robot.get_dofs_position()
    qpos0_offset = build_qpos - build_dofs  # qpos0 per DOF
    print(f"qpos0_offset: {qpos0_offset.cpu().numpy().flatten()}")

    # Set initial pose (always teleport first frame)
    qpos0 = torch.from_numpy(traj[0]).unsqueeze(0).float().cuda()
    robot.set_qpos(qpos0)

    if args.pd:
        # Set PD gains and force limits for all DOFs
        robot.set_dofs_kp(np.full(robot.n_dofs, args.kp))
        robot.set_dofs_kv(np.full(robot.n_dofs, args.kv))
        force_limit = np.full(robot.n_dofs, 1000.0)
        robot.set_dofs_force_range(-force_limit, force_limit)
        print(f"PD mode: kp={args.kp}, kv={args.kv}, force_range=1000")

    scene.step()

    # Replay loop
    dt = 1.0 / 60 / args.speed
    while True:
        # Reset scene and teleport to first frame
        scene.reset()
        robot.set_qpos(torch.from_numpy(traj[0]).unsqueeze(0).float().cuda())

        for i in range(traj.shape[0]):
            t_start = time.time()

            qpos = torch.from_numpy(traj[i]).unsqueeze(0).float().cuda()
            if args.pd and i > 0:
                # control_dofs_position uses relative space (dofs_pos = qpos - qpos0)
                # Convert absolute qpos to relative dofs_position
                dofs_pos = qpos - qpos0_offset
                robot.control_dofs_position(dofs_pos)
            else:
                robot.set_qpos(qpos)
            scene.step()
            scene.visualizer.update()

            # Print per-DOF tracking error
            if i > 0 and (i + 1) % 10 == 0:
                actual = robot.get_dofs_position().cpu().numpy().flatten()
                target = traj[i]
                diff = actual - target
                np.set_printoptions(precision=4, suppress=True, linewidth=200)
                print(f"Frame {i + 1:4d} diff: {diff}")

            # Real-time pacing
            elapsed = time.time() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            if (i + 1) % 100 == 0:
                print(f"Frame {i + 1}/{traj.shape[0]}")

        print("Trajectory complete.")
        if not args.loop:
            break

    # Keep viewer open
    while True:
        scene.step()
        scene.visualizer.update()
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()
