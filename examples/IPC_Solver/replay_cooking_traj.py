"""
Replay a cooking teleop trajectory from JSON.

Usage:
    python examples/IPC_Solver/replay_cooking_traj.py \
        --traj ../gs-core/tmp/teleop/20260405_013254_149627_collect/trajectory.json \
        [--loop] [--speed 1.0] [--no-ipc]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import genesis as gs

# Asset paths (relative to Genesis-IPC repo root)
_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "DemoAssets"
MARVIN_URDF = str(_DEMO / "marvin_robot/urdf/marvin_pika.urdf")
PAN_USD = str(_DEMO / "cook/Pan025/Pan025.usd")
SPATULA_USD = str(_DEMO / "cook/Spatula018/Spatula018.usd")
BEEF_GLB = str(_DEMO / "cook/beef.glb")


def main():
    parser = argparse.ArgumentParser(description="Replay cooking teleop trajectory")
    _default_traj = str(_DEMO / "cook/trajectories/cooking_keyframed_v1.json")
    parser.add_argument("--traj", type=str, default=_default_traj, help="Path to trajectory.json")
    parser.add_argument("--loop", action="store_true", help="Loop replay")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--no-ipc", action="store_true", help="Disable IPC (faster, no contact)")
    args = parser.parse_args()

    with open(args.traj) as f:
        data = json.load(f)
    frames = data["frames"]
    print(f"Loaded {len(frames)} frames from {args.traj}")
    if frames:
        entities_in_traj = [k for k in frames[0] if k != "sim_time"]
        print(f"Entities: {entities_in_traj}")
        print(f"Time range: {frames[0]['sim_time']:.3f} - {frames[-1]['sim_time']:.3f}s")

    gs.init(backend=gs.gpu)

    use_ipc = not args.no_ipc

    scene_kwargs = dict(
        sim_options=gs.options.SimOptions(dt=1 / 60, gravity=(0, 0, -9.8)),
        rigid_options=gs.options.RigidOptions(gravity=(0, 0, -9.8)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -0.5, 1.5),
            camera_lookat=(0.6, 0.0, 0.85),
            camera_fov=40,
        ),
        show_viewer=True,
    )
    if use_ipc:
        scene_kwargs["coupler_options"] = gs.options.IPCCouplerOptions(
            contact_d_hat=5e-4,
            contact_friction_enable=True,
        )
        scene_kwargs["fem_options"] = gs.options.FEMOptions(
            use_rigid_compatible_transform=True,
        )

    scene = gs.Scene(**scene_kwargs)

    # Ground
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(coup_type="ipc_only") if use_ipc else None,
    )

    # Robot (only if trajectory has robot data)
    robot = None
    has_robot = frames and "robot" in frames[0]
    if has_robot:
        robot = scene.add_entity(
            gs.morphs.URDF(
                file=MARVIN_URDF,
                pos=(0, 0, 1.08),
                collision=False,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
            vis_mode="visual",
        )

    # Pan
    coup = "two_way_soft_constraint" if use_ipc else None
    pan = scene.add_entity(
        gs.morphs.USD(
            file=PAN_USD,
            pos=(0.59, 0.0, 0.83),
            euler=(0, 0, -90),
            scale=1.2,
            fixed=False,
            convexify=False,
            decimate=True,
        ),
        material=gs.materials.Rigid(
            rho=100.0,
            friction=0.5,
            coup_friction=0.5,
            coup_type=coup,
            gravity_compensation=1.0,
        ),
    )

    # Spatula
    spatula = scene.add_entity(
        gs.morphs.USD(
            file=SPATULA_USD,
            pos=(0.49, 0.35, 0.80),
            euler=(0, 0, -90),
            scale=1.1,
            fixed=False,
            convexify=False,
            decimate=True,
        ),
        material=gs.materials.Rigid(
            rho=100.0,
            friction=0.5,
            coup_friction=0.5,
            coup_type=coup,
            gravity_compensation=1.0,
        ),
    )

    # Meat x2
    meat_kwargs = dict(
        material=gs.materials.Rigid(
            rho=500.0,
            friction=0.8,
            coup_friction=0.8,
            coup_type="ipc_only" if use_ipc else None,
        ),
    )
    meat_0 = scene.add_entity(
        gs.morphs.Mesh(
            file=BEEF_GLB,
            pos=(0.55, -0.02, 0.82),
            euler=(90, 0, 0),
            scale=1.0,
            fixed=False,
            convexify=True,
            decimate=True,
            decimate_face_num=100,
        ),
        **meat_kwargs,
    )
    meat_1 = scene.add_entity(
        gs.morphs.Mesh(
            file=BEEF_GLB,
            pos=(0.63, 0.04, 0.82),
            euler=(90, 0, 45),
            scale=1.0,
            fixed=False,
            convexify=True,
            decimate=True,
            decimate_face_num=100,
        ),
        **meat_kwargs,
    )

    scene.build()
    scene.step()

    entities = {
        "pan": pan,
        "spatula": spatula,
        "meat_0": meat_0,
        "meat_1": meat_1,
    }
    if robot is not None:
        entities["robot"] = robot
    skip_keys = {"table", "ground", "sim_time"}

    # Robot: free base (7) + joints (18) = 25 qpos
    base_qpos = np.array([0.0, 0.0, 1.08, 1.0, 0.0, 0.0, 0.0])
    if robot is not None:
        print(f"Robot: n_qs={robot.n_qs}, n_dofs={robot.n_dofs}")

    # Replay
    dt_frame = 1.0 / 60 / args.speed

    while True:
        scene.reset()

        for i, frame in enumerate(frames):
            t_start = time.time()

            # Robot
            if robot is not None and "robot" in frame:
                joint_qpos = np.array(frame["robot"]["qpos"], dtype=np.float32)
                full_qpos = np.concatenate([base_qpos, joint_qpos]).astype(np.float32)
                robot.set_qpos(torch.from_numpy(full_qpos).unsqueeze(0).cuda())

            # Objects
            for name, obj_data in frame.items():
                if name in skip_keys or name == "robot":
                    continue
                if name not in entities:
                    continue
                entity = entities[name]
                if "pos" in obj_data:
                    entity.set_pos(obj_data["pos"])
                if "quat" in obj_data:
                    entity.set_quat(obj_data["quat"])

            scene.step()
            scene.visualizer.update()

            # Pacing
            elapsed = time.time() - t_start
            if elapsed < dt_frame:
                time.sleep(dt_frame - elapsed)

            if (i + 1) % 100 == 0:
                print(f"Frame {i + 1}/{len(frames)}")

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
