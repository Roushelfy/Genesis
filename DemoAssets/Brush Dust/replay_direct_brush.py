"""Replay a recorded direct_teleop_brush trajectory.

Loads trajectory.npz from a recording session,
replays brush qpos in Genesis, and optionally saves a new video.

Usage:
    python scripts/replay_direct_brush.py recordings/direct_brush_20260310_050000
    python scripts/replay_direct_brush.py recordings/direct_brush_20260310_050000 --video_output replay.mp4
    python scripts/replay_direct_brush.py recordings/direct_brush_20260310_050000 --speed 2.0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import genesis as gs
import numpy as np
import torch

from gs_env_schemas.sim.objects.registry import MORPH_TABLE_IPC
from gs_env_sim.utils.file import get_file

BRUSH_GLB = "/home/zhehuan/blenderkit_data/models/mr-diy-dual-clea_a0ea1b5e-0755-4b8c-9082-d82dc8537637/cleaning_brush.glb"


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay direct brush teleop trajectory")
    p.add_argument("session_dir", type=str, help="Path to recording session directory")
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--video_output", type=str, default=None, help="Save replay video to this path")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--loop", action="store_true", help="Loop playback")
    return p


def main() -> None:
    args = make_parser().parse_args()

    session_dir = Path(args.session_dir)
    npz_path = session_dir / "trajectory.npz"
    if not npz_path.exists():
        print(f"[ERROR] {npz_path} not found")
        return

    # Load trajectory
    data = np.load(str(npz_path))
    timestamps = data["timestamps"]
    brush_qpos_all = data["brush_qpos"]
    rate_hz = float(data["rate_hz"])
    n_frames = len(timestamps)

    print(f"[INFO] Loaded {n_frames} frames at {rate_hz} Hz from {session_dir}")
    print(f"[INFO] Duration: {timestamps[-1]:.1f}s")
    print(f"[INFO] Brush qpos shape: {brush_qpos_all.shape}")

    # Init Genesis
    backend = gs.gpu if args.backend == "gpu" else gs.cpu  # type: ignore[attr-defined]
    gs.init(backend=backend, precision="64")

    ego_pos = (0.093883, -0.0115, 1.3985)
    ego_lookat = (0.77535, -0.0115, 0.68040)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / rate_hz, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(rate_hz * args.speed),
            camera_pos=ego_pos,
            camera_lookat=ego_lookat,
            camera_fov=59,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=get_file(MORPH_TABLE_IPC.file_asset),
            pos=MORPH_TABLE_IPC.pos,
            scale=MORPH_TABLE_IPC.scale,
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )

    brush = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=BRUSH_GLB,
            pos=(0.55, 0.0, 0.885),
            euler=(0, 0, 180),
            scale=1.0,
            fixed=False,
        ),
        material=gs.materials.Rigid(rho=800.0, friction=0.5),
    )

    # Camera for video output
    video_writer = None
    cam = None
    cam_w, cam_h = 640, 480
    if args.video_output:
        cam = scene.add_camera(
            res=(cam_w, cam_h),
            pos=ego_pos,
            lookat=ego_lookat,
            fov=59,
        )

    scene.build()

    if args.video_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.video_output, fourcc, rate_hz * args.speed, (cam_w, cam_h))

    print(f"[INFO] Replaying at {args.speed}x speed. {'Looping.' if args.loop else 'Press Ctrl+C to stop.'}")

    try:
        while True:
            for i in range(n_frames):
                brush.set_qpos(torch.tensor(brush_qpos_all[i], dtype=torch.float64).unsqueeze(0))
                scene.step()

                if video_writer is not None and cam is not None:
                    rgb = cam.render(rgb=True)[0]
                    if isinstance(rgb, torch.Tensor):
                        rgb = rgb.cpu().numpy()
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr)

                if (i + 1) % 300 == 0:
                    print(f"[REPLAY] frame {i + 1}/{n_frames} ({timestamps[i]:.1f}s)")

            if not args.loop:
                break
            print("[REPLAY] Looping...")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Video saved to {args.video_output}")


if __name__ == "__main__":
    main()
