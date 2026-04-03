"""Directly teleop a cleaning brush with Oculus right controller (no robot).

Right controller → brush. Dust particles on the table.
Press A (right primary) to start recording, press A again to stop.
Records: brush qpos & qvel per frame, gallery video.

Usage:
    python scripts/direct_teleop_brush.py [--port 8051] [--backend cpu]
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import genesis as gs
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gs_env_schemas.sim.objects.registry import MORPH_TABLE_IPC
from gs_env_sim.utils.file import get_file

# New brush asset path
BRUSH_GLB = (
    "/home/zhehuan/blenderkit_data/models/mr-diy-dual-clea_a0ea1b5e-0755-4b8c-9082-d82dc8537637/cleaning_brush.glb"
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct Oculus teleop of cleaning brush")
    p.add_argument("--port", type=int, default=8051, help="Oculus UDP port")
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--rate", type=float, default=60, help="Sim rate Hz")
    p.add_argument("--output_dir", type=str, default="recordings", help="Output directory")
    return p


# ── Oculus UDP receiver ──────────────────────────────────────────────────────


class OculusReceiver:
    """Minimal threaded UDP receiver for Oculus controller poses."""

    def __init__(self, port: int = 8051) -> None:
        self.port = port
        self.latest: dict | None = None  # type: ignore[type-arg]
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._sock: socket.socket | None = None
        self._prev_button_a = False

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(0.1)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[Oculus] Listening on UDP port {self.port}...")

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65536)
                msg = json.loads(data.decode())
                with self._lock:
                    self.latest = msg
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sock:
            self._sock.close()

    def get_pose_matrix(self, side: str) -> np.ndarray | None:
        """Get 4x4 transform matrix in Oculus right-handed frame, or None."""
        with self._lock:
            if self.latest is None:
                return None
            side_data = self.latest.get(side)
        if not isinstance(side_data, dict):
            return None
        pos = side_data.get("position")
        rot = side_data.get("rotation")
        if not isinstance(pos, list) or not isinstance(rot, list):
            return None
        pos_rh = np.array([pos[0], pos[1], -pos[2]], dtype=np.float64)
        quat_xyzw_rh = np.array([-rot[0], -rot[1], rot[2], rot[3]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(quat_xyzw_rh).as_matrix()
        T[:3, 3] = pos_rh
        return T

    def get_trigger(self, side: str) -> float:
        with self._lock:
            if self.latest is None:
                return 0.0
            side_data = self.latest.get(side)
        if not isinstance(side_data, dict):
            return 0.0
        return float(side_data.get("trigger", 0.0))

    def button_a_toggled(self) -> bool:
        """Returns True on rising edge of button A (right primaryButton)."""
        with self._lock:
            if self.latest is None:
                return False
            right = self.latest.get("right")
        if not isinstance(right, dict):
            return False
        pressed = float(right.get("primaryButton", 0)) > 0.5
        toggled = pressed and not self._prev_button_a
        self._prev_button_a = pressed
        return toggled


# ── Coordinate transform (same as gs-core signal_composers.py) ──────────────

R = Rotation.from_euler("ZYX", [-np.pi / 2, 0, np.pi / 2]).as_matrix()
R_T = R.T


def rotmat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(m).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


# ── Recorder ─────────────────────────────────────────────────────────────────


class Recorder:
    def __init__(self, output_dir: str, rate_hz: float) -> None:
        self.output_dir = Path(output_dir)
        self.rate_hz = rate_hz
        self.recording = False
        self._brush_qpos: list[np.ndarray] = []
        self._brush_qvel: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._video_writer: cv2.VideoWriter | None = None
        self._session_dir: Path | None = None
        self._start_time: float = 0.0

    def start_recording(self, width: int, height: int) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = self.output_dir / f"direct_brush_{timestamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(self._session_dir / "gallery.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(video_path, fourcc, self.rate_hz, (width, height))
        self._brush_qpos = []
        self._brush_qvel = []
        self._timestamps = []
        self._start_time = time.time()
        self.recording = True
        print(f"[REC] Started recording → {self._session_dir}")

    def record_frame(self, brush_entity, gallery_frame: np.ndarray) -> None:
        if not self.recording:
            return
        self._timestamps.append(time.time() - self._start_time)
        self._brush_qpos.append(brush_entity.get_qpos().cpu().numpy().flatten())
        try:
            self._brush_qvel.append(brush_entity.get_dofs_velocity().cpu().numpy().flatten())
        except Exception:
            self._brush_qvel.append(np.zeros(6))
        if self._video_writer is not None:
            bgr = cv2.cvtColor(gallery_frame, cv2.COLOR_RGB2BGR)
            self._video_writer.write(bgr)

    def stop_recording(self) -> None:
        if not self.recording:
            return
        self.recording = False
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        assert self._session_dir is not None
        np.savez(
            str(self._session_dir / "trajectory.npz"),
            timestamps=np.array(self._timestamps),
            brush_qpos=np.array(self._brush_qpos),
            brush_qvel=np.array(self._brush_qvel),
            rate_hz=self.rate_hz,
        )
        n_frames = len(self._timestamps)
        print(f"[REC] Stopped. {n_frames} frames saved to {self._session_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = make_parser().parse_args()

    backend = gs.gpu if args.backend == "gpu" else gs.cpu  # type: ignore[attr-defined]
    gs.init(backend=backend, precision="64")

    ego_pos = (0.093883, -0.0115, 1.3985)
    ego_lookat = (0.77535, -0.0115, 0.68040)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / args.rate, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(args.rate),
            camera_pos=ego_pos,
            camera_lookat=ego_lookat,
            camera_fov=59,
        ),
        show_viewer=True,
    )

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Table
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=get_file(MORPH_TABLE_IPC.file_asset),
            pos=MORPH_TABLE_IPC.pos,
            scale=MORPH_TABLE_IPC.scale,
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )

    # Brush — new cleaning_brush.glb, free joint
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

    # Gallery camera
    cam_w, cam_h = 640, 480
    gallery_cam = scene.add_camera(
        res=(cam_w, cam_h),
        pos=(1.0, 0.0, 1.65),
        lookat=(0.45, 0.0, 0.95),
        fov=75,
    )

    scene.build()

    print(f"[MASS] brush: {brush.get_mass()} kg")

    # Initial brush pose
    brush_init_qpos = brush.get_qpos().cpu().numpy().flatten()
    print(f"[INIT] brush qpos: {brush_init_qpos}")
    brush_init_pos = brush_init_qpos[:3].copy()
    brush_init_rot = Rotation.from_quat(
        [brush_init_qpos[4], brush_init_qpos[5], brush_init_qpos[6], brush_init_qpos[3]]
    ).as_matrix()

    # Start Oculus
    oculus = OculusReceiver(port=args.port)
    oculus.start()
    recorder = Recorder(output_dir=args.output_dir, rate_hz=args.rate)

    print("[INFO] Waiting for Oculus data...")
    while oculus.latest is None:
        time.sleep(0.1)
    print("[INFO] Oculus connected!")

    # Capture initial right controller pose
    initial_right_T = None
    while initial_right_T is None:
        initial_right_T = oculus.get_pose_matrix("right")
        time.sleep(0.01)

    initial_right_pos = initial_right_T[:3, 3].copy()
    initial_right_rot = initial_right_T[:3, :3].copy()

    print(f"[INFO] Initial right pos: {initial_right_pos}")
    print("[INFO] Teleop started! Right=brush. Press A to record.")

    step = 0
    try:
        while True:
            # Check button A toggle
            if oculus.button_a_toggled():
                if recorder.recording:
                    recorder.stop_recording()
                else:
                    recorder.start_recording(cam_w, cam_h)

            # Right controller → brush
            right_T = oculus.get_pose_matrix("right")
            if right_T is not None:
                target_pos = R @ (right_T[:3, 3] - initial_right_pos) + brush_init_pos
                delta_rot = R @ right_T[:3, :3] @ initial_right_rot.T @ R_T
                target_rot = delta_rot @ brush_init_rot
                target_quat = rotmat_to_quat_wxyz(target_rot)
                new_qpos = np.concatenate([target_pos, target_quat])
                brush.set_qpos(torch.tensor(new_qpos, dtype=torch.float64).unsqueeze(0))

            scene.step()

            # Record
            if recorder.recording:
                rgb = gallery_cam.render(rgb=True)[0]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                recorder.record_frame(brush, rgb)

            step += 1
            if step % 300 == 0:
                rec_status = " [RECORDING]" if recorder.recording else ""
                print(f"[step {step}]{rec_status}")

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        if recorder.recording:
            recorder.stop_recording()
    finally:
        oculus.stop()


if __name__ == "__main__":
    main()
