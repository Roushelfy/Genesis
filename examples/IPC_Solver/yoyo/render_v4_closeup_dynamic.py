"""
Render the v4 yoyo trajectory with a dynamic close-up camera:

    1. Starts at the initial yoyo pose, further back than the standard close-up.
    2. Holds still for 0.5s after the yoyo begins dropping.
    3. Then "chases" the falling yoyo, moving from far back to the close-up
       distance as it arrives at the yoyo's lowest position.
    4. Stays at the close-up pose tracking the yoyo for the rest of the replay.

Usage (Nyx):
    python examples/IPC_Solver/render_v4_closeup_dynamic.py --render --nyx --sage-bg

Usage (Luisa):
    python examples/IPC_Solver/render_v4_closeup_dynamic.py --render
"""

import sys
from pathlib import Path

# Add the IPC_Solver parent dir to sys.path so _replay_common / replay_yoyo_traj
# are importable when this script is imported from yoyo/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

if "--seq-dir" not in sys.argv and "--trajectory" not in sys.argv:
    sys.argv += ["--trajectory", "v4"]

from _replay_common import CameraTrajectory  # noqa: E402
from replay_yoyo_traj import (  # noqa: E402
    CLOSEUP_DISTANCE,
    CLOSEUP_FOV,
    YoyoReplay,
    _smooth,
)


# Far distance the camera starts at (same direction as CLOSEUP_DISTANCE, further)
FAR_DISTANCE = 0.45
# How long (seconds) the camera stays still after the yoyo begins dropping
WAIT_AFTER_DROP_SEC = 0.5
# Drop detection threshold — z must drop this much from the initial z
DROP_THRESHOLD = 0.005


# Alternating cyan / deep-blue metallic bearings (same palette as other scripts)
_CYAN = (0.20, 0.85, 1.00, 1.0)
_DEEP_BLUE = (0.05, 0.15, 0.55, 1.0)
_BEARING_PALETTE = [_CYAN, _DEEP_BLUE] * 4


class V4DynamicChaseCamera(CameraTrajectory):
    """Static → wait → chase → tracking close-up."""

    def __init__(self, rigid_data, fps, offset_dir=None):
        self._rigid_data = rigid_data
        ball_pos = rigid_data["yoyo_ball"][:, :3]

        # Detect the first frame where the yoyo starts dropping (z decreases)
        initial_z = float(ball_pos[0, 2])
        drop_start = 0
        for i in range(len(ball_pos)):
            if ball_pos[i, 2] < initial_z - DROP_THRESHOLD:
                drop_start = i
                break

        wait_frames = max(1, int(round(WAIT_AFTER_DROP_SEC * fps)))
        self._chase_start = drop_start + wait_frames

        # The chase ends at the lowest-z frame
        self._lowest_frame = int(np.argmin(ball_pos[:, 2]))
        if self._lowest_frame <= self._chase_start:
            self._lowest_frame = self._chase_start + max(1, int(0.5 * fps))

        self._offset_dir = offset_dir if offset_dir is not None else np.array([0.0, -1.0, 0.0])
        self._far_distance = FAR_DISTANCE
        self._close_distance = CLOSEUP_DISTANCE

        self._initial_ball = ball_pos[0].astype(np.float64).copy()
        self._initial_cam = self._initial_ball + self._offset_dir * self._far_distance

        print(
            f"[dynamic-chase] drop_start={drop_start} wait_frames={wait_frames} "
            f"chase_start={self._chase_start} lowest_frame={self._lowest_frame}"
        )

    def get_pose(self, frame_idx, n_frames):
        rb = self._rigid_data["yoyo_ball"]
        idx = min(frame_idx, rb.shape[0] - 1)
        ball_pos = rb[idx, :3].astype(np.float64)

        if frame_idx < self._chase_start:
            # Static: fixed position and fixed lookat (yoyo falls out of frame)
            cam_pos = self._initial_cam
            cam_lookat = self._initial_ball
        elif frame_idx < self._lowest_frame:
            # Chase: interpolate distance from far → close while tracking yoyo
            t = (frame_idx - self._chase_start) / max(self._lowest_frame - self._chase_start, 1)
            t = _smooth(t)
            distance = self._far_distance * (1 - t) + self._close_distance * t
            cam_pos = ball_pos + self._offset_dir * distance
            cam_lookat = ball_pos
        else:
            # Locked close-up tracking
            cam_pos = ball_pos + self._offset_dir * self._close_distance
            cam_lookat = ball_pos

        return tuple(cam_pos), tuple(cam_lookat)


# View angle presets: offset direction for the camera relative to the yoyo
VIEW_ANGLES = {
    "side": np.array([0.0, -1.0, 0.0]),   # along spin axis (sees yoyo face)
    "front": np.array([1.0, 0.0, 0.0]),   # perpendicular to spin axis (sees string through gap)
}


class V4CloseupDynamic(YoyoReplay):
    _base_name = "v4_closeup_dynamic"

    @property
    def name(self):
        view = getattr(self.args, "view", "side") if hasattr(self, "args") else "side"
        return f"{self._base_name}_{view}"

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            "--view", type=str, default="side", choices=list(VIEW_ANGLES.keys()),
            help="Camera view angle: 'side' (yoyo face) or 'front' (sees string through gap)",
        )
        parser.add_argument(
            "--subsample", type=int, default=1,
            help="Take every Nth frame from the trajectory data (default: 1 = all frames)",
        )

    def load_trajectory(self):
        n_frames = super().load_trajectory()
        self._force_closeup_camera = True
        self.cam_fov = CLOSEUP_FOV
        self.args.camera_traj = "_v4_dynamic_chase"

        # Subsample the loaded trajectory data if requested
        stride = self.args.subsample
        if stride > 1:
            for name in self._rigid_data:
                self._rigid_data[name] = self._rigid_data[name][::stride]
            for name in self._fem_data:
                self._fem_data[name] = self._fem_data[name][::stride]
            if self._raw_joint_data is not None:
                self._raw_joint_data = self._raw_joint_data[::stride]
            n_frames = min(len(v) for v in self._rigid_data.values()) if self._rigid_data else n_frames // stride
            print(f"[subsample] stride={stride} → {n_frames} frames")
        return n_frames

    def make_camera_traj(self, name):
        if name == "_v4_dynamic_chase":
            offset_dir = VIEW_ANGLES.get(self.args.view, VIEW_ANGLES["side"])
            return V4DynamicChaseCamera(self._rigid_data, self.fps, offset_dir=offset_dir)
        return super().make_camera_traj(name)

    def build_scene(self, scene):
        import genesis as gs

        original_add_entity = scene.add_entity

        def colored_add_entity(*args, **kwargs):
            name = kwargs.get("name", "")
            if name.startswith("bearing_sphere_"):
                try:
                    idx = int(name.split("_")[-1])
                    color = _BEARING_PALETTE[idx % len(_BEARING_PALETTE)]
                    kwargs["surface"] = gs.surfaces.BSDF(
                        color=color,
                        metallic=0.9,
                        roughness=0.15,
                    )
                except (ValueError, IndexError):
                    pass
            return original_add_entity(*args, **kwargs)

        scene.add_entity = colored_add_entity
        try:
            super().build_scene(scene)
        finally:
            scene.add_entity = original_add_entity


if __name__ == "__main__":
    V4CloseupDynamic().run()
