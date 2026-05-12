"""
Render only the frames needed for the bullet-time combined video, streaming
directly to disk (no memory buffering).

Computes which source frames are needed (stride 10 normally, stride 1 during
bullet windows), renders them one-by-one, and writes to an MP4 immediately.

Usage:
    # Orbit replay (Nyx + sage bg):
    python examples/IPC_Solver/render_bullet_time_frames.py --mode orbit --render --nyx --sage-bg

    # Close-up side (Luisa):
    python examples/IPC_Solver/render_bullet_time_frames.py --mode closeup --render
"""

import sys
import tempfile
from pathlib import Path

# Add the IPC_Solver parent dir to sys.path so _replay_common / replay_yoyo_traj
# / render_v4_closeup_dynamic are importable when this script is invoked from
# yoyo/ rather than IPC_Solver/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.interpolate import CubicSpline

_SEQ = Path(__file__).resolve().parents[3] / "DemoAssets" / "yoyo" / "v4" / "seq_full_version"

# String interpolation settings (only for closeup mode)
STRING_SUBDIVISIONS = 8
if "--seq-dir" not in sys.argv:
    sys.argv += ["--seq-dir", str(_SEQ), "--trajectory", "v4"]

import imageio  # noqa: E402

from _replay_common import TrajectoryReplay  # noqa: E402
from replay_yoyo_traj import (  # noqa: E402
    CLOSEUP_FOV,
    YoyoReplay,
)
from render_v4_closeup_dynamic import (  # noqa: E402
    V4DynamicChaseCamera,
    VIEW_ANGLES,
    _BEARING_PALETTE,
)

# Bullet-time config (must match compose_bullet_time.py).
# Windows are defined by FULL-SEQUENCE SOURCE FRAME ranges, so they don't
# shift when extra output frames are inserted by the bullet time itself.
# Each window: (src_start, src_end, stride).
# Outside windows: NORMAL_STRIDE.
NORMAL_STRIDE = 20  # matches sub10 video: 60fps render + 30fps downsample = 20 full-seq frames per output frame
SHOWCASE_FRAMES = 165
OUT_FPS = 30

# Window 1: 10× slow — same physical moment as before (full-seq 580–860)
# Window 2: 10× slow — new orig frames 400–432 → replay 184–216 → full-seq 3680–4320
BULLET_WINDOWS_SRC = [
    (580, 860, 2),  # 10× slow (stride 2 vs normal 20)
    (3680, 4220, 2),  # 10× slow (stride 2 vs normal 20), ends at orig frame 375
]


def _get_stride_for_src(src_frame: int) -> int:
    for s0, s1, stride in BULLET_WINDOWS_SRC:
        if s0 <= src_frame < s1:
            return stride
    return NORMAL_STRIDE


def compute_needed_frames(n_src: int) -> list[int]:
    """Compute the source frame indices needed for the full combined video."""
    indices = []
    src_idx = 0.0
    while src_idx < n_src:
        idx = int(round(src_idx))
        indices.append(idx)
        stride = _get_stride_for_src(idx)
        src_idx += stride
    return indices


_REPO_ROOT_BT = Path(__file__).resolve().parents[3]

# Shell logo texture (same as showcase/replay)
_SHELL_PARTS = {"yoyo-top_shell", "yoyo-bottom_shell", "yoyo-top_ring", "yoyo-bottom_ring"}
_YOYO_ASSETS = _REPO_ROOT_BT / "DemoAssets" / "yoyo" / "v3"
_LOGO_IMG = _REPO_ROOT_BT / "DemoAssets" / "yoyo" / "logo_centered.png"

# Close-up distance: original 2cm + 3cm further = 5cm
CLOSEUP_DISTANCE_FAR = 0.05


class BulletTimeRenderer(YoyoReplay):
    _base_name = "bt_render"

    @property
    def name(self):
        mode = getattr(self.args, "mode", "orbit") if hasattr(self, "args") else "orbit"
        return f"{self._base_name}_{mode}"

    # Same lighting as trashbag (Luisa settings on Nyx)
    def nyx_lights(self):
        return [
            {"type": "point", "pos": (0.85, 1.25, 2.45), "color": (1.0, 0.97, 0.92), "intensity": 20.0, "shadow": True},
            {"type": "point", "pos": (0.6, -1.7, 4.3), "color": (0.48, 0.52, 0.6), "intensity": 1.0, "shadow": False},
            {"type": "point", "pos": (-0.8, -3.16, 0.5), "color": (0.8, 0.88, 1.0), "intensity": 100.0, "shadow": True},
            {"type": "point", "pos": (0.85, 1.25, 0.0), "color": (1.0, 0.97, 0.92), "intensity": 20.0, "shadow": True},
        ]

    def nyx_light_field(self):
        return None

    def make_renderer(self):
        # Match Nyx --dark-bg setup: same g_warm_light_gray_02.exr env at the same
        # multiplier, lights at the same positions/intensities as nyx_lights()
        # scaled by 0.2 (Luisa SphereLight power ≈ Nyx point-light intensity * 0.2).
        import genesis as gs
        from genesis.options.renderers import SphereLight
        import genesis.vis.raytracer as _gr

        _gr.sphere_light_as_mesh = False  # hide SphereLight bodies from the render

        env_path = _REPO_ROOT_BT / "DemoAssets/textures/g_warm_light_gray_02.exr"
        SCALE = 0.2
        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path=str(env_path),
                    encoding="linear",
                ),
            ),
            env_radius=100.0,  # treat env as effectively infinite, like Nyx env_map
            env_euler=(0, 0, 0),
            lights=[
                SphereLight(pos=(0.85, 1.25, 2.45), radius=0.001, color=(1.0, 0.97, 0.92), intensity=20.0 * SCALE),
                SphereLight(pos=(0.6, -1.7, 4.3), radius=0.001, color=(0.48, 0.52, 0.6), intensity=1.0 * SCALE),
                SphereLight(pos=(-0.8, -3.16, 0.5), radius=0.001, color=(0.8, 0.88, 1.0), intensity=100.0 * SCALE),
                SphereLight(pos=(0.85, 1.25, 0.0), radius=0.001, color=(1.0, 0.97, 0.92), intensity=20.0 * SCALE),
            ],
        )

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument("--mode", type=str, default="orbit", choices=["orbit", "closeup"])
        parser.add_argument("--view", type=str, default="side", choices=list(VIEW_ANGLES.keys()))

    def load_trajectory(self):
        n_frames = super().load_trajectory()
        if self.args.mode == "closeup":
            self._force_closeup_camera = True
            self.cam_fov = CLOSEUP_FOV
            self.cam_near = 0.01  # 1cm near plane (default 10cm clips the close-up)
            self.args.camera_traj = "_v4_dynamic_chase"
            self._smooth_string(n_frames)
        return n_frames

    def _smooth_string(self, n_frames):
        """Interpolate the yoyo string from 195 → ~1553 vertices per frame."""
        if "yoyo_string" not in self._fem_data:
            return
        orig = self._fem_data["yoyo_string"]
        n_orig = orig.shape[1]
        n_dense = (n_orig - 1) * STRING_SUBDIVISIONS + 1
        print(f"[smooth-string] Interpolating {n_orig} → {n_dense} verts ({orig.shape[0]} frames)...")

        dense = np.zeros((orig.shape[0], n_dense, 3), dtype=orig.dtype)
        for fi in range(orig.shape[0]):
            pts = orig[fi]
            diffs = np.diff(pts, axis=0)
            seg_len = np.linalg.norm(diffs, axis=1)
            cum = np.concatenate([[0], np.cumsum(seg_len)])
            total = max(cum[-1], 1e-12)
            t = cum / total
            cs = CubicSpline(t, pts, bc_type="natural")
            dense[fi] = cs(np.linspace(0, 1, n_dense))
            if (fi + 1) % 500 == 0:
                print(f"  frame {fi + 1}/{orig.shape[0]}")

        self._fem_data["yoyo_string"] = dense

        # Write dense line-mesh OBJ for entity creation
        self._dense_string_obj = Path(tempfile.mkdtemp(prefix="smooth_str_")) / "mesh.obj"
        with open(self._dense_string_obj, "w") as f:
            for i in range(n_dense):
                f.write(f"v 0 0 {i * 0.001}\n")
            for i in range(n_dense - 1):
                f.write(f"l {i + 1} {i + 2}\n")
        print(f"[smooth-string] Done. Dense mesh: {self._dense_string_obj}")

    def make_camera_traj(self, name):
        if name == "_v4_dynamic_chase":
            offset_dir = VIEW_ANGLES.get(self.args.view, VIEW_ANGLES["side"])
            cam = V4DynamicChaseCamera(self._rigid_data, self.fps, offset_dir=offset_dir)
            cam._close_distance = CLOSEUP_DISTANCE_FAR  # 3cm further than default
            return cam
        return super().make_camera_traj(name)

    def build_scene(self, scene):
        import genesis as gs

        original_add_entity = scene.add_entity
        dense_obj = getattr(self, "_dense_string_obj", None)
        is_closeup = self.args.mode == "closeup"

        def patched_add_entity(*args, **kwargs):
            name = kwargs.get("name", "")
            # Rings always use smoothed GLB (keep their built-in material).
            # Shells use logo GLB + logo BSDF, except in the Luisa closeup case
            # where we keep _add_ball_part's transparent BSDF so the shells go
            # see-through and reveal the internals.
            use_nyx = bool(self.args.nyx)
            luisa_closeup = is_closeup and not use_nyx
            is_ring = name in ("yoyo-top_ring", "yoyo-bottom_ring")
            is_shell = name in ("yoyo-top_shell", "yoyo-bottom_shell")
            if is_ring:
                glb_file = _YOYO_ASSETS / f"{name}_smooth.glb"
                if glb_file.exists():
                    if args:
                        args = (gs.morphs.Mesh(file=str(glb_file), fixed=True, collision=False),) + args[1:]
                    else:
                        kwargs["morph"] = gs.morphs.Mesh(file=str(glb_file), fixed=True, collision=False)
            elif is_shell and not luisa_closeup:
                glb_file = _YOYO_ASSETS / f"{name}_logo.glb"
                if glb_file.exists():
                    if args:
                        args = (gs.morphs.Mesh(file=str(glb_file), fixed=True, collision=False),) + args[1:]
                    else:
                        kwargs["morph"] = gs.morphs.Mesh(file=str(glb_file), fixed=True, collision=False)
                    # Plain G-Warm light gray 03 (matches showcase), no logo PNG
                    # kwargs["surface"] = gs.surfaces.BSDF(
                    #     color=(0.55, 0.52, 0.50, 1.0),
                    #     metallic=0.3, roughness=0.4,
                    # )
            # Closeup-only: remove near-side shell (so internals visible),
            # swap string mesh, color bearings, override robot materials
            if is_closeup:
                # Robot surface override (matches replay_yoyo_v4_traj.py)
                if name == "robot":
                    kwargs["surface"] = {
                        "paint_white_glossy": gs.surfaces.BSDF(
                            color=(0.74, 0.74, 0.74),
                            roughness=0.25,
                            metallic=0.25,
                        ),
                        "plastic_black_rough": gs.surfaces.BSDF(
                            color=(0.02, 0.02, 0.03),
                            roughness=0.35,
                            metallic=0.0,
                            ior=1.45,
                        ),
                    }
                # Skip top_shell + top_ring (near-side from the side camera)
                if name in ("yoyo-top_shell", "yoyo-top_ring"):
                    return None  # don't create this entity
                if name == "yoyo_string" and dense_obj is not None:
                    if args:
                        args = (gs.morphs.Mesh(file=str(dense_obj)),) + args[1:]
                    else:
                        kwargs["morph"] = gs.morphs.Mesh(file=str(dense_obj))
                if name.startswith("bearing_sphere_"):
                    try:
                        idx = int(name.split("_")[-1])
                        # Alternating blue / cyan, opaque
                        color = (0.40, 0.50, 1.00, 1.0) if idx % 2 == 0 else (0.30, 0.80, 1.00, 1.0)
                        kwargs["surface"] = gs.surfaces.BSDF(
                            color=color,
                            metallic=0.1,
                            roughness=0.05,
                        )
                    except (ValueError, IndexError):
                        pass
            return original_add_entity(*args, **kwargs)

        scene.add_entity = patched_add_entity
        try:
            super().build_scene(scene)
        finally:
            scene.add_entity = original_add_entity

        # Drop entities that patched_add_entity skipped (returned None) so
        # apply_frame doesn't crash trying to call .set_pos on None.
        for name in list(self._rigid_entities):
            self._rigid_entities[name] = [e for e in self._rigid_entities[name] if e is not None]
            if not self._rigid_entities[name]:
                del self._rigid_entities[name]

    def run(self):
        """Override run() to stream frames to disk for only the needed indices."""
        import genesis as gs

        args = self.args
        use_nyx = args.nyx

        gs.init(backend=gs.gpu if args.render else gs.cpu, logging_level="warning")

        # Ensure attributes expected by _replay_common exist
        if not hasattr(self, "_use_luisa_preview"):
            self._use_luisa_preview = False

        self._n_frames = self.load_trajectory()
        print(f"[{self.name}] {self._n_frames} source frames")

        self._make_scene()
        self.build_scene(self._scene)

        if args.render:
            self._add_camera()

        self._scene.build(n_envs=1 if use_nyx else 0)
        self.post_build()

        self._camera_traj = None
        if args.camera_traj:
            self._camera_traj = self.make_camera_traj(args.camera_traj)

        if not args.render:
            print("No --render flag; exiting.")
            return

        # Compute needed frames
        needed = compute_needed_frames(self._n_frames)
        print(f"[{self.name}] Need {len(needed)} frames (out of {self._n_frames})")

        # Output path
        renderer_name = "nyx" if use_nyx else "luisa"
        traj = getattr(args, "trajectory", "v4")
        stem = f"{self.name}_{traj}_{renderer_name}"
        out_dir = Path(f"data/ipc_demo/ipc_{self.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.mp4"

        writer = imageio.get_writer(str(out_path), fps=self.fps, macro_block_size=1)

        for out_i, src_i in enumerate(needed):
            src_i = min(src_i, self._n_frames - 1)
            self.apply_frame(self._scene, src_i)
            self._scene._visualizer.update_visual_states(force_render=True)

            if self._camera_traj is not None:
                cam_pos, cam_lookat = self._camera_traj.get_pose(src_i, self._n_frames)
                if use_nyx:
                    self._cam.update_camera_pose(pos=cam_pos, lookat=cam_lookat, up=(0, 0, 1))
                else:
                    self._cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=(0, 0, 1))

            if use_nyx:
                self._cam._stale = True
                data = self._cam.read()
                rgb = data.rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]
            else:
                rgb_result = self._cam.render(rgb=True, force_render=True)
                rgb_tensor = rgb_result[0]
                rgb = rgb_tensor.cpu().numpy() if hasattr(rgb_tensor, "cpu") else np.array(rgb_tensor)

            writer.append_data(rgb)

            if (out_i + 1) % 50 == 0:
                tag = "BT" if _get_stride_for_src(src_i) < NORMAL_STRIDE else "  "
                print(f"  [{out_i + 1}/{len(needed)}] src={src_i} {tag}")

        writer.close()
        print(f"Saved {out_path} ({len(needed)} frames @ {self.fps} fps)")


if __name__ == "__main__":
    BulletTimeRenderer().run()
