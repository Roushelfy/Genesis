"""
Yoyo showcase: exploded view with orbiting camera.

Renders a close-up of the yoyo with transparent shells, animates the parts
splitting apart (exploded view) and merging back, while the camera orbits
360° around the yoyo and ends at the ipc_robot_yoyo replay camera pose.

Usage:
    python ipc_show_yoyo.py --render data/ipc_demo/ipc_yoyo/yoyo_show.mp4
    python ipc_show_yoyo.py           # interactive viewer
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from _yoyo_common import (
    REPLAY_CAMERA_FOV,
    REPLAY_CAMERA_LOOKAT,
    REPLAY_CAMERA_POS,
    REPLAY_CAMERA_UP,
    SHELL_METALLIC,
    SHELL_OPACITY_TRANSPARENT,
    SHELL_ROUGHNESS,
    YOYO_ASSETS_DIR,
    make_raytracer,
)

# ── Yoyo pose (matches frame-0 of IPC simulation) ──

YOYO_CENTER = np.array([0.256, 0.008, -0.056])
YOYO_QUAT_WXYZ = np.array([0.7071068, 0.0, -0.7071068, 0.0])

# ── Camera ──

# Close-up orbit radius (camera orbits AT this distance from yoyo center)
ORBIT_RADIUS = 0.12
# Orbit elevation angle (radians above the yoyo's equator plane)
ORBIT_ELEVATION = math.radians(25)

# Replay camera (= final pose, = start of ipc_robot_yoyo)
_REPLAY_POS = np.array(REPLAY_CAMERA_POS)
_REPLAY_LOOKAT = np.array(REPLAY_CAMERA_LOOKAT)
_REPLAY_UP = np.array(REPLAY_CAMERA_UP)

# ── Animation timeline (seconds at FPS) ──

FPS = 30
PHASE_ASSEMBLED = 2.0
PHASE_EXPLODE = 3.0
PHASE_HOLD = 2.5
PHASE_MERGE = 3.0
PHASE_PULLBACK = 3.5
# Total ~14s

# ── Explosion offsets along spin axis ──

from scipy.spatial.transform import Rotation as _Rot


def _quat_to_R(quat_wxyz):
    w, x, y, z = quat_wxyz
    return _Rot.from_quat([x, y, z, w]).as_matrix()


# Spin axis in world frame
_SPIN_AXIS = _quat_to_R(YOYO_QUAT_WXYZ) @ np.array([0.0, 0.0, 1.0])

# Top shell + top ring move together; bottom shell + bottom ring move together
EXPLODE_TOP = 0.035
EXPLODE_BOTTOM = -0.035
# Axle and hub stay at center (static)
EXPLODE_BEARING_OUTER = 0.018
EXPLODE_BEARING_SPHERE = 0.028


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _orbit_pos(angle_rad, radius, center, elevation):
    """Camera position orbiting around center.

    angle_rad: azimuth (0 = +X from center)
    elevation: angle above the equator plane (radians)
    """
    r_horiz = radius * math.cos(elevation)
    z_off = radius * math.sin(elevation)
    return center + np.array(
        [
            r_horiz * math.cos(angle_rad),
            r_horiz * math.sin(angle_rad),
            z_off,
        ]
    )


def _find_asset(name):
    p = YOYO_ASSETS_DIR / f"{name}.glb"
    if p.exists():
        return str(p)
    p = YOYO_ASSETS_DIR / f"{name}.obj"
    if p.exists():
        return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="Yoyo showcase: exploded view + orbit camera.")
    parser.add_argument("--render", type=str, default=None, metavar="FILE")
    parser.add_argument("--spp", type=int, default=128)
    args = parser.parse_args()

    import genesis as gs

    use_raytracer = args.render is not None
    gs.init(backend=gs.gpu if use_raytracer else gs.cpu, logging_level="warning")

    renderer_kwargs = {}
    if use_raytracer:
        renderer_kwargs["renderer"] = make_raytracer()

    # Compute start angle from replay camera angle (orbit ends here)
    replay_angle = math.atan2(
        _REPLAY_POS[1] - YOYO_CENTER[1],
        _REPLAY_POS[0] - YOYO_CENTER[0],
    )
    start_angle = replay_angle + math.pi  # start opposite side
    start_pos = _orbit_pos(start_angle, ORBIT_RADIUS, YOYO_CENTER, ORBIT_ELEVATION)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / FPS, gravity=(0, 0, 0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, enable_self_collision=False),
        vis_options=gs.options.VisOptions(ambient_light=(0.3, 0.3, 0.35)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=tuple(start_pos),
            camera_lookat=tuple(YOYO_CENTER),
            camera_fov=REPLAY_CAMERA_FOV,
        ),
        show_viewer=not use_raytracer,
        **renderer_kwargs,
    )

    # ── Load yoyo parts (each as separate entity for exploded view) ──

    def _add(name, surface, asset_name=None):
        return scene.add_entity(
            morph=gs.morphs.Mesh(file=_find_asset(asset_name or name), fixed=True, collision=False),
            material=gs.materials.Rigid(),
            surface=surface,
            name=name,
        )

    # Shells: BSDF with metallic + roughness + transparency
    shell_srf = gs.surfaces.BSDF(
        metallic_texture=gs.textures.ColorTexture(color=(SHELL_METALLIC,)),
        roughness_texture=gs.textures.ColorTexture(color=(SHELL_ROUGHNESS,)),
        opacity_texture=gs.textures.ColorTexture(color=(SHELL_OPACITY_TRANSPARENT,)),
    )

    # Shells + rings: move together during explosion
    top_shell = _add("top_shell", shell_srf, "yoyo-top_shell")
    bottom_shell = _add("bottom_shell", shell_srf, "yoyo-bottom_shell")
    # Non-shell parts: use GLB's baked PBR materials
    top_ring = _add("top_ring", None, "yoyo-top_ring")
    bottom_ring = _add("bottom_ring", None, "yoyo-bottom_ring")

    # Axle + hub: static at center (no animation)
    axle = _add("axle", None, "yoyo-axle")
    hub = _add("hub", None, "yoyo-hub")

    # Bearings: use GLB materials
    bearing_outer = _add("bearing_outer", None)
    bearing_spheres = []
    for i in range(8):
        sp = _find_asset(f"bearing_sphere_{i}")
        if sp is None:
            continue
        bearing_spheres.append(_add(f"bearing_sphere_{i}", None))

    cam = None
    if use_raytracer:
        cam = scene.add_camera(
            res=(1920, 1080),
            pos=tuple(start_pos),
            lookat=tuple(YOYO_CENTER),
            fov=REPLAY_CAMERA_FOV,
            spp=args.spp,
        )

    scene.build()

    # ── Pose helpers ──

    def _set_part(ent, offset):
        ent.set_pos(YOYO_CENTER + _SPIN_AXIS * offset)
        ent.set_quat(YOYO_QUAT_WXYZ)

    def _set_poses(t):
        """Set all part poses. t=0: assembled, t=1: fully exploded."""
        # Shells move the full distance
        _set_part(top_shell, EXPLODE_TOP * t)
        _set_part(bottom_shell, EXPLODE_BOTTOM * t)
        # Rings move half the shell distance (between shell and axle)
        _set_part(top_ring, EXPLODE_TOP * 0.5 * t)
        _set_part(bottom_ring, EXPLODE_BOTTOM * 0.5 * t)
        # Axle + hub: always static at center
        _set_part(axle, 0.0)
        _set_part(hub, 0.0)
        # Bearings
        _set_part(bearing_outer, EXPLODE_BEARING_OUTER * t)
        for ent in bearing_spheres:
            _set_part(ent, EXPLODE_BEARING_SPHERE * t)

    # ── Frame counts ──
    n_assembled = int(PHASE_ASSEMBLED * FPS)
    n_explode = int(PHASE_EXPLODE * FPS)
    n_hold = int(PHASE_HOLD * FPS)
    n_merge = int(PHASE_MERGE * FPS)
    n_pullback = int(PHASE_PULLBACK * FPS)
    n_orbit = n_assembled + n_explode + n_hold + n_merge
    n_total = n_orbit + n_pullback

    if use_raytracer:
        output_path = Path(args.render)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cam.start_recording()

    for frame in range(n_total):
        # ── Explosion parameter ──
        if frame < n_assembled:
            explode_t = 0.0
        elif frame < n_assembled + n_explode:
            explode_t = _smooth((frame - n_assembled) / n_explode)
        elif frame < n_assembled + n_explode + n_hold:
            explode_t = 1.0
        elif frame < n_orbit:
            explode_t = 1.0 - _smooth((frame - n_assembled - n_explode - n_hold) / n_merge)
        else:
            explode_t = 0.0

        _set_poses(explode_t)

        # ── Camera ──
        if frame < n_orbit:
            # Orbit: full 360° around yoyo
            frac = frame / n_orbit
            angle = start_angle + frac * 2 * math.pi
            cam_pos = _orbit_pos(angle, ORBIT_RADIUS, YOYO_CENTER, ORBIT_ELEVATION)
            cam_lookat = YOYO_CENTER.copy()
        else:
            # Pullback: orbit end → exact replay camera pose
            pullback_frame = frame - n_orbit
            # Use n_pullback-1 so the last frame (pullback_frame == n_pullback-1) gives t=1.0
            t = _smooth(pullback_frame / max(n_pullback - 1, 1))
            orbit_end_pos = _orbit_pos(start_angle, ORBIT_RADIUS, YOYO_CENTER, ORBIT_ELEVATION)
            cam_pos = _lerp(orbit_end_pos, _REPLAY_POS, t)
            cam_lookat = _lerp(YOYO_CENTER, _REPLAY_LOOKAT, t)

        cam_up = _REPLAY_UP  # constant throughout (Z-up)
        if use_raytracer:
            cam.set_pose(pos=tuple(cam_pos), lookat=tuple(cam_lookat), up=tuple(cam_up))
        else:
            scene.viewer.set_camera_pose(pos=tuple(cam_pos), lookat=tuple(cam_lookat))

        scene.step()

        if use_raytracer:
            cam.render(rgb=True)
            if frame % 30 == 0:
                print(f"[show] Frame {frame}/{n_total} (explode={explode_t:.2f})")

    # Verify final camera matches replay exactly (position + lookat + up)
    pose_match = (
        np.allclose(cam_pos, _REPLAY_POS, atol=1e-6)
        and np.allclose(cam_lookat, _REPLAY_LOOKAT, atol=1e-6)
        and np.allclose(cam_up, _REPLAY_UP, atol=1e-6)
    )
    print(f"[show] Final  pos={tuple(np.round(cam_pos, 4))} lookat={tuple(np.round(cam_lookat, 4))} up={tuple(cam_up)}")
    print(
        f"[show] Replay pos={tuple(np.round(_REPLAY_POS, 4))} lookat={tuple(np.round(_REPLAY_LOOKAT, 4))} up={tuple(_REPLAY_UP)}"
    )
    print(f"[show] Pose match: {pose_match}")

    if use_raytracer:
        # Save final camera transform for seam verification with ipc_robot_yoyo
        final_transform = cam.transform
        seam_path = output_path.parent / "_show_final_cam_transform.npy"
        np.save(str(seam_path), final_transform)
        print(f"[show] Saved camera transform to {seam_path.name}")

        cam.stop_recording(save_to_filename=str(output_path), fps=FPS)
        print(f"[show] Saved {output_path} ({n_total} frames, {n_total / FPS:.1f}s)")
    else:
        print(f"[show] Finished {n_total} frames")


if __name__ == "__main__":
    main()
