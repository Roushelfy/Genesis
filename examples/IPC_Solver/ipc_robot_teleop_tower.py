"""
Franka Panda teleop with IPC — ring tower stacking puzzle.

Scene layout:
  - Franka Panda robot (fixed base)
  - Tower of Hanoi-style ring stack on a base pole
  - Stacking order (bottom to top): Base -> 1 -> 2 -> 3 -> 4 -> 6 -> 5 -> Ball
  - Rings are ordered by radius: 1 (largest) to 6 (smallest)

Usage:
    LD_PRELOAD=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12 \\
        python examples/IPC_Solver/ipc_robot_teleop_tower.py
    python examples/IPC_Solver/ipc_robot_teleop_tower.py --no-ipc   # pure rigid (no IPC)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import genesis as gs
from robot_teleop import RobotTeleop

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOWER_DIR = _REPO_ROOT / "DemoAssets" / "tower"
_PANDA_MJCF = "xml/franka_emika_panda/panda_non_overlap.xml"

# ── Tower geometry ───────────────────────────────────────────────────────────
# All rings have z_range [-0.0102, 0.0102], height = 0.0204 m
# Base pole top surface at z = 0.1544 m (z_range [-0.0102, 0.1544])
# Ball z_range [-0.0210, 0.0225], height ≈ 0.0435 m

RING_HEIGHT = 0.0204
BASE_TOP_Z = 0.1544
BALL_HALF_H = 0.0225

# Stacking order (bottom to top): base, ring_01..04, ring_06, ring_05, ball
# Note: ring_06 before ring_05 (smaller ring below larger — intentional puzzle state)
STACK_ORDER = ["ring_01", "ring_02", "ring_03", "ring_04", "ring_06", "ring_05", "ball"]

# Tower position on the table (in front of the robot)
TOWER_XY = (0.45, 0.0)


def _compute_stack_positions() -> dict[str, tuple[float, float, float]]:
    """Compute initial (x, y, z) for each piece so they stack correctly.

    A small gap (GAP) between pieces prevents IPC sanity-check intersections.
    """
    GAP = 0.002
    positions = {}
    tx, ty = TOWER_XY

    # Base pole sits on the ground (its bottom is at z=-0.0102, so shift up by 0.0102)
    positions["base_pole"] = (tx, ty, 0.0102)

    # Stack rings on top of the base pole
    # Ring center z = base_top + half_ring_height + gap
    z = BASE_TOP_Z + RING_HEIGHT / 2 + GAP
    for name in STACK_ORDER:
        if name == "ball":
            z_ball = z - RING_HEIGHT / 2 + BALL_HALF_H + GAP
            positions[name] = (tx, ty, z_ball)
        else:
            positions[name] = (tx, ty, z)
            z += RING_HEIGHT + GAP
    return positions


def main():
    parser = argparse.ArgumentParser(description="Panda teleop — ring tower (IPC)")
    parser.add_argument("--no-ipc", action="store_true", help="Pure rigid, no IPC coupler")
    parser.add_argument("--vis-collision", action="store_true", help="Visualize collision geometry")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="info")

    vis_surface = gs.surfaces.Default(vis_mode="collision") if args.vis_collision else None

    scene_kwargs = dict(
        sim_options=gs.options.SimOptions(dt=0.02),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -0.8, 0.8),
            camera_lookat=(0.4, 0.0, 0.2),
            camera_fov=45,
        ),
        show_viewer=True,
    )

    if not args.no_ipc:
        scene_kwargs["coupler_options"] = gs.options.IPCCouplerOptions(
            n_linesearch_iterations=8,
            newton_tolerance=1e-2,
            newton_translation_tolerance=1,
            newton_semi_implicit_enable=False,
            linear_system_tolerance=1e-3,
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            enable_rigid_ground_contact=True,
            contact_d_hat=0.0002,
            contact_resistance=1e9,
        )

    scene = gs.Scene(**scene_kwargs)

    # ── Ground plane ──
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(coup_type="ipc_only") if not args.no_ipc else None,
    )

    # ── Panda robot ──
    panda_mat_kwargs = {}
    if not args.no_ipc:
        panda_mat_kwargs["coup_type"] = "external_articulation"

    panda = scene.add_entity(
        gs.morphs.MJCF(
            file=_PANDA_MJCF,
            pos=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(**panda_mat_kwargs),
        surface=vis_surface,
    )

    # ── Tower pieces ──
    positions = _compute_stack_positions()

    tower_entities = {}
    for name in ["base_pole"] + STACK_ORDER:
        glb_path = _TOWER_DIR / f"{name}.glb"
        if not glb_path.exists():
            print(f"[warn] Missing: {glb_path.name}")
            continue
        pos = positions[name]
        mat_kwargs = {
            "rho": 50.0,
            "coup_friction": 0.1,
        }
        if not args.no_ipc:
            mat_kwargs["coup_type"] = "ipc_only"
        if name == "ball":
            scale = 1.0
            fixed = False
            convexify = False
        elif name == "base_pole":
            scale = 1.0
            fixed = True
            convexify = False
        else:
            scale = 1.0
            fixed = False
            convexify = False

        # Base pole is fixed; rings and ball are free
        ent = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(glb_path),
                pos=pos,
                scale=scale,
                file_meshes_are_zup=True,
                fixed=fixed,
                collision=True,
                convexify=convexify,
                decimate=False,
            ),
            material=gs.materials.Rigid(**mat_kwargs),
            surface=vis_surface,
            name=name,
        )
        tower_entities[name] = ent

    scene.build()

    # ── Teleop ──
    # EE starts above the tower, pointing down — IK computes the joint angles.
    tx, ty = TOWER_XY
    teleop = RobotTeleop.franka(
        scene=scene,
        robot=panda,
        init_pos=(tx, ty, 0.5),
        init_euler=(0.0, 180.0, 0.0),
    )
    teleop.setup()
    teleop.run()


if __name__ == "__main__":
    main()
