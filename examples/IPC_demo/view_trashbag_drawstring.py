"""
Genesis trashbag with drawstring rope — cloth bag closing via rope tension.

Scene:
  - Cloth trashbag with drawstring channel
  - Rope drawstring (closed loop) threaded through the channel
  - Ground plane at z=0
  - IPC contact for cloth-rope, cloth-ground, rope-ground

Meshes are Y-up; rotated 90 deg around X for Genesis Z-up convention.

Usage:
    LD_PRELOAD=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12 \
        python examples/IPC_demo/view_trashbag_drawstring.py [--no-viewer] [--steps N]
"""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ASSETS = REPO_ROOT / "DemoAssets" / "trashbag_drawstring"
BAG_MESH = DEMO_ASSETS / "Trashbag_drawstring.glb"
ROPE_MESH = DEMO_ASSETS / "rope.obj"
BAG_POS = (0.2, 0.2, 0.2)


def main():
    parser = argparse.ArgumentParser(description="Genesis trashbag drawstring (IPC)")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--use-al", action="store_true", help="Use AL-IPC contact constitution")
    parser.add_argument("--no-bag", action="store_true", help="Remove the cloth bag")
    parser.add_argument("--no-rope", action="store_true", help="Remove the drawstring rope")
    parser.add_argument("--no-plane", action="store_true", help="Remove ground plane")
    parser.add_argument("--no-gravity", action="store_true")
    parser.add_argument("--verbose-ipc", action="store_true", help="Print full libuipc log")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=not args.no_viewer,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.25),
            camera_fov=45,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0 if args.no_gravity else -9.8),
        ),
        fem_options=gs.options.FEMOptions(
            use_rigid_compatible_transform=True,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.001,
            contact_friction_enable=True,
            contact_resistance=1e8,
            # Newton solver
            newton_tolerance=0.5,
            newton_translation_tolerance=10,
            newton_min_iterations=2,
            verbose_ipc_log=args.verbose_ipc,
            # AL-IPC (opt-in via --use-al)
            **(dict(contact_constitution="al-ipc") if args.use_al else {}),
        ),
    )

    # Ground plane
    if not args.no_plane:
        scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(
                coup_type="ipc_only",
                coup_friction=0.3,
            ),
        )

    # Cloth trashbag with drawstring channel
    # GLB is loaded as Z-up by gs.Mesh (glTF Y-up converted automatically).
    # OBJ is raw Y-up and would need euler=(90,0,0).
    if not args.no_bag:
        bag = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(BAG_MESH),
                pos=(0.2, 0.2, 0.2),
                euler=(0, 0, 0),
                scale=1.0,
            ),
            material=gs.materials.FEM.Cloth(
                E=10e3,
                nu=0.4,
                rho=200.0,
                thickness=0.0005,
                bending_stiffness=20.0,
                friction_mu=0.3,
            ),
            # No surface specified — uses GLB's per-sub-mesh PBR materials
        )

    # Drawstring rope (closed loop through channel)
    if not args.no_rope:
        rope = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(ROPE_MESH),
                pos=(0.2, 0.2, 0.2),
                euler=(90, 0, 0),
                scale=1.0,
            ),
            material=gs.materials.FEM.Rope(
                E=5e5,
                rho=500.0,
                thickness=0.003,
                bending_stiffness=1e3,
                friction_mu=0.3,
            ),
            surface=gs.surfaces.Default(color=(0.8, 0.15, 0.1, 1.0)),
        )

    # Four small cubes to pinch and pull the rope at the two gap openings.
    # Uses two_way_soft_constraint so the rigid solver drives position
    # while IPC handles contact with the rope.
    cube_size = 0.02
    bag_pos = np.array(BAG_POS)
    gap_z = bag_pos[2] + 0.575  # rope gap arc Z in world
    gap_offset = 0.015  # vertical offset above/below rope for pinch cubes

    # Gap 1: +Y side of bag — place cubes outside the bag at the gap exit
    gap1_dir = np.array([0.022, 1.0])
    gap1_dir /= np.linalg.norm(gap1_dir)
    gap1_xy = bag_pos[:2] + np.array([0.004, 0.177]) + gap1_dir * 0.04
    # Gap 2: -X/-Y side of bag
    gap2_dir = np.array([-0.685, -0.729])
    gap2_dir /= np.linalg.norm(gap2_dir)
    gap2_xy = bag_pos[:2] + np.array([-0.129, -0.148]) + gap2_dir * 0.04

    grippers = []
    for gap_xy, gap_dir in [(gap1_xy, gap1_dir), (gap2_xy, gap2_dir)]:
        for z_sign in (+1, -1):
            cube = scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(gap_xy[0], gap_xy[1], gap_z + z_sign * gap_offset),
                    size=(cube_size, cube_size, cube_size),
                    fixed=False,
                ),
                material=gs.materials.Rigid(
                    rho=2000,
                    coup_type="two_way_soft_constraint",
                    coup_friction=0.8,
                    enable_coup_collision=True,
                ),
                surface=gs.surfaces.Plastic(color=(0.3, 0.3, 0.8, 1.0)),
            )
            grippers.append((cube, gap_xy, gap_dir, z_sign))

    scene.build()

    # Scripted gripper motion:
    # Phase 1 (steps 0-50): cubes close vertically to pinch the rope
    # Phase 2 (steps 50+): cubes pull outward along gap direction
    pinch_speed = 0.002
    pull_speed = 0.001
    dt = 0.01

    for step in range(args.steps):
        for cube, gap_xy, gap_dir, z_sign in grippers:
            # Set linear velocity on the free joint DOFs [vx, vy, vz, wx, wy, wz]
            dof_vel = np.zeros(6)
            if step < 50:
                dof_vel[2] = -z_sign * pinch_speed / dt
            else:
                dof_vel[0] = pull_speed * gap_dir[0] / dt
                dof_vel[1] = pull_speed * gap_dir[1] / dt
            cube.set_dofs_velocity(dof_vel)

        scene.step()
        if step % 50 == 0:
            print(f"  Step {step}")

    print("Done.")


if __name__ == "__main__":
    main()
