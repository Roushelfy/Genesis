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

import genesis as gs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ASSETS = REPO_ROOT / "DemoAssets" / "trashbag_drawstring"
BAG_MESH = DEMO_ASSETS / "Trashbag_drawstring.glb"
ROPE_MESH = DEMO_ASSETS / "rope.obj"


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
            **(dict(contact_constitution="al-ipc", al_ipc_toi_threshold=0.1) if args.use_al else {}),
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

    scene.build()

    for step in range(args.steps):
        scene.step()
        if step % 50 == 0:
            print(f"  Step {step}")

    print("Done.")


if __name__ == "__main__":
    main()
