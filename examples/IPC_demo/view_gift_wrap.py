"""
Genesis gift_wrap scene — cloth ribbon wrapping around a rigid box.

Scene:
  - Cloth ribbon: StrainLimitingBaraffWitkinShell + DiscreteShellBending
    E=20kPa, nu=0.4, rho=200, thickness=0.0001, bending=40
    Pre-positioned around the box, rotated 90° around X (Y-up → Z-up), scale=1.0
  - Non-fixed rigid box: AffineBody kappa=100MPa, density=1000
    Pre-positioned at mesh origin, rotated 90° around X (Y-up → Z-up), scale=1.0
    Smallest edge ~10.3cm
  - Ground at z=0
  - Contact: d_hat=0.0001, friction=0.02, resistance=1e8
  - Newton: velocity_tol=0.5, transrate_tol=10, min_iter=2
  - dt=0.01, gravity=(0, 0, -9.8)

Usage:
    LD_PRELOAD=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12 \
        python examples/IPC_demo/view_gift_wrap.py [--no-viewer] [--steps N] [--use-al]
"""

import argparse
import os

import genesis as gs

DEMO_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DemoAssets", "gift_wrap")
RIBBON_MESH = os.path.join(DEMO_ASSETS, "ribbon_reordered_fine.obj")
BOX_MESH = os.path.join(DEMO_ASSETS, "box.obj")


def main():
    parser = argparse.ArgumentParser(description="Genesis gift wrap (IPC)")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--use-al", action="store_true", help="Use AL-IPC contact constitution")
    parser.add_argument("--no-box", action="store_true", help="Remove the rigid box")
    parser.add_argument("--no-ribbon", action="store_true", help="Remove the cloth ribbon")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=not args.no_viewer,
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.8),
        ),
        fem_options=gs.options.FEMOptions(
            use_rigid_compatible_transform=True,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.0001,
            contact_friction_enable=True,
            contact_resistance=1e8,
            # Newton solver
            newton_tolerance=0.5,
            newton_translation_tolerance=10,
            newton_min_iterations=2,
            # AL-IPC (opt-in via --use-al)
            **(dict(contact_constitution="al-ipc", al_ipc_toi_threshold=0.1) if args.use_al else {}),
        ),
    )

    # Ground plane at z=0 (box bottom is at z≈0.067 after rotation)
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_type="ipc_only",
            coup_friction=0.02,
        ),
    )

    # Cloth ribbon: pre-positioned around the box
    # Meshes are Y-up; rotate 90° around X to convert to Genesis Z-up
    if not args.no_ribbon:
        ribbon = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=RIBBON_MESH,
                scale=1.0,
                euler=(90, 0, 0),
            ),
            material=gs.materials.FEM.Cloth(
                E=20e3,
                nu=0.4,
                rho=200.0,
                thickness=0.0001,
                bending_stiffness=40.0,
                friction_mu=0.02,
            ),
            surface=gs.surfaces.Default(color=(0.8, 0.1, 0.1, 1.0)),
        )

    # Non-fixed rigid box, same Y-up → Z-up rotation
    if not args.no_box:
        box = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=BOX_MESH,
                scale=1.0,
                euler=(90, 0, 0),
                convexify=False,
                decimate=False,
            ),
            material=gs.materials.Rigid(
                rho=1000.0,
                coup_type="ipc_only",
                coup_friction=0.02,
                enable_coup_collision=True,
            ),
            surface=gs.surfaces.Default(color=(0.9, 0.85, 0.6, 1.0)),
        )

    scene.build()

    for step in range(args.steps):
        scene.step()
        if step % 50 == 0:
            print(f"  Step {step}")

    print("Done.")


if __name__ == "__main__":
    main()
