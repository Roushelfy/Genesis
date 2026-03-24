"""
Genesis version of trashbag.py — identical scene setup via IPC coupler.

Original libuipc scene:
  - Cloth bag: StrainLimitingBaraffWitkinShell + DiscreteShellBending
    E=20kPa, nu=0.4, rho=200, thickness=0.001, bending=40
    pos=(0.5, 0.0, 1.0), rotation=90° around X
  - Fixed rigid trash can: AffineBody kappa=100MPa, density=20
    pos=(0.5, 0.0, 0.5)
  - Ground at z=-0.001
  - AL-IPC contact: d_hat=0.001, friction=0.02, resistance=1e8
    toi_threshold=0.1, mu_scale=default (5e6)
  - Newton: velocity_tol=0.5, transrate_tol=10, min_iter=2
  - dt=0.01, gravity=(0, 0, -9.8)

Usage:
    LD_PRELOAD=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12 \
        python examples/IPC_demo/view_trashbag.py [--no-viewer] [--steps N]
"""

import argparse
from pathlib import Path

import genesis as gs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ASSETS = REPO_ROOT / "DemoAssets" / "trashbag"
BAG_MESH = DEMO_ASSETS / "Trashbag_coarse.obj"
CAN_MESH = DEMO_ASSETS / "trashcan_128.obj"


def main():
    parser = argparse.ArgumentParser(description="Genesis trashbag (AL-IPC)")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--use-al", action="store_true", help="Use AL-IPC contact constitution")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=not args.no_viewer,
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.8),
        ),
        fem_options=gs.options.FEMOptions(
            # libuipc applies scale→rotate→translate; Genesis legacy FEM bakes pos into
            # vertices first then rotates around COM. This flag fixes the order.
            use_rigid_compatible_transform=True,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            # Contact (matching libuipc trashbag.py exactly)
            contact_d_hat=0.001,
            contact_friction_enable=True,
            contact_resistance=1e8,
            # Newton solver
            newton_tolerance=0.5,
            newton_translation_tolerance=10,
            newton_min_iterations=2,
            # AL-IPC (opt-in via --use-al)
            **(dict(contact_constitution="al-ipc") if args.use_al else {}),
        ),
    )

    # Ground plane at z=-0.001 (matching libuipc: ground(-0.001, UnitZ))
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, -0.001)),
        material=gs.materials.Rigid(
            coup_type="ipc_only",
            coup_friction=0.02,
        ),
    )

    # Cloth bag: identical to libuipc trashbag.py
    # Original: pos=(0.5, 0, 1.0), rotation=90° around X, scale=1.0
    # E=20kPa, nu=0.4, rho=200, thickness=0.001, bending_stiffness=40
    # Global friction=0.02 from contact_tabular default_model
    bag = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(BAG_MESH),
            pos=(0.5, 0.0, 1.0),
            euler=(90, 0, 0),
            scale=1.0,
        ),
        material=gs.materials.FEM.Cloth(
            E=20e3,
            nu=0.4,
            rho=200.0,
            thickness=0.001,
            bending_stiffness=40.0,
            friction_mu=0.02,
        ),
    )

    # Fixed rigid trash can: AffineBody kappa=100MPa, density=20, is_fixed=True
    # Original: pos=(0.5, 0, 0.5), no rotation, scale=1.0
    can = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(CAN_MESH),
            pos=(0.5, 0.0, 0.5),
            scale=1.0,
            fixed=True,
            convexify=False,
            decimate=False,
        ),
        material=gs.materials.Rigid(
            rho=20.0,
            coup_type="ipc_only",
            coup_friction=0.02,
            enable_coup_collision=True,
        ),
    )

    scene.build()

    for step in range(args.steps):
        scene.step()
        if step % 50 == 0:
            print(f"  Step {step}")

    print("Done.")


if __name__ == "__main__":
    main()
