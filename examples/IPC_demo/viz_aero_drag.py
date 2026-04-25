"""Visualize aerodynamic drag: three trashbags dropped onto a platform.

All bags fall and settle on a platform. Then the platform teleports away,
and the bags resume falling — compare drag behaviors side by side.

Left   bag (red):   no aerodynamic drag
Middle bag (green): drag + curvature drag scale
Right  bag (blue):  drag + curvature drag scale + inflate

Uses Trashbag_rope.glb which has a bag mesh + drawstring channel at the
boundary (two submeshes merged into a single FEM cloth entity).

Usage:
    python examples/IPC_demo/viz_aero_drag.py
    python examples/IPC_demo/viz_aero_drag.py --open-up
    python examples/IPC_demo/viz_aero_drag.py --drag 0.2 --curv 3.0 --infl 3.0
"""

import argparse
from pathlib import Path

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-up", action="store_true", help="Bags open-side UP (bowl). Default is open-side DOWN.")
    parser.add_argument("--drag", type=float, default=0.1, help="Base drag coefficient")
    parser.add_argument("--curv", type=float, default=1.0, help="Curvature drag scale")
    # parser.add_argument("--infl", type=float, default=1.0, help="Curvature inflate scale")
    parser.add_argument("--infl", type=float, default=0.001, help="Curvature inflate scale")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    REPO_ROOT = Path(__file__).resolve().parents[2]
    BAG_GLB = str(REPO_ROOT / "DemoAssets" / "trashbag" / "Trashbag_rope.glb")

    DROP_HEIGHT = 1.5
    PLATFORM_HEIGHT = 0.6
    SPACING = 0.6
    SETTLE_STEPS = 500

    bag_euler = (0, 0, 0) if args.open_up else (160, 0, 0)
    orientation = "open-up" if args.open_up else "default"
    print(f"Orientation: {orientation}, euler={bag_euler}")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 60),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_enable=True,
            contact_friction_enable=True,
            contact_d_hat=0.001,
            newton_tolerance=1e-2,
            enable_rigid_rigid_contact=False,
            ignore_end_effector_check=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.5, 1.2),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=True,
    )

    # Ground plane
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(coup_type="ipc_only"),
        name="ground",
    )

    # Platform that catches the bags, then teleports away
    platform = scene.add_entity(
        gs.morphs.Box(
            size=(1.5, 1.5, 0.05),
            pos=(0.0, 0.0, PLATFORM_HEIGHT),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_type="ipc_only",
            coup_friction=0.3,
        ),
        surface=gs.surfaces.Plastic(color=(0.6, 0.6, 0.6, 1.0)),
        name="platform",
    )

    # Shared cloth parameters
    cloth_params = dict(
        E=20e3,
        nu=0.4,
        rho=50,
        thickness=0.001,
        bending_stiffness=40.0,
        friction_mu=0.02,
    )

    # Left bag (red): no aerodynamic drag
    # print("  [0] bag_no_drag: no aero damping")
    # scene.add_entity(
    #     morph=gs.morphs.Mesh(
    #         file=BAG_GLB,
    #         scale=1.0,
    #         pos=(-SPACING, 0.0, DROP_HEIGHT),
    #         euler=bag_euler,
    #     ),
    #     material=gs.materials.FEM.Cloth(
    #         **cloth_params,
    #         aerodynamic_drag=args.drag,
    #     ),
    #     surface=gs.surfaces.Plastic(color=(0.9, 0.3, 0.3, 1.0)),
    #     name="bag_no_drag",
    # )

    # # Middle bag (green): drag + curvature drag scale
    # print(f"  [1] bag_drag_curv: drag={args.drag}, curv={args.curv}")
    # scene.add_entity(
    #     morph=gs.morphs.Mesh(
    #         file=BAG_GLB,
    #         scale=1.0,
    #         pos=(0.0, 0.0, DROP_HEIGHT),
    #         euler=bag_euler,
    #     ),
    #     material=gs.materials.FEM.Cloth(
    #         **cloth_params,
    #         aerodynamic_drag=args.drag,
    #         curvature_drag_scale=args.curv,
    #     ),
    #     surface=gs.surfaces.Plastic(color=(0.3, 0.9, 0.3, 1.0)),
    #     name="bag_drag_curv",
    # )

    # Right bag (blue): drag + curvature drag scale + inflate
    print(f"  [2] bag_drag_curv_infl: drag={args.drag}, curv={args.curv}, infl={args.infl}")
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=BAG_GLB,
            scale=1.0,
            pos=(SPACING, 0.0, DROP_HEIGHT),
            euler=bag_euler,
        ),
        material=gs.materials.FEM.Cloth(
            **cloth_params,
            aerodynamic_drag=args.drag,
            curvature_drag_scale=args.curv,
            curvature_inflate_scale=args.infl,
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.3, 0.9, 1.0)),
        name="bag_drag_curv_infl",
    )

    scene.build()

    step = 0
    while True:
        scene.step()
        step += 1

        if step == SETTLE_STEPS:
            # Teleport the platform far below so bags resume free-falling
            platform.set_pos((0.0, 0.0, -10.0))
            print(f"Step {step}: Platform teleported away!")


if __name__ == "__main__":
    main()
