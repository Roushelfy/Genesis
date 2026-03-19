"""
Franka robot teleop with cloth — IPC demo.

Keyboard controls: see franka_teleop.py for the full keybind list.
"""

import argparse
import os

import genesis as gs
from huggingface_hub import snapshot_download
from robot_teleop import RobotTeleop


def main():
    gs.init(backend=gs.cpu, logging_level="info")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coup_type",
        type=str,
        default="external_articulation",
        choices=["two_way_soft_constraint", "external_articulation"],
    )
    parser.add_argument("--use-al", action="store_true", help="Use AL-IPC contact constitution")
    args = parser.parse_args()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            n_linesearch_iterations=8,
            linesearch_report_energy=False,
            newton_tolerance=1e-1,
            newton_translation_tolerance=1,
            newton_semi_implicit_enable=False,  # Must be false to avoid time stealing artifact
            linear_system_tolerance=1e-3,
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=0.001,
            contact_resistance=1e7,
            **(dict(contact_constitution="al-ipc") if args.use_al else {}),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.0, 1.5),
            camera_lookat=(0.5, 0.0, 0.2),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    # Ground plane
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_type="ipc_only",
        ),
    )

    # Franka robot
    franka_material_kwargs = dict(
        coup_type=args.coup_type,
    )
    if args.coup_type == "two_way_soft_constraint":
        franka_material_kwargs["coup_links"] = ("left_finger", "right_finger")
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
            pos=(0.0, 0.0, 0.005),
            convexify=True,
        ),
        material=gs.materials.Rigid(**franka_material_kwargs),
    )

    # Cloth layers
    cloth_asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="8aa8fcd60500b9f3a36c356080224bdb1be9ee59",
        allow_patterns="/IPC/grid20x20.obj",
        max_workers=1,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{cloth_asset_path}/IPC/grid20x20.obj",
            scale=0.5,
            pos=(0.5, 0.0, 0.1),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=6e4,
            nu=0.49,
            rho=200,
            thickness=0.001,
            bending_stiffness=10.0,
            friction_mu=0.5,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.1, 0.8, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{cloth_asset_path}/IPC/grid20x20.obj",
            scale=0.3,
            pos=(0.5, 0.0, 0.14),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=6e4,
            nu=0.49,
            rho=200,
            thickness=0.001,
            bending_stiffness=40.0,
            friction_mu=0.5,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    # Rigid cubes (4x4 grid) under the cloth
    cube_size = 0.05
    cube_height = 0.02501
    grid_spacing = 0.15
    for i in range(4):
        for j in range(4):
            x = (i + 1.7) * grid_spacing
            y = (j - 1.5) * grid_spacing
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(x, y, cube_height),
                    size=(cube_size, cube_size, cube_size),
                    fixed=True,
                ),
                material=gs.materials.Rigid(
                    rho=500,
                    coup_friction=0.5,
                    coup_type="ipc_only",
                ),
                surface=gs.surfaces.Plastic(
                    color=(0.8, 0.3, 0.2, 0.8),
                ),
            )

    scene.build()

    teleop = RobotTeleop.franka(
        scene=scene,
        robot=franka,
        init_pos=(0.5, 0.0, 0.6),
        init_euler=(0.0, 180.0, 0.0),
    )
    teleop.setup()
    teleop.run()


if __name__ == "__main__":
    main()
