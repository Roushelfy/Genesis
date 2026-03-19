"""
Franka robot teleop for paper folding — IPC demo.

A paper sheet sits on a table. Use keyboard teleop to fold it.
The paper uses PlasticDiscreteShellBending so folds persist as permanent creases.

Keyboard controls: see robot_teleop.py for the full keybind list.
"""

import argparse
import os

import genesis as gs
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
    args = parser.parse_args()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            n_linesearch_iterations=8,
            newton_tolerance=1e-1,
            newton_translation_tolerance=1,
            newton_semi_implicit_enable=False,
            linear_system_tolerance=1e-3,
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=0.001,
            contact_resistance=1e7,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -0.8, 1.2),
            camera_lookat=(0.5, 0.0, 0.1),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    # Ground plane
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(coup_type="ipc_only"),
    )

    # Franka robot
    franka_material_kwargs = dict(
        coup_type=args.coup_type,
        coup_friction=0.5,
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

    # Table — fixed rigid box at the far end of the robot's reach.
    # The paper overhangs the near edge (-x, toward robot) so the gripper
    # can grab the hanging edge without the arm passing over the table.
    table_height = 0.15
    table_size_x = 0.20
    table_size_y = 0.30
    table_x = 0.65
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(table_x, 0.0, table_height / 2),
            size=(table_size_x, table_size_y, table_height),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            rho=1000,
            coup_friction=0.5,
            coup_type="ipc_only",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.55, 0.35, 0.2, 1.0),
        ),
    )

    # Paper sheet on the table.
    # Offset toward -x so the near edge hangs off the table toward the robot.
    # The robot grabs the hanging edge from below and folds it up and over.
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paper_mesh = os.path.join(
        os.path.dirname(_repo_root),
        "libuipc-samples",
        "assets",
        "sim_data",
        "trimesh",
        "grid20x20.obj",
    )
    paper_size = 0.25
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=paper_mesh,
            scale=paper_size,
            pos=(table_x - 0.06, 0.0, table_height + 0.002),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Paper(
            E=5e5,
            rho=700.0,
            thickness=0.001,
            bending_stiffness=1e4,
            yield_threshold=0.05,
            hardening_modulus=0.2,
            friction_mu=0.5,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.95, 0.92, 0.85, 1.0),
        ),
    )

    scene.build()

    # Start EE above the near (hanging) edge of the paper, pointing down
    paper_near_edge_x = table_x - 0.06 - paper_size / 2
    teleop = RobotTeleop.franka(
        scene=scene,
        robot=franka,
        init_pos=(paper_near_edge_x, 0.0, 0.5),
        init_euler=(0.0, 180.0, 0.0),
    )
    teleop.setup()
    teleop.run()


if __name__ == "__main__":
    main()
