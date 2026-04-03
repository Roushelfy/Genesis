"""
Franka robot teleop for trashbag drawstring — grab the rope tails to tighten the bag.

Scene:
  - Cloth trashbag with drawstring channel (GLB)
  - Rope drawstring (closed loop, OBJ)
  - Franka robot with keyboard teleop
  - Ground plane

Keyboard controls: see robot_teleop.py for the full keybind list.
"""

import argparse
import os

import genesis as gs
from robot_teleop import RobotTeleop

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_ASSETS = os.path.join(REPO_ROOT, "DemoAssets", "trashbag")
BAG_MESH = os.path.join(DEMO_ASSETS, "Trashbag_rope.glb")
ROPE1_MESH = os.path.join(DEMO_ASSETS, "rope1.obj")
ROPE2_MESH = os.path.join(DEMO_ASSETS, "rope2.obj")
CAN_MESH = os.path.join(REPO_ROOT, "DemoAssets", "trashbag", "trashcan_128.obj")

BAG_POS = (0.5, 0.0, 0.65)


def main():
    gs.init(backend=gs.gpu, logging_level="info")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coup_type",
        type=str,
        default="external_articulation",
        choices=["two_way_soft_constraint", "external_articulation"],
    )
    parser.add_argument("--use-al", action="store_true", help="Use AL-IPC contact constitution")
    parser.add_argument("--verbose-ipc", action="store_true", help="Print full libuipc log")
    args = parser.parse_args()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.8),
        ),
        fem_options=gs.options.FEMOptions(
            use_rigid_compatible_transform=True,
        ),
        # rigid_options=gs.options.RigidOptions(
        #     enable_self_collision=False,
        # ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.001,
            contact_friction_enable=True,
            contact_resistance=1e8,
            linear_system_tolerance=1e-5,
            enable_fem_fem_friction=False,
            verbose_ipc_log=args.verbose_ipc,
            **(dict(contact_constitution="al-ipc") if args.use_al else {}),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -0.8, 0.9),
            camera_lookat=BAG_POS,
            camera_fov=45,
        ),
        show_viewer=True,
    )

    # Ground plane
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_type="ipc_only",
            coup_friction=0.3,
        ),
    )

    # Franka robot
    franka_material_kwargs = dict(
        coup_type=args.coup_type,
        coup_friction=1.0,
        gravity_compensation=1.0,
        rho=20.0,
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

    # Fixed rigid trash can — bag sits on top with body hanging inside
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=CAN_MESH,
            pos=(BAG_POS[0], BAG_POS[1], 0.3),
            scale=1.0,
            fixed=True,
            convexify=False,
            decimate=False,
        ),
        material=gs.materials.Rigid(
            rho=20.0,
            coup_type="ipc_only",
            coup_friction=0.3,
            enable_coup_collision=True,
        ),
    )

    # Cloth trashbag with drawstring channel
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=BAG_MESH,
            pos=BAG_POS,
            scale=1.0,
        ),
        material=gs.materials.FEM.Cloth(
            E=10e3,
            nu=0.4,
            rho=20.0,
            thickness=0.0005,
            bending_stiffness=20.0,
            friction_mu=0.5,
        ),
    )

    # Drawstring rope (closed loop through channel)
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=ROPE1_MESH,
            pos=BAG_POS,
            euler=(90, 0, 0),
            scale=1.0,
        ),
        material=gs.materials.FEM.Rope(
            E=5e5,
            rho=10.0,
            thickness=0.002,
            bending_stiffness=1e3,
            friction_mu=0.5,
        ),
        surface=gs.surfaces.Default(color=(0.8, 0.15, 0.1, 1.0)),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=ROPE2_MESH,
            pos=BAG_POS,
            euler=(90, 0, 0),
            scale=1.0,
        ),
        material=gs.materials.FEM.Rope(
            E=5e5,
            rho=10.0,
            thickness=0.002,
            bending_stiffness=1e3,
            friction_mu=0.5,
        ),
        surface=gs.surfaces.Default(color=(0.8, 0.15, 0.1, 1.0)),
    )

    scene.build()

    # Start EE above and to the side of the bag — far enough to avoid intersection
    teleop = RobotTeleop.franka(
        scene=scene,
        robot=franka,
        init_pos=(0.0, 0.4, 0.5),
        init_euler=(0.0, 180.0, 0.0),
    )
    teleop.setup()
    teleop.run()


if __name__ == "__main__":
    main()
