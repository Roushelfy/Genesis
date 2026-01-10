"""
Simple test for external_articulation coupling strategy.

This example demonstrates joint-level coupling between Genesis and IPC using
ExternalArticulationConstraint. The robot's joints are coupled at the DOF level,
allowing IPC to modify joint positions based on contact forces.

Usage:
    python test_external_articulation.py
"""

import genesis as gs
import logging
import numpy as np


def main():
    # Initialize Genesis with GPU backend
    gs.init(backend=gs.gpu, logging_level=logging.INFO)

    dt = 1e-3

    # Configure IPC coupler with external_articulation strategy
    coupler_options = gs.options.IPCCouplerOptions(
        dt=dt,
        gravity=(0.0, 0.0, -9.8),
        coupling_strategy="external_articulation",  # Use joint-level coupling
        contact_friction_mu=0.5,
        IPC_self_contact=False,
        enable_ipc_gui=True,  # Show IPC visualization
    )

    # Disable Genesis rigid collision (IPC handles all contact)
    rigid_options = gs.options.RigidOptions(
        enable_collision=False,
    )

    # Create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        rigid_options=rigid_options,
        coupler_options=coupler_options,
        show_viewer=True,
    )

    # Add ground plane
    # scene.add_entity(gs.morphs.Plane())

    # Add a simple robot (Franka arm)
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda_non_overlap.xml"),
    )

    # Add a soft FEM cube to interact with
    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.6, 0.0, 0.1), size=(0.08, 0.08, 0.08)),
        material=gs.materials.FEM.Elastic(
            E=5.0e4,
            nu=0.45,
            rho=1000.0,
            model="stable_neohookean"
        ),
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )

    # Build the scene
    print("Building scene with external_articulation coupling...")
    scene.build()
    print("Scene built successfully!")

    # Set initial robot configuration
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Initial pose - arm extended toward cube
    qpos_init = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    robot.control_dofs_position(qpos_init)

    # Let the scene settle
    print("Settling...")
    for i in range(1000):
        scene.step()
        robot.control_dofs_position(qpos_init)

    print("Starting interaction test...")

    # Phase 1: Move toward cube (100 steps ~ 0.1s)
    print("Phase 1: Moving toward cube...")
    end_effector = robot.get_link("hand")
    target_pos = np.array([0.6, 0.0, 0.15])
    target_quat = np.array([0, 1, 0, 0])

    qpos_target = robot.inverse_kinematics(
        link=end_effector,
        pos=target_pos,
        quat=target_quat,
    )

    for i in range(100):
        robot.control_dofs_position(qpos_target[:-2], motors_dof)
        robot.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
        scene.step()

        if i % 20 == 0:
            print(f"  Step {i}/100")

    # Phase 2: Close gripper and push cube (200 steps ~ 0.2s)
    print("Phase 2: Closing gripper and pushing...")
    for i in range(200):
        # Gradually close gripper
        finger_pos = max(0.0, 0.04 - i * 0.04 / 100)

        robot.control_dofs_position(qpos_target[:-2], motors_dof)
        robot.control_dofs_position(np.array([finger_pos, finger_pos]), fingers_dof)
        scene.step()

        if i % 40 == 0:
            print(f"  Step {i}/200, finger_pos: {finger_pos:.4f}")

    # Phase 3: Lift cube (200 steps ~ 0.2s)
    print("Phase 3: Lifting...")
    target_pos_lift = np.array([0.6, 0.0, 0.3])
    qpos_lift = robot.inverse_kinematics(
        link=end_effector,
        pos=target_pos_lift,
        quat=target_quat,
    )

    for i in range(200):
        robot.control_dofs_position(qpos_lift[:-2], motors_dof)
        robot.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
        scene.step()

        if i % 40 == 0:
            print(f"  Step {i}/200")

    print("Test completed!")
    print("\nExpected behavior:")
    print("  - Robot arm should move smoothly toward the cube")
    print("  - Gripper should close and contact the cube")
    print("  - IPC should handle contact forces at joint level")
    print("  - Cube should be pushed/grasped and lifted")
    print("\nKey features tested:")
    print("  - ExternalArticulationConstraint creation")
    print("  - Joint-level coupling (delta_theta_tilde, ref_dof_prev)")
    print("  - Contact response through joint modifications")


if __name__ == "__main__":
    main()
