"""
Test for external_articulation coupling with three cube chain (two joints).
Tests joint positioning and oscillating motion.
"""

import genesis as gs
import numpy as np

def main():
    # Initialize Genesis
    gs.init(backend=gs.gpu, logging_level="info", performance_mode=True)

    # Create scene
    dt = 0.01

    # Configure IPC coupler with external_articulation strategy
    coupler_options = gs.options.IPCCouplerOptions(
        dt=dt,
        gravity=(0.0, 0.0, -9.8),
        coupling_strategy="external_articulation",
        contact_friction_mu=0.5,
        IPC_self_contact=False,
        enable_ipc_gui=True,
    )

    # Disable rigid collision when using IPC
    rigid_options = gs.options.RigidOptions(
        enable_collision=False,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=10,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=rigid_options,
        coupler_options=coupler_options,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        ),
        show_viewer=True,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Create three-cube chain robot with two joints
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/simple/three_cube_chain.urdf", pos=(0.0, 0.0, 0.5), fixed=True)
    )

    # Build scene
    print("Building scene...")
    scene.build()
    print("Scene built!")

    # Get initial qpos
    qpos_init = robot.get_qpos().cpu().numpy()
    print(f"\nInitial qpos: {qpos_init}")
    print(f"Number of DOFs: {robot.n_dofs}")
    print(f"Number of links: {robot.n_links}")
    print(f"Number of joints: {len([j for link_joints in robot._joints for j in link_joints])}")

    # Phase 1: Hold at zero position (settle)
    print("\n=== Phase 1: Settling at zero (50 steps) ===")
    target_zero = np.zeros(robot.n_dofs)
    for i in range(50):
        robot.control_dofs_position(target_zero)
        scene.step()

        if i % 10 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"Step {i:3d}: qpos = {current_qpos}")

    # Phase 2: Oscillating motion (back and forth)
    print("\n=== Phase 2: Oscillating motion ===")

    # Parameters for oscillation
    amplitude1 = 8  # radians for joint1 (Z-axis rotation)
    amplitude2 = 8  # radians for joint2 (Y-axis rotation)
    period = 500      # steps per cycle
    total_steps = 500

    for i in range(total_steps):
        # Calculate sinusoidal targets for both joints
        phase = 2.0 * np.pi * i / period

        target = np.zeros(robot.n_dofs)
        if robot.n_dofs >= 1:
            target[0] = amplitude1 * np.sin(phase)
        if robot.n_dofs >= 2:
            target[1] = amplitude2 * np.sin(phase + np.pi / 4)  # Slightly phase-shifted

        robot.control_dofs_position(target)
        scene.step()

        if i % 50 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"Step {i:3d}: target = {target}, qpos = {current_qpos}")

    # Phase 3: Return to zero
    print("\n=== Phase 3: Returning to zero (100 steps) ===")
    target_zero = np.zeros(robot.n_dofs)
    for i in range(100):
        robot.control_dofs_position(target_zero)
        scene.step()

        if i % 20 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"Step {i:3d}: qpos = {current_qpos}")

    # Final result
    final_qpos = robot.get_qpos().cpu().numpy()
    print(f"\n=== Final Results ===")
    print(f"Target qpos:  {target_zero}")
    print(f"Final qpos:   {final_qpos}")
    print(f"Final error:  {np.linalg.norm(final_qpos - target_zero):.6f}")

    # Keep viewer open with continued oscillation
    print("\nTest complete! Continuing oscillation... Close viewer to exit.")
    step = 0
    while scene.viewer.is_alive():
        phase = 2.0 * np.pi * step / period

        target = np.zeros(robot.n_dofs)
        if robot.n_dofs >= 1:
            target[0] = amplitude1 * np.sin(phase)
        if robot.n_dofs >= 2:
            target[1] = amplitude2 * np.sin(phase + np.pi / 4)

        robot.control_dofs_position(target)
        scene.step()
        step += 1

if __name__ == "__main__":
    main()
