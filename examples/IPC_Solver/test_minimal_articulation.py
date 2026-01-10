"""
Minimal test for external_articulation coupling - single joint.
Uses the simplest robot to debug the coupling mechanism.
"""

import genesis as gs
import numpy as np

def main():
    # Initialize Genesis
    gs.init(backend=gs.gpu,logging_level="info",performance_mode=True)

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

    # Create simple two-cube robot with one joint
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/simple/two_cube_joint.urdf", pos=(0.0, 0.0, 0.5), fixed=True)
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
    for i in range(10):
        robot.control_dofs_position(target_zero)
        scene.step()

        if i % 10 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"Step {i:3d}: qpos = {current_qpos}")

    # Phase 2: Move joint
    print("\n=== Phase 2: Moving joint (200 steps) ===")
    # Create target with correct size
    target_move = np.zeros(robot.n_dofs)
    # Set first joint to non-zero value
    if robot.n_dofs >= 1:
        target_move[0] = 0.5  # First joint (should be only joint)

    for i in range(200):
        robot.control_dofs_position(target_move)
        scene.step()

        if i % 40 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            error = np.linalg.norm(current_qpos - target_move)
            print(f"Step {i:3d}: qpos = {current_qpos}")
            print(f"         error = {error:.4f}")

    # Final result
    final_qpos = robot.get_qpos().cpu().numpy()
    print(f"\n=== Final Results ===")
    print(f"Target qpos:  {target_move}")
    print(f"Final qpos:   {final_qpos}")
    print(f"Final error:  {np.linalg.norm(final_qpos - target_move):.6f}")

    # Keep viewer open
    print("\nTest complete! Close viewer to exit.")
    while scene.viewer.is_alive():
        robot.control_dofs_position(target_move)
        scene.step()

if __name__ == "__main__":
    main()
