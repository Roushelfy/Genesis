"""
Minimal test for external_articulation coupling strategy.

This is the simplest possible test - a single robot with minimal motion,
to verify that ExternalArticulationConstraint is working correctly.

Usage:
    python test_external_articulation_simple.py
"""

import genesis as gs
import logging
import numpy as np


def main():
    # Initialize Genesis
    gs.init(backend=gs.gpu, logging_level=logging.DEBUG)

    dt = 2e-3  # Larger timestep for easier debugging

    # Configure IPC with external_articulation
    coupler_options = gs.options.IPCCouplerOptions(
        dt=dt,
        gravity=(0.0, 0.0, -9.8),
        coupling_strategy="external_articulation",
        contact_friction_mu=0.5,
        IPC_self_contact=False,
        enable_ipc_gui=True,
    )

    rigid_options = gs.options.RigidOptions(
        enable_collision=False,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        rigid_options=rigid_options,
        coupler_options=coupler_options,
        show_viewer=True,
    )

    # Add ground
    # scene.add_entity(gs.morphs.Plane())

    # Add Franka robot
    print("Adding robot...")
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda_non_overlap.xml"),
    )

    # Build scene
    print("Building scene...")
    scene.build()
    print("✓ Scene built successfully!")
    print(f"✓ Robot has {robot.n_dofs} DOFs")
    print(f"✓ Robot has {robot.n_links} links")

    # Set a simple initial pose
    qpos = np.zeros(robot.n_dofs)
    qpos[0] = 0.0  # Joint 1
    qpos[1] = -0.3  # Joint 2
    qpos[2] = 0.0  # Joint 3
    qpos[3] = -1.5  # Joint 4
    qpos[4] = 0.0  # Joint 5
    qpos[5] = 1.2  # Joint 6
    qpos[6] = 0.0  # Joint 7
    qpos[7:] = 0.02  # Fingers

    robot.set_qpos(qpos)
    print(f"✓ Initial qpos set: {qpos[:7]}")

    # Settle for a few steps
    print("\nSettling (50 steps)...")
    for i in range(10):
        scene.step()
        if i % 10 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"  Step {i}: qpos[0:3] = {current_qpos[:3]}")

    print("\n✓ Settling complete!")

    # Simple test: move joint 1 slowly
    print("\nTest: Moving joint 1 from 0.0 to 0.5 rad...")
    motors_dof = np.arange(7)

    for step in range(100):
        # Linearly interpolate joint 1 position
        target_qpos = qpos.copy()
        target_qpos[0] = 0.5 * (step / 100.0)

        robot.control_dofs_position(target_qpos[:7], motors_dof)
        scene.step()

        if step % 20 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"  Step {step}: target={target_qpos[0]:.3f}, actual={current_qpos[0]:.3f}")

    print("\n✓ Test completed!")

    final_qpos = robot.get_qpos().cpu().numpy()
    print(f"\nFinal state:")
    print(f"  Joint 1: {final_qpos[0]:.4f} rad (target: 0.5)")
    print(f"  Deviation: {abs(final_qpos[0] - 0.5):.4f} rad")

    if abs(final_qpos[0] - 0.5) < 0.05:
        print("\n✅ SUCCESS: Joint reached target position!")
    else:
        print("\n⚠️  WARNING: Large deviation from target")

    print("\nKey checks:")
    print("  1. ExternalArticulationConstraint created? Check logs")
    print("  2. ref_dof_prev initialized? Check logs")
    print("  3. delta_theta_tilde computed correctly?")
    print("  4. Joint motion smooth and stable?")


if __name__ == "__main__":
    main()
