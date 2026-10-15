"""Initialize a declared velocity control from the same actual rigid capture frame."""

import numpy as np


def initial_motion(static, full_displacement, joint_position, frame, enabled):
    """Use measured COM twist with zero internal elastic velocity at the solved closure."""
    n = static.body.basis.N
    joint_velocity, body_velocity = np.zeros(9), np.zeros(n)
    metadata = {"mode": "zero_initial_velocity"}
    if not enabled:
        return body_velocity, joint_velocity, metadata
    body = static.body
    mass = body.mass[body.component_dofs[:, 0]][:, body.component_dofs[:, 0]]
    weights = np.asarray(mass.sum(axis=1)).ravel()
    positions = static.rest[: n // 3] + full_displacement[:n].reshape(-1, 3)
    centre = weights @ positions / body.total_mass
    linear = np.array(frame["egg_velocity_m_s"])
    angular = np.array(frame["egg_angular_velocity_rad_s"])
    joint_velocity = np.array(frame["robot_joint_velocity"])
    body_velocity = (linear + np.cross(angular, positions - centre)).reshape(-1)
    metadata = {
        "mode": "captured_COM_twist_and_joint_velocity_at_solved_closure",
        "source_capture_mode": frame["mode"],
        "source_tick": frame["tick"],
        "source_substep": frame["substep"],
        "source_linear_velocity_world_m_s": linear.tolist(),
        "source_angular_velocity_world_rad_s": angular.tolist(),
        "source_joint_velocity": joint_velocity.tolist(),
        "solved_closure_minus_captured_joint_position": np.asarray(
            joint_position - np.array(frame["robot_joint_position"]), dtype=float
        ).tolist(),
        "initial_body_com_world_m": (centre + static.world_coordinate_origin_m).tolist(),
        "limitation": "Only initial velocities match the captured frame. The independently solved closure geometry/elastic state replaces the penetrated rigid snapshot; original approach/lift/support history and integrator/contact-law equivalence remain unverified.",
    }
    return body_velocity, joint_velocity, metadata
