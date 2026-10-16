"""Recover COM motion and angular momentum by independent consistent-mass projection."""

import numpy as np


def motion_moments(body, permutation, positions, velocity):
    mass = body.mass[permutation][:, permutation]
    translations = np.tile(np.eye(3), (len(positions), 1))
    translation_mass = translations.T @ (mass @ translations)
    np.testing.assert_allclose(translation_mass, body.total_mass * np.eye(3), rtol=1e-12, atol=1e-15)
    centre = np.linalg.solve(translation_mass, translations.T @ (mass @ positions.reshape(-1)))
    relative = positions - centre
    rotations = np.column_stack([np.cross(axis, relative).reshape(-1) for axis in np.eye(3)])
    modes = np.column_stack((translations, rotations))
    generalized_momentum = modes.T @ (mass @ velocity)
    coefficients = np.linalg.solve(modes.T @ (mass @ modes), generalized_momentum)
    elastic_velocity = velocity - modes @ coefficients
    return {
        "centre_m": centre,
        "linear_velocity_world_m_s": coefficients[:3],
        "angular_velocity_world_rad_s": coefficients[3:],
        "angular_momentum_about_com_world_Nms": generalized_momentum[3:],
        "kinetic_energy_J": float(0.5 * velocity @ (mass @ velocity)),
        "internal_velocity_rms_m_s": float(np.sqrt(elastic_velocity @ (mass @ elastic_velocity) / body.total_mass)),
    }


def load_initial_motion(run, config, frame, static, full):
    body, permutation = static.body, static.parts[0].permutation
    n = body.basis.N
    path = run / "results/initial_velocity.npz"
    enabled = bool(config.get("captured_initial_velocity", False))
    if path.exists():
        with np.load(path) as saved:
            body_velocity, joint_velocity = saved["body_velocity"].copy(), saved["joint_velocity"].copy()
    else:
        assert not enabled
        body_velocity, joint_velocity = np.zeros(n), np.zeros(9)
    assert body_velocity.shape == (n,) and joint_velocity.shape == (9,)
    positions = static.rest[: n // 3] + full[:n].reshape(-1, 3)
    motion = motion_moments(body, permutation, positions, body_velocity)
    if enabled:
        np.testing.assert_array_equal(joint_velocity, frame["robot_joint_velocity"])
        angular = np.array(frame["egg_angular_velocity_rad_s"])
        # get_vel(relative=False) is measured at the internal link origin.
        # Verify the implied velocity there, instead of reproducing the
        # simulation's origin-to-COM conversion in this independent audit.
        offset = np.array(frame["egg_position_m"]) - np.array(frame["egg_world_com_m"])
        recovered_origin_velocity = motion["linear_velocity_world_m_s"] + np.cross(angular, offset)
        np.testing.assert_allclose(recovered_origin_velocity, frame["egg_velocity_m_s"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(motion["angular_velocity_world_rad_s"], angular, rtol=0, atol=1e-12)
        assert motion["internal_velocity_rms_m_s"] < 1e-12
    else:
        np.testing.assert_array_equal(body_velocity, np.zeros(n))
        np.testing.assert_array_equal(joint_velocity, np.zeros(9))
    return body_velocity, joint_velocity, motion
