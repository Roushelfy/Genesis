"""Rigid rotation, covariance, moment and energy-gradient controls for P2 StVK."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..skfem_reference import SkfemElasticReference
from .artifacts import runtime_identity, write_json
from .contact_fem_exact_columns import ColumnField
from .p2_stvk import P2StVK, cauchy_from_gradient


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    vertices = np.array([[0, 0, 0], [1.1, 0, 0], [0.1, 0.9, 0], [-0.1, 0.2, 1.2]]) * 0.01
    reference = SkfemElasticReference(vertices, np.array([[0, 1, 2, 3]]), 1e10, 0.3, 2000)
    model, rest = P2StVK(reference), reference.coordinates
    rotation = Rotation.from_rotvec([0.2, -0.1, 0.15]).as_matrix()
    translation = np.array([0.03, -0.02, 0.01])
    rigid = (rest @ rotation.T + translation - rest).reshape(-1)
    rigid_force, rigid_energy = model.force_energy(rigid)
    rigid_stress = model.corner_stress(rigid)
    assert np.linalg.norm(rigid_force) < 1e-8 and np.max(np.abs(rigid_stress)) < 1e-3
    deformation = np.array([[1.001, 0.0002, 0.0], [0.0, 0.9997, -0.0001], [0.0, 0.0, 1.0004]])
    current = rest @ deformation.T
    displacement = (current - rest).reshape(-1)
    force, energy = model.force_energy(displacement)
    tensor = model.corner_stress(displacement)
    rotated = (current @ rotation.T + translation - rest).reshape(-1)
    rotated_force, rotated_energy = model.force_energy(rotated)
    rotated_tensor = model.corner_stress(rotated)
    force_error = np.linalg.norm(rotated_force.reshape(-1, 3) - force.reshape(-1, 3) @ rotation.T) / np.linalg.norm(
        force
    )
    stress_error = np.linalg.norm(rotated_tensor - rotation @ tensor @ rotation.T) / np.linalg.norm(tensor)
    energy_error = abs(rotated_energy / energy - 1)
    moment = np.linalg.norm(np.cross(current, force.reshape(-1, 3)).sum(axis=0))
    assert max(force_error, stress_error, energy_error) < 1e-9
    assert moment / (np.linalg.norm(force) * 0.01) < 1e-12
    generator = np.random.default_rng(71)
    direction = generator.normal(size=len(displacement))
    direction /= np.linalg.norm(direction)
    epsilon = 1e-9
    plus = model.force_energy(displacement + epsilon * direction)[1]
    minus = model.force_energy(displacement - epsilon * direction)[1]
    gradient_error = abs((plus - minus) / (2 * epsilon) - force @ direction) / np.linalg.norm(force)
    assert gradient_error < 1e-6
    small = displacement * 1e-5
    small_force, _ = model.force_energy(small)
    linear = reference.stiffness @ small
    linear_error = np.linalg.norm(small_force - linear) / np.linalg.norm(linear)
    assert linear_error < 1e-7
    correction = model.force_correction(displacement)
    correction_error = np.linalg.norm(correction - (reference.stiffness @ displacement - force)) / np.linalg.norm(force)
    assert correction_error < 1e-11
    shifted_displacement = displacement + np.tile(translation, len(rest))
    shifted_correction = model.force_correction(shifted_displacement)
    correction_translation_error = np.linalg.norm(shifted_correction - correction) / np.linalg.norm(correction)
    assert correction_translation_error < 1e-9
    cached = P2StVK(reference, cache_evaluations=True)
    mutable = displacement.copy()
    for state in (displacement, rotated, displacement, small):
        # Mutate the same caller-owned array to detect identity-based stale reuse.
        mutable[:] = state
        expected_force, expected_energy = model.force_energy(state)
        expected_correction = model.force_correction(state)
        for _ in range(2):
            actual_correction = cached.force_correction(mutable)
            np.testing.assert_array_equal(actual_correction, expected_correction)
            actual_force, actual_energy = cached.force_energy(mutable)
            np.testing.assert_array_equal(actual_force, expected_force)
            assert actual_energy == expected_energy
            # Returned arrays must not let a caller overwrite the internal cache.
            actual_correction[:] = 0
    np.testing.assert_allclose(
        cauchy_from_gradient(np.broadcast_to(deformation - np.eye(3), tensor.shape), 1e10, 0.3),
        tensor,
        rtol=1e-9,
        atol=1e-5,
    )
    tets = np.array([[0, 1, 2, 5], [0, 1, 4, 5], [0, 3, 4, 5]])
    gradient = np.zeros((3, 4, 3, 3))
    gradient[:, :, 0, 2] = 0.01 * (tets // 3)
    column = ColumnField(
        np.zeros_like(gradient),
        np.array([0]),
        np.array([[0.2, 0.3, 0.5]]),
        1,
        1,
        tets,
        3,
        np.array([[0, 1, 2]]),
        stvk_gradient=gradient,
        material_rotation=np.eye(3),
    )
    lame, shear = reference.lame
    column_error = 0.0
    for depth in (0.0, 0.2, 0.6, 0.9, 1.0):
        gamma = 0.01 * depth
        expected = np.diag(
            [
                (lame / 2 + 2 * shear) * gamma**2 + (lame / 2 + shear) * gamma**4,
                lame / 2 * gamma**2,
            ]
        )
        observed = column.sample(np.array([depth]), np.eye(3)[None, :, :2])[0]
        column_error = max(column_error, float(np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1)))
    assert column_error < 1e-12
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": True,
            "rigid_force_norm_n": float(np.linalg.norm(rigid_force)),
            "rigid_energy_j": rigid_energy,
            "rigid_maximum_stress_pa": float(np.max(np.abs(rigid_stress))),
            "force_covariance_error": float(force_error),
            "stress_covariance_error": float(stress_error),
            "energy_covariance_error": float(energy_error),
            "physical_moment_nm": float(moment),
            "energy_gradient_relative_error": float(gradient_error),
            "small_strain_linear_force_relative_error": float(linear_error),
            "direct_correction_full_force_relative_error": float(correction_error),
            "direct_correction_translation_relative_error": float(correction_translation_error),
            "exact_cache_state_mutation_and_return_alias_controls_passed": True,
            "nonlinear_column_sampling_relative_error": column_error,
        },
    )


if __name__ == "__main__":
    main()
