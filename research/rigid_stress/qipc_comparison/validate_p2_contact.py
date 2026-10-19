"""Check quadratic elasticity against exact stress and bending before contact reference use."""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTet

from .artifacts import runtime_identity, write_json
from .p2_elastic import FreeElasticP2, quadratic_triangle_minimum, quadratic_triangle_weights
from .contact_fem_exact_columns import ColumnField, ShellColumnField, surface_quadrature
from .prepare_inputs import make_mesh
from .finite_area_reference import finite_area_load
from .contact_qp import solve_contact_qp
from .pad_equilibrium import normal_pad_equilibrium, shell_pad_boundary_gaps


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--cholespy-path", type=Path)
    cli.add_argument("--contact-solver-path", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    qp_controls = []
    if args.contact_solver_path is not None:
        points = np.tile(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), (2, 1))
        sign = np.repeat([1.0, -1.0], 4)
        rigid = np.column_stack((sign, sign * points[:, 1], -sign * points[:, 0], sign < 0))
        for form in ("dual", "whitened-primal"):
            for duplicate in (False, True):
                interpolation = np.vstack((np.eye(8), np.eye(8)[[0, 3, 4]])) if duplicate else np.eye(8)
                solved = solve_contact_qp(
                    np.eye(8),
                    np.zeros(8),
                    rigid,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    np.empty((0, 6), dtype=int),
                    args.contact_solver_path,
                    interpolation=interpolation,
                    form=form,
                )
                assert solved.candidate_passed, solved.status
                # Eight unit-compliance springs share each plate's unit force;
                # duplicates split multipliers while preserving physical nodal force.
                np.testing.assert_allclose(solved.nodal_force, 0.25, rtol=0, atol=1e-10)
                np.testing.assert_allclose(solved.coordinates, [-0.25, 0, 0, -0.5], rtol=0, atol=1e-10)
                qp_controls.append({"form": form, "duplicate_constraints": duplicate, "passed": True})
            for pad_compliance in (0.01, 3.0, 100.0):
                solved = solve_contact_qp(
                    np.eye(8),
                    np.zeros(8),
                    rigid,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    np.empty((0, 6), dtype=int),
                    args.contact_solver_path,
                    form=form,
                    round_limit=1,
                    constraint_compliance=np.full(8, pad_compliance),
                )
                assert solved.candidate_passed, solved.status
                np.testing.assert_allclose(solved.nodal_force, 0.25, rtol=0, atol=1e-9)
                np.testing.assert_allclose(
                    solved.coordinates,
                    np.array([-0.25, 0, 0, -0.5]) * (1 + pad_compliance),
                    rtol=0,
                    atol=1e-8,
                )
                qp_controls.append({"form": form, "pad_compliance": pad_compliance, "passed": True})
        coordinates, force, error, _ = normal_pad_equilibrium(np.zeros(8), rigid, np.ones(8), 2.0, 1.0)
        np.testing.assert_allclose(force, 0.25, atol=1e-10)
        np.testing.assert_allclose(coordinates, [-0.125, 0, 0, -0.25], atol=1e-10)
        assert error <= 1e-10
        # Asymmetric rest gaps require tilt as well as translation. Recover
        # the same known uniform pressure and all four prescribed coordinates.
        expected = np.array([-0.2, 0.01, -0.02, -0.5])
        gap = -0.125 - rigid @ expected
        coordinates, force, error, _ = normal_pad_equilibrium(gap, rigid, np.ones(8), 2.0, 1.0)
        np.testing.assert_allclose(force, 0.25, atol=1e-10)
        np.testing.assert_allclose(coordinates, expected, atol=1e-10)
        qp_controls.append({"rigid_pad_energy_controls": True, "passed": True})
    midpoint = np.array([[0.0, 0, 0], [2, 0, 0.1], [0.2, 1, -0.1]])
    outer = midpoint + np.array([[0.1, 0.1, 0.2], [0.1, 0.0, 0.2], [0.0, 0.1, 0.1]])
    translation, rotation = np.array([0.3, -0.2, 0.4]), np.array([0.02, 0.04, -0.03])
    current = midpoint + translation + np.cross(rotation, midpoint)
    bottom, top = shell_pad_boundary_gaps(outer, midpoint, np.array([[0, 1, 2]]), current, np.zeros(4))
    actual = outer + translation + np.cross(rotation, outer)
    np.testing.assert_allclose(bottom[0], actual[:, 2] - outer[:, 2].min(), atol=1e-14)
    np.testing.assert_allclose(top[0], outer[:, 2].max() - actual[:, 2], atol=1e-14)
    for count in (1, 3, 9, 25):
        points, area = surface_quadrature(count)
        assert points.min() >= 0 and area.min() > 0
        np.testing.assert_allclose(points.sum(axis=1), 1, atol=1e-15)
        np.testing.assert_allclose(area.sum(), 1, atol=1e-15)
        np.testing.assert_allclose(area @ points, np.full(3, 1 / 3), atol=1e-15)
        if count >= 3:
            np.testing.assert_allclose(area @ points**2, np.full(3, 1 / 6), atol=1e-15)
        if count >= 9:
            np.testing.assert_allclose(area @ points**4, np.full(3, 1 / 15), atol=1e-15)
    mesh = MeshTet().refined(1)
    vertices = mesh.p.T * np.array([0.002, 0.002, 0.0005])
    load_model = FreeElasticP2(
        vertices - [0.001, 0.001, 0.00025], mesh.t.T, 1e10, 0.3, 2000, cholespy_path=args.cholespy_path
    )
    loads = []
    for order in (8, 10):
        force, points, point_force, _, _, _ = finite_area_load(load_model, len(vertices), 0.003, order)

        def virtual_displacement(coordinates):
            x, y, z = coordinates.T
            return x**2 + 2 * x * y + 3 * z**2 + x * z

        nodal_work = virtual_displacement(load_model.vertices) @ force[:, 2]
        quadrature_work = virtual_displacement(points) @ point_force[:, 2]
        np.testing.assert_allclose(nodal_work, quadrature_work, rtol=1e-12, atol=1e-18)
        loads.append(force)
    np.testing.assert_allclose(loads[0], loads[1], rtol=1e-11, atol=1e-13)
    rows = []
    for ordered in (False, True):
        model = FreeElasticP2(
            vertices,
            mesh.t.T,
            1e10,
            0.3,
            2000.0,
            column_faces=np.empty((0, 3)) if ordered else None,
            cholespy_path=args.cholespy_path,
        )
        constant = np.array([[1e4, 2e3, -3e3], [2e3, -5e3, 1e3], [-3e3, 1e3, 7e3]])
        for bending in (False, True):
            force = (
                model.reference.pure_bending_load(0.01) if bending else model.reference.uniform_stress_load(constant)
            )
            stress, displacement, correction, residual = model.solve(force.reshape(-1, 3))
            expected = np.broadcast_to(constant, stress.shape).copy()
            if bending:
                expected[:] = 0
                expected[..., 0, 0] = 1e8 * model.reference.corner_positions[..., 2]
            error = np.linalg.norm(stress - expected) / np.linalg.norm(expected)
            equilibrium = np.linalg.norm(residual) / np.linalg.norm(force)
            balancing = np.linalg.norm(correction) / np.linalg.norm(force)
            assert max(error, equilibrium, balancing) < 1e-8, (error, equilibrium, balancing)
            assert np.isfinite(model.displacement_gradient(displacement)).all()
            rows.append(
                {
                    "ordered": ordered,
                    "bending": bending,
                    "stress_relative_error": float(error),
                    "equation_relative_error": float(equilibrium),
                    "balance_relative_correction": float(balancing),
                }
            )
        surface, triangles, quadratic = model.surface(mesh.facets[:, mesh.boundary_facets()].T)
        assert len(triangles) == 4 * len(quadratic)
        assert np.isin(quadratic, surface).all()
    samples = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]])
    np.testing.assert_allclose(
        quadratic_triangle_weights(np.column_stack((1 - samples.sum(axis=1), samples))), np.eye(6), atol=1e-15
    )
    interior = (samples[:, 0] - 0.2) ** 2 + 2 * (samples[:, 1] - 0.3) ** 2 - 0.04
    edge = (samples[:, 0] - 0.4) ** 2 + (samples[:, 1] + 0.2) ** 2
    concave = -(samples**2).sum(axis=1)
    np.testing.assert_allclose(
        quadratic_triangle_minimum(np.array([interior, edge, concave])), [-0.04, 0.04, -1], atol=1e-14
    )
    # All six nodal gaps are positive, but an interior penetration must still be rejected.
    assert interior.min() > 0 and quadratic_triangle_minimum(interior[None])[0] < 0
    # Interpolate a physical affine stress across every prism split and orientation.
    outer = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[2, 0, 1]])
    weights = np.array([[0.2, 0.3, 0.5], [0.4, 0.5, 0.1]])
    basis = np.broadcast_to(np.eye(3)[:, :2], (2, 3, 2))
    for depths in (
        np.linspace(0, 1, 2),
        np.linspace(0, 1, 3),
        np.linspace(0, 1, 5),
        np.array([0.0, 0.05, 0.15, 0.4, 1.0]),
    ):
        layers = len(depths) - 1
        coordinates, tets, _, volume = make_mesh(outer, outer + [0, 0, 1], faces, layers, depths)
        np.testing.assert_allclose(volume.sum(), 0.5, atol=1e-14)
        solutions = []
        for through_wall in (False, True):
            model = FreeElasticP2(
                coordinates * [0.002, 0.002, 0.0005],
                tets,
                1e10,
                0.3,
                2000,
                column_faces=faces,
                through_wall_ordering=through_wall,
                cholespy_path=args.cholespy_path,
            )
            force = model.reference.pure_bending_load(0.01)
            stress, displacement, correction, residual = model.solve(force.reshape(-1, 3))
            expected = np.zeros_like(stress)
            expected[..., 0, 0] = 1e8 * model.reference.corner_positions[..., 2]
            error = np.linalg.norm(stress - expected) / np.linalg.norm(expected)
            equilibrium = np.linalg.norm(residual) / np.linalg.norm(force)
            assert max(error, equilibrium, np.linalg.norm(correction) / np.linalg.norm(force)) < 1e-8
            solutions.append(displacement)
            rows.append(
                {
                    "wall_depths": depths.tolist(),
                    "through_wall_ordering": through_wall,
                    "stress_relative_error": float(error),
                    "equation_relative_error": float(equilibrium),
                }
            )
        np.testing.assert_allclose(solutions[0], solutions[1], rtol=1e-8, atol=1e-15)
        values = 2 + coordinates[tets, 0] - 3 * coordinates[tets, 1] + 5 * coordinates[tets, 2]
        tensors = values[..., None, None] * np.eye(3)
        field = ColumnField(
            tensors, np.zeros(2, dtype=int), weights, layers, 1, tets, 3, np.sort(faces, axis=1), depths
        )
        constant = 2 + weights[:, 1] - 3 * weights[:, 2]
        shell = ShellColumnField((constant[:, None] + np.array([5, 2.5, 0]))[..., None, None] * np.eye(3))
        for depth in np.linspace(0, 1, 37):
            actual = field.sample(np.full(2, depth), basis)
            expected = (2 + weights[:, 1] - 3 * weights[:, 2] + 5 * depth)[:, None, None] * np.eye(2)
            np.testing.assert_allclose(actual, expected, atol=1e-13)
            np.testing.assert_allclose(shell.sample(np.full(2, depth), basis), expected, atol=1e-13)
        membrane, bending, square = np.zeros(2), np.zeros(2), np.zeros(2)
        breaks = field.breaks()
        for lower, upper in zip(breaks[:-1], breaks[1:]):
            for offset in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
                depth = (lower + upper) / 2 + offset * (upper - lower) / 2
                value = field.sample(depth, basis)[:, 0, 0]
                width = (upper - lower) / 2
                membrane += width * value
                bending += width * value * (0.5 - depth)
                square += width * value**2
        constant = 2 + weights[:, 1] - 3 * weights[:, 2]
        np.testing.assert_allclose(membrane, constant + 2.5, atol=1e-13)
        np.testing.assert_allclose(bending, -5 / 12, atol=1e-13)
        np.testing.assert_allclose(square, constant**2 + 5 * constant + 25 / 3, atol=1e-13)
        warped, warped_tets, _, warped_volumes = make_mesh(
            outer, outer * [0.8, 0.9, 1.0] + [0.07, 0.03, 0.8], faces, layers, depths
        )
        for count in (1, 3, 9, 25):
            bary, angular = surface_quadrature(count)
            geometry = ColumnField(
                np.zeros((len(warped_tets), 3, 3)),
                np.zeros(len(bary), dtype=int),
                bary,
                layers,
                1,
                warped_tets,
                len(outer),
                np.sort(faces, axis=1),
                depths,
                warped_volumes,
            )
            volume = 0.0
            breaks = geometry.breaks()
            for lower, upper in zip(breaks[:-1], breaks[1:]):
                volume += angular @ ((upper - lower) * geometry.volume_density((lower + upper) / 2))
            np.testing.assert_allclose(volume, warped_volumes.sum(), atol=1e-13)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "controls": rows,
            "quadratic_gap_extrema_passed": True,
            "finite_area_polynomial_virtual_work_and_quadrature_passed": True,
            "shell_and_solid_linear_fiber_mapping_passed": True,
            "independent_contact_controls": qp_controls,
        },
    )


if __name__ == "__main__":
    main()
