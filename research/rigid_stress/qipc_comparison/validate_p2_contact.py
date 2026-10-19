"""Check quadratic elasticity against exact stress and bending before contact reference use."""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTet

from .artifacts import runtime_identity, write_json
from .p2_elastic import FreeElasticP2, quadratic_triangle_minimum
from .contact_fem_exact_columns import ColumnField
from .prepare_inputs import make_mesh


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mesh = MeshTet().refined(1)
    vertices = mesh.p.T * np.array([0.002, 0.002, 0.0005])
    rows = []
    for ordered in (False, True):
        model = FreeElasticP2(vertices, mesh.t.T, 1e10, 0.3, 2000.0, column_faces=np.empty((0, 3)) if ordered else None)
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
        values = 2 + coordinates[tets, 0] - 3 * coordinates[tets, 1] + 5 * coordinates[tets, 2]
        tensors = values[..., None, None] * np.eye(3)
        field = ColumnField(
            tensors, np.zeros(2, dtype=int), weights, layers, 1, tets, 3, np.sort(faces, axis=1), depths
        )
        for depth in np.linspace(0, 1, 37):
            actual = field.sample(np.full(2, depth), basis)
            expected = (2 + weights[:, 1] - 3 * weights[:, 2] + 5 * depth)[:, None, None] * np.eye(2)
            np.testing.assert_allclose(actual, expected, atol=1e-13)
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
    write_json(
        args.output / "result.json",
        {"identity": runtime_identity(), "controls": rows, "quadratic_gap_extrema_passed": True},
    )


if __name__ == "__main__":
    main()
