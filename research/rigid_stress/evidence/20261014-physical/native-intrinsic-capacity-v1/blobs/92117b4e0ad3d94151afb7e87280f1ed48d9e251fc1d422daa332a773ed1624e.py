"""Qualify nested P2 transfer, exact affine-tensor volume norms and the Galerkin energy identity."""

import argparse
import json
from pathlib import Path

import numpy as np
from skfem import MeshTet

from .audit_frozen_traction_bulk_axis import prolongation, volume_squared
from .frozen_traction_bulk_reference import assemble_samples
from .refine_thickness_tail_regions import verify_parent_partition
from .skfem_reference import SkfemElasticReference


def reference(mesh):
    vertices, tetrahedra = mesh.p.T.copy(), mesh.t.T.copy()
    determinants = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
    flipped = determinants < 0
    tetrahedra[flipped, 2:4] = tetrahedra[flipped, 3:1:-1]
    return SkfemElasticReference(vertices, tetrahedra, 7e4, 0.3, 2000)


def manufactured_tensor(points):
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    tensor = np.zeros((*points.shape[:-1], 3, 3))
    tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2] = x, 2 * y, 3 * z
    tensor[..., 0, 1] = tensor[..., 1, 0] = y
    tensor[..., 0, 2] = tensor[..., 2, 0] = z
    tensor[..., 1, 2] = tensor[..., 2, 1] = x
    return tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coarse_mesh = MeshTet.init_tensor(*[np.array([0.0, 1.0])] * 3)
    fine_mesh = coarse_mesh.refined(np.array([0, 1]))
    partition = verify_parent_partition(coarse_mesh, fine_mesh)
    coarse, fine = reference(coarse_mesh), reference(fine_mesh)
    matrix, polynomial_errors = prolongation(coarse, fine, partition["parent_ids"])
    corners = fine.mesh.p.T[fine.mesh.t.T]
    volume = abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    sigma = manufactured_tensor(corners)
    integral = volume_squared(sigma, volume)
    exact = 20 / 3
    np.testing.assert_allclose(integral, exact, rtol=1e-12, atol=1e-12)
    barycentric = np.full((4, 4), 0.1381966011250105)
    np.fill_diagonal(barycentric, 0.5854101966249685)
    quadrature_points = np.einsum("qc,eci->eqi", barycentric, corners)
    quadrature = manufactured_tensor(quadrature_points)
    independently_integrated = float(np.sum(volume * np.mean(np.sum(quadrature**2, axis=(-1, -2)), axis=1)))
    np.testing.assert_allclose(integral, independently_integrated, rtol=1e-12, atol=1e-12)
    yz = np.array([(y, z) for y in [0.2, 0.5, 0.8] for z in [0.2, 0.5, 0.8]])
    points = np.concatenate((np.column_stack((np.zeros(len(yz)), yz)), np.column_stack((np.ones(len(yz)), yz))))
    weights = 1 + 0.3 * yz[:, 0] + 0.2 * yz[:, 1]
    forces = np.zeros_like(points)
    forces[: len(yz), 0] = weights
    forces[len(yz) :, 0] = -weights
    solutions = [item.solve(assemble_samples(item, points, forces)) for item in [coarse, fine]]
    assert all(item.residual_budget_fraction <= 1 for item in solutions)
    embedded = np.empty(fine.basis.N)
    for component in range(3):
        embedded[fine.component_dofs[:, component]] = (
            matrix @ solutions[0].displacement[coarse.component_dofs[:, component]]
        )
    delta = solutions[1].displacement - embedded
    error_squared = float(delta @ (fine.stiffness @ delta))
    gap = 2 * (solutions[1].energy_j - solutions[0].energy_j)
    np.testing.assert_allclose(error_squared, gap, rtol=1e-8, atol=1e-12)
    assert gap > 0
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "manufactured_volume_tensor.npz",
        fine_corner_positions_m=corners,
        fine_volume_m3=volume,
        affine_tensor_Pa=sigma,
    )
    np.savez_compressed(
        args.output / "nested_discrete_solutions.npz",
        coarse_displacement_m=solutions[0].displacement,
        fine_displacement_m=solutions[1].displacement,
        embedded_coarse_displacement_m=embedded,
        common_quadrature_positions_m=points,
        common_quadrature_forces_N=forces,
    )
    report = {
        "scope": __doc__,
        "coarse_tetrahedra": coarse.mesh.nelements,
        "fine_tetrahedra": fine.mesh.nelements,
        "polynomial_transfer_maximum_errors": polynomial_errors,
        "analytic_affine_tensor_squared_volume_integral_Pa2_m3": exact,
        "computed_exact_simplex_formula_Pa2_m3": integral,
        "independent_four_point_quadrature_Pa2_m3": independently_integrated,
        "nested_energy_error_squared_J": error_squared,
        "twice_discrete_energy_increase_J": gap,
        "energy_identity_absolute_difference_J": abs(error_squared - gap),
        "original_force_budget_fractions": [item.residual_budget_fraction for item in solutions],
        "qualification_passed": True,
        "goal_completion": False,
        "limitations": "This is a numerical manufactured control on a unit cube with identical discrete point-quadrature loads, not a finite-footprint physical convergence case. Analytic tensor-volume integration, all quadratic nodal-transfer polynomials and a nonzero nested Galerkin energy defect are checked independently. Original body/contact/geometry/history/model accuracy and warmed native cost remain separate requirements.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
