"""Independent P2 response to a declared smooth finite-area load, separate from contact."""

import argparse
from pathlib import Path
import resource
import time

import numpy as np
from skfem import FacetBasis, LinearForm, asm

from .artifacts import file_id, runtime_identity, write_json
from .cvxopt_factor import CvxoptCholeskyFactor
from .p2_elastic import FreeElasticP2, CholeskyFactor


@LinearForm
def axial_traction(test, data):
    return data.pressure * test[2]


def finite_area_load(model, outer_count, radius, order):
    """Integrate the same prescribed profile against the independent P2 shape functions."""
    reference = model.reference
    boundary = reference.mesh.boundary_facets()
    outer = boundary[(reference.mesh.facets[:, boundary] < outer_count).all(axis=0)]
    basis = FacetBasis(reference.mesh, reference.element, facets=outer, intorder=order)
    assert (basis.dx > 0).all(), "Physical force packets require positive surface quadrature weights"
    points = np.asarray(basis.global_coordinates()).transpose(1, 2, 0)
    shape = np.maximum(1 - (points[..., :2] ** 2).sum(axis=-1) / radius**2, 0) ** 3
    upper = points[..., 2] > 0
    pressure = np.zeros_like(shape)
    integrals = []
    for side, sign in ((False, 1), (True, -1)):
        selected = upper == side
        integral = float(np.sum(shape[selected] * basis.dx[selected]))
        assert integral > 0
        pressure[selected] = sign * shape[selected] / integral
        integrals.append(integral)
    nodal = asm(axial_traction, basis, pressure=pressure).reshape(-1, 3)
    point_force = np.zeros_like(points)
    point_force[..., 2] = pressure * basis.dx
    selected = shape > 0
    points, point_force = points[selected], point_force[selected]
    force_defect = nodal.sum(axis=0) - point_force.sum(axis=0)
    moment_defect = np.cross(model.vertices, nodal).sum(axis=0) - np.cross(points, point_force).sum(axis=0)
    np.testing.assert_allclose(force_defect, 0, atol=1e-12)
    np.testing.assert_allclose(moment_defect, 0, atol=1e-14)
    return nodal, points, point_force, basis.dx[selected], pressure[selected], integrals


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--radius", type=float, default=0.002)
    cli.add_argument("--loads", type=float, nargs="+", default=[0.1])
    cli.add_argument("--quadrature-orders", type=int, nargs="+", default=[8, 10])
    cli.add_argument("--through-wall-ordering", action="store_true")
    cli.add_argument("--cholespy-path", type=Path)
    cli.add_argument("--cvxopt-factor", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert not (args.cvxopt_factor and args.cholespy_path), "Select one explicit CHOLMOD adapter"
    assert args.radius > 0 and min(args.loads) > 0
    assert len(set(args.quadrature_orders)) == len(args.quadrature_orders)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with np.load(args.mesh) as saved:
        vertices, tetrahedra, outer, faces = (
            saved[key] for key in ("vertices", "tetrahedra", "outer_vertices", "mid_faces")
        )
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "radius_m": args.radius,
            "loads_n": args.loads,
            "quadrature_orders": args.quadrature_orders,
            "through_wall_ordering": args.through_wall_ordering,
            "cholespy_path": str(args.cholespy_path) if args.cholespy_path is not None else None,
            "cvxopt_long_index_factor": args.cvxopt_factor,
            "profile": "Axial inward traction proportional to max(1-rho^2/radius^2,0)^3 on each outer pole, normalized separately to the declared force. Radius is prescribed before stress evaluation.",
        },
    )
    model = FreeElasticP2(
        vertices,
        tetrahedra,
        1e10,
        0.3,
        2000,
        column_faces=faces,
        through_wall_ordering=args.through_wall_ordering,
        cholespy_path=args.cholespy_path,
        cvxopt_factor=args.cvxopt_factor,
    )
    factor_seconds = time.perf_counter() - start
    rows = []
    for order in args.quadrature_orders:
        output = args.output / f"order-{order}"
        output.mkdir()
        nodal, points, point_force, point_area, point_pressure, integrals = finite_area_load(
            model, len(outer), args.radius, order
        )
        for load in args.loads:
            forces = load * nodal
            stress, displacement, correction, residual = model.solve(forces)
            force_scale = np.linalg.norm(forces)
            correction_relative = float(np.linalg.norm(correction) / force_scale)
            residual_relative = float(np.linalg.norm(residual) / force_scale)
            resultant = forces.sum(axis=0)
            moment = np.cross(model.vertices, forces).sum(axis=0)
            wrench_error = max(np.linalg.norm(resultant) / load, np.linalg.norm(moment) / (load * 0.03))
            dev = stress - np.trace(stress, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
            mises = np.sqrt(1.5 * (dev * dev).sum(axis=(-2, -1)))
            path = output / f"load-{load:g}.npz"
            np.savez_compressed(
                path,
                vertices_m=model.vertices,
                stress_pa=stress,
                displacement_m=displacement,
                forces_n=forces,
                load_n=load,
                traction_points_m=points,
                traction_point_forces_n=load * point_force,
                traction_quadrature_area_m2=point_area,
                axial_traction_pa=load * point_pressure,
            )
            rows.append(
                {
                    "quadrature_order": order,
                    "load_n": load,
                    "pressure_shape_integrals_m2": integrals,
                    "force_moment_relative_error": float(wrench_error),
                    "balance_correction_relative": correction_relative,
                    "equation_relative_residual": residual_relative,
                    "elastic_equations_admitted": bool(
                        wrench_error <= 1e-8 and correction_relative <= 1e-8 and residual_relative <= 1e-7
                    ),
                    "raw_maximum_mises_pa": float(mises.max()),
                    "raw_maximum_tension_pa": float(np.linalg.eigvalsh(stress)[..., -1].max()),
                    "maximum_displacement_gradient": float(
                        np.linalg.norm(model.displacement_gradient(displacement), axis=(-2, -1)).max()
                    ),
                    "field": file_id(path),
                }
            )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "radius_m": args.radius,
            "mass_kg": float(model.reference.total_mass),
            "through_wall_ordering": args.through_wall_ordering,
            "cholespy_binary": file_id(Path(model.factor.module_path))
            if isinstance(model.factor, CholeskyFactor)
            else None,
            "cvxopt_binary": file_id(Path(model.factor.module_path))
            if isinstance(model.factor, CvxoptCholeskyFactor)
            else None,
            "cvxopt_index_bytes": model.factor.index_bytes if isinstance(model.factor, CvxoptCholeskyFactor) else None,
            "factor_seconds": factor_seconds,
            "total_seconds": time.perf_counter() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cases": rows,
            "physical_reference_qualified": False,
            "scope": "Free full solid under independently prescribed finite-area traction. All six factorization gauges are removed; no egg vertices are physically pinned. Retained quadrature point forces and consistent nodal forces preserve force, moment and rigid virtual work. Quadrature, surface and wall convergence still require measured controls. This is a same-traction elasticity benchmark, not a solved plate-contact state or an end-to-end ranking.",
        },
    )
    assert all(row["elastic_equations_admitted"] for row in rows), "Prescribed-traction equations failed"


if __name__ == "__main__":
    main()
