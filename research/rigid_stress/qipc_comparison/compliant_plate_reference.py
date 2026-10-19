"""Free P2 solid with the predeclared compression-only physical spring pads."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import sparse
from skfem import FacetBasis

from .artifacts import file_id, runtime_identity, write_json
from .contact_qp import solve_contact_qp
from .p2_elastic import FreeElasticP2, quadratic_triangle_minimum, CholeskyFactor
from .pad_elastic_fixed_point import solve_pad_fixed_point


def pad_quadrature(model, outer_count, radius, order, sparse_interpolation=False):
    """Positive projected-area quadrature and its consistent P2 normal interpolation."""
    reference = model.reference
    boundary = reference.mesh.boundary_facets()
    outer = boundary[(reference.mesh.facets[:, boundary] < outer_count).all(axis=0)]
    corners = reference.mesh.p.T[reference.mesh.facets[:, outer].T]
    selected = (np.linalg.norm(corners[..., :2], axis=-1) <= radius).any(axis=1)
    facets = outer[selected]
    nodes, _, _ = model.surface(reference.mesh.facets[:, facets].T)
    if order == 0:
        basis = FacetBasis(
            reference.mesh,
            reference.element,
            facets=facets,
            quadrature=(np.full((2, 1), 1 / 3), np.array([0.5])),
        )
    else:
        basis = FacetBasis(reference.mesh, reference.element, facets=facets, intorder=order)
    assert (basis.dx > 0).all(), "Pad energy requires positive quadrature weights"
    area = basis.dx * np.abs(np.asarray(basis.normals)[2])
    assert (area > 0).all(), "The polar candidate must have nonzero projected area"
    points = np.asarray(basis.global_coordinates()).transpose(1, 2, 0).reshape(-1, 3)
    dofs = basis.element_dofs
    # P2 tetrahedron face basis has zero entries for off-face nodes. Keep only
    # z-component functions that are nonzero at the chosen face quadrature.
    rows, cols, values = [], [], []
    indices = np.arange(area.size).reshape(area.shape)
    for local, component in enumerate(basis.basis):
        value = np.asarray(component[0])[2]
        nonzero = (value != 0) & np.isin(dofs[local] // 3, nodes)[:, None]
        rows.extend(indices[nonzero])
        cols.extend(np.broadcast_to(dofs[local, :, None] // 3, area.shape)[nonzero])
        values.extend(value[nonzero])
    interpolation = sparse.coo_matrix(
        (values, (rows, np.searchsorted(nodes, cols))), shape=(area.size, len(nodes))
    ).tocsr()
    np.testing.assert_allclose(np.array(interpolation.sum(axis=1)).reshape(-1), 1, atol=1e-13)
    np.testing.assert_allclose(interpolation @ model.vertices[nodes], points, atol=1e-14)
    if not sparse_interpolation:
        interpolation = interpolation.toarray()
    return nodes, interpolation, points, area.reshape(-1), reference.mesh.facets[:, facets].T


def condense(model, nodes, sign, journal):
    """Project out all free-body rigid modes before extracting normal compliance."""
    compliance = np.empty((len(nodes), len(nodes)))
    for first in range(0, len(nodes), 16):
        selected = slice(first, first + 16)
        force = np.zeros((3 * len(model.vertices), len(nodes[selected])))
        force[3 * nodes[selected] + 2, np.arange(len(nodes[selected]))] = sign[selected]
        force -= model.mass_modes @ np.linalg.solve(model.gram, model.rigid_modes.T @ force)
        displacement = np.zeros_like(force)
        displacement[model.free] = model.factor.solve(force[model.free] / model.young)
        displacement -= model.rigid_modes @ np.linalg.solve(model.gram, model.mass_modes.T @ displacement)
        compliance[:, selected] = sign[:, None] * displacement[3 * nodes + 2]
        journal.write(json.dumps({"condensed_columns": min(first + 16, len(nodes))}) + "\n")
    reciprocity = float(np.linalg.norm(compliance - compliance.T) / np.linalg.norm(compliance))
    assert reciprocity < 1e-7
    return (compliance + compliance.T) / 2, reciprocity


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--qp-solver-path", type=Path, required=True)
    cli.add_argument("--cholespy-path", type=Path)
    cli.add_argument("--candidate-radius", type=float, default=0.003)
    cli.add_argument("--pad-stiffness", type=float, default=1e9)
    cli.add_argument("--quadrature-orders", type=int, nargs="+", default=[4, 6])
    cli.add_argument("--loads", type=float, nargs="+", default=[0.1])
    cli.add_argument("--fixed-point", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert args.pad_stiffness > 0 and args.candidate_radius > 0 and min(args.loads) > 0
    assert len(set(args.quadrature_orders)) == len(args.quadrature_orders)
    assert min(args.quadrature_orders) >= 0, "Order zero requests one explicit face-centroid point"
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with np.load(args.mesh) as saved:
        vertices, tets, outer, faces = (saved[key] for key in ("vertices", "tetrahedra", "outer_vertices", "mid_faces"))
    center = vertices.mean(axis=0)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "pad_stiffness_n_per_m3": args.pad_stiffness,
            "candidate_radius_m": args.candidate_radius,
            "quadrature_orders": args.quadrature_orders,
            "loads_n": args.loads,
            "full_elastic_fixed_point": args.fixed_point,
        },
    )
    model = FreeElasticP2(
        vertices,
        tets,
        1e10,
        0.3,
        2000,
        column_faces=faces,
        through_wall_ordering=True,
        cholespy_path=args.cholespy_path,
    )
    factor_seconds = time.perf_counter() - start
    outer_nodes, _, quadratic_faces = model.surface(faces)
    vertices = model.vertices
    lever = 0.03
    records = []
    previous_nodes, compliance, reciprocity = None, None, None
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for order in args.quadrature_orders:
            nodes, interpolation, points, area, candidate_faces = pad_quadrature(
                model, len(outer), args.candidate_radius, order, sparse_interpolation=args.fixed_point
            )
            top = vertices[nodes, 2] > center[2]
            sign = np.where(top, -1.0, 1.0)
            point_top = points[:, 2] > center[2]
            np.testing.assert_allclose(interpolation @ sign, np.where(point_top, -1, 1), atol=1e-13)
            initial_gap = np.where(top, outer[:, 2].max() - vertices[nodes, 2], vertices[nodes, 2] - outer[:, 2].min())
            rigid = np.column_stack(
                (
                    sign,
                    sign * (vertices[nodes, 1] - center[1]) / lever,
                    -sign * (vertices[nodes, 0] - center[0]) / lever,
                    top,
                )
            )
            if not args.fixed_point and (previous_nodes is None or not np.array_equal(previous_nodes, nodes)):
                compliance, reciprocity = condense(model, nodes, sign, journal)
                previous_nodes = nodes
            output = args.output / f"order-{order}"
            output.mkdir()
            operator = (
                {
                    "interpolation_data": interpolation.data,
                    "interpolation_indices": interpolation.indices,
                    "interpolation_indptr": interpolation.indptr,
                    "interpolation_shape": interpolation.shape,
                }
                if args.fixed_point
                else {"compliance_m_per_n": compliance, "interpolation": interpolation}
            )
            np.savez_compressed(
                output / "contact-operator.npz",
                nodes=nodes,
                sign=sign,
                rigid=rigid,
                initial_gap_m=initial_gap,
                projected_area_m2=area,
                points_m=points,
                **operator,
            )
            # Exact quadratic minima on every omitted boundary face check
            # whether the numerical search region excluded physical compression.
            selected_keys = {tuple(sorted(face)) for face in candidate_faces}
            omitted = np.array([tuple(sorted(face)) not in selected_keys for face in quadratic_faces[:, :3]])
            for load in args.loads:
                solve_start = time.perf_counter()
                scale = load / (args.pad_stiffness * np.pi * 0.002**2)
                target = np.array([0.0, 0.0, 0.0, 1.0])
                if args.fixed_point:
                    qp = solve_pad_fixed_point(
                        model, nodes, sign, interpolation, initial_gap, rigid, area, args.pad_stiffness, load, journal
                    )
                else:
                    qp = solve_contact_qp(
                        compliance * load / scale,
                        initial_gap / scale,
                        rigid,
                        target,
                        np.empty((0, 6), dtype=int),
                        args.qp_solver_path,
                        interpolation=interpolation,
                        round_limit=1,
                        journal=journal,
                        constraint_compliance=load / (scale * args.pad_stiffness * area),
                    )
                forces = np.zeros_like(vertices)
                forces[nodes, 2] = qp.nodal_force * load * sign
                stress, elastic, correction, residual = model.solve(forces)
                coordinate = qp.coordinates * scale
                displacement = elastic.copy()
                displacement[:, 2] += coordinate[0]
                displacement += np.cross(
                    np.array([coordinate[1] / lever, coordinate[2] / lever, 0.0]), vertices - center
                )
                bottom_gap = vertices[:, 2] + displacement[:, 2] - outer[:, 2].min()
                top_gap = outer[:, 2].max() + coordinate[3] - vertices[:, 2] - displacement[:, 2]
                body_gap = interpolation @ np.where(top, top_gap[nodes], bottom_gap[nodes])
                point_force = qp.multipliers * load
                law_force = args.pad_stiffness * area * np.maximum(-body_gap, 0)
                law_error = float(np.linalg.norm(point_force - law_force) / np.linalg.norm(point_force))
                equation_error = float(np.linalg.norm(residual) / np.linalg.norm(forces))
                correction_error = float(np.linalg.norm(correction) / np.linalg.norm(forces))
                force_moment_error = float(np.linalg.norm(rigid.T @ qp.nodal_force - target))
                outside_gap = float(
                    min(
                        quadratic_triangle_minimum(bottom_gap[quadratic_faces[omitted]]).min(),
                        quadratic_triangle_minimum(top_gap[quadratic_faces[omitted]]).min(),
                    )
                )
                point_vectors = np.zeros_like(points)
                point_vectors[:, 2] = point_force * np.where(point_top, -1, 1)
                transfer_error = float(
                    max(
                        np.linalg.norm(forces.sum(axis=0) - point_vectors.sum(axis=0)) / load,
                        np.linalg.norm(
                            np.cross(vertices, forces).sum(axis=0) - np.cross(points, point_vectors).sum(axis=0)
                        )
                        / (load * lever),
                    )
                )
                admitted = bool(
                    qp.candidate_passed
                    and max(force_moment_error, correction_error, transfer_error) < 1e-8
                    and equation_error < 1e-7
                    and law_error < 1e-7
                    and outside_gap >= -1e-8 * scale
                )
                dev = stress - np.trace(stress, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
                mises = np.sqrt(1.5 * (dev * dev).sum(axis=(-2, -1)))
                footprints = []
                for upper in (False, True):
                    mask = point_top == upper
                    compressed = mask & (body_gap < 0)
                    radii = np.linalg.norm(points[mask, :2], axis=1)
                    weights = point_force[mask]
                    permutation = np.argsort(radii)
                    cumulative = np.cumsum(weights[permutation]) / weights.sum()
                    footprints.append(
                        {
                            "upper": upper,
                            "normal_force_n": float(weights.sum()),
                            "compressed_projected_area_m2": float(area[compressed].sum()),
                            "force_rms_radius_m": float(np.sqrt(np.sum(weights * radii**2) / weights.sum())),
                            "force_99pct_radius_m": float(radii[permutation][np.searchsorted(cumulative, 0.99)]),
                            "maximum_pressure_pa": float(np.max(point_force[mask] / area[mask])),
                        }
                    )
                path = output / f"load-{load:g}.npz"
                np.savez_compressed(
                    path,
                    vertices_m=vertices,
                    outer_nodes=outer_nodes,
                    tetrahedra=tets,
                    element_order=2,
                    displacement_m=displacement,
                    elastic_displacement_m=elastic,
                    stress_pa=stress,
                    forces_n=forces,
                    contact_force_n=forces,
                    load_n=load,
                    rigid_coordinates_m=coordinate,
                    traction_points_m=points,
                    traction_point_forces_n=point_vectors,
                    traction_quadrature_area_m2=area,
                    axial_traction_pa=point_vectors[:, 2] / area,
                    pad_body_gap_m=body_gap,
                    pad_stiffness_n_per_m3=args.pad_stiffness,
                    equation_residual_n=residual,
                    balance_correction_n=correction,
                )
                row = {
                    "load_n": load,
                    "quadrature_order": order,
                    "quadrature_rule": "explicit_face_centroid" if order == 0 else "scikit_fem_degree",
                    "pad_equations_admitted": admitted,
                    "physical_reference_qualified": False,
                    "qp_status": qp.status,
                    "qp_stationarity": qp.stationarity if not args.fixed_point else None,
                    "contact_method": "full_elastic_fixed_point" if args.fixed_point else "independent_primal_dual_qp",
                    "fixed_point_force_change_relative": qp.stationarity if args.fixed_point else None,
                    "qp_complementarity": qp.complementarity,
                    "qp_fixed_passed": qp.fixed_passed,
                    "qp_solver_version": qp.solver_version,
                    "qp_solver_module": qp.solver_module,
                    "qp_iterations": qp.iterations,
                    "qp_solve_seconds": qp.solve_seconds,
                    "pad_law_relative_residual": law_error,
                    "equation_relative_residual": equation_error,
                    "balance_correction_relative": correction_error,
                    "force_moment_relative_error": force_moment_error,
                    "nodal_point_transfer_relative_error": transfer_error,
                    "minimum_omitted_boundary_gap_m": outside_gap,
                    "compliance_reciprocity_error": reciprocity,
                    "plate_travel_m": float(-coordinate[3]),
                    "pad_energy_j": float(0.5 * np.sum(point_force**2 / (args.pad_stiffness * area))),
                    "elastic_energy_j": float(0.5 * np.sum(forces * elastic)),
                    "raw_maximum_mises_pa": float(mises.max()),
                    "raw_maximum_tension_pa": float(np.linalg.eigvalsh(stress)[..., -1].max()),
                    "maximum_displacement_gradient": float(
                        np.linalg.norm(model.displacement_gradient(displacement), axis=(-2, -1)).max()
                    ),
                    "footprints": footprints,
                    "candidate_nodes": len(nodes),
                    "quadrature_points": len(points),
                    "solve_seconds": time.perf_counter() - solve_start,
                    "field": file_id(path),
                }
                records.append(row)
                journal.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "pad_stiffness_n_per_m3": args.pad_stiffness,
            "candidate_radius_m": args.candidate_radius,
            "factor_seconds": factor_seconds,
            "total_seconds": time.perf_counter() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cholespy_binary": file_id(Path(model.factor.module_path))
            if isinstance(model.factor, CholeskyFactor)
            else None,
            "cases": records,
            "physical_reference_qualified": False,
            "scope": "Full free small-strain P2 solid with compression-only spring pads of predeclared physical stiffness per projected reference area. Consistent positive facet quadrature, free plate travel and body tilts, no physical vertex pins. Negative body gap is pad compression. Full omitted-boundary quadratic gaps check the search extent. This changes the physical contact law and is separate from hard contact and prescribed traction. Mesh, quadrature, nonlinear geometry and matched Shell history remain separate qualifications.",
        },
    )


if __name__ == "__main__":
    main()
