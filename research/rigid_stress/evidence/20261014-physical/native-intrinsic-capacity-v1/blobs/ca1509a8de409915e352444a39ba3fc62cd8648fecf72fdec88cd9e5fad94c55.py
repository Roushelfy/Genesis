"""Bracket fixed-stencil wrench support with inner/outer Coulomb polygons."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .contact_reference import IndependentContactSystem
from .skfem_reference import SkfemElasticReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--body-mesh", type=Path, required=True)
    parser.add_argument("--force", type=float, default=0.05)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--dhat", type=float, default=2.5e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    references, input_hashes = [], {}
    for name, path, young, poisson, density in (
        ("body", args.body_mesh, 1e10, 0.3, 2000),
        ("left_shoe", args.failure.parent / "left_shoe_mesh.npz", 1e6, 0, 1000),
        ("right_shoe", args.failure.parent / "right_shoe_mesh.npz", 1e6, 0, 1000),
    ):
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as mesh:
            references.append(
                (name, SkfemElasticReference(mesh["vertices"], mesh["tetrahedra"], young, poisson, density))
            )
    system = IndependentContactSystem(references, args.dhat, 1e6, args.friction, 1e-8)
    with np.load(args.failure) as saved:
        np.testing.assert_array_equal(saved["rest_positions"], system.rest)
        np.testing.assert_array_equal(saved["surface_triangles"], system.faces)
        assert np.all(saved["boundary_dofs"] % 3 != 0), "This audit requires free body and x-free backplates"
        displacement = saved["displacement"]
    input_hashes[str(args.failure)] = hashlib.sha256(args.failure.read_bytes()).hexdigest()
    normal, current = system.normal_collisions(displacement)
    if not len(normal):
        raise ValueError("Fixed-stencil support requires at least one active contact")
    full_vertices = system.mesh.to_full_vertex_id()
    body = references[0][1]
    radius = np.linalg.norm(body.coordinates - body.com, axis=1).max()
    wrench_columns, actual_columns, collision_rows = [], [], []
    reconstructed = np.zeros_like(displacement)
    for index in range(len(normal)):
        collision = normal[index]
        ids = np.array(collision.vertex_ids(system.mesh.edges, system.mesh.faces))
        ids = ids[ids >= 0]
        nodes = full_vertices[ids]
        positions = collision.dof(current, system.mesh.edges, system.mesh.faces)
        force = -system.barrier.gradient(collision, positions).reshape(-1, 3)
        np.add.at(reconstructed.reshape(-1, 3), nodes, force)
        groups = system.groups[nodes]
        assert set(groups) in ({0, 1}, {0, 2}), "Expected body/shoe stencils only"
        body_mask = groups == 0
        resultant = force[body_mask].sum(axis=0)
        magnitude = np.linalg.norm(resultant)
        assert magnitude > 0
        inward = resultant / magnitude
        coefficients = force @ inward / magnitude
        np.testing.assert_allclose(force / magnitude, coefficients[:, None] * inward, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(coefficients[body_mask].sum(), 1, rtol=0, atol=1e-10)
        first = np.eye(3)[np.argmin(abs(inward))]
        first -= inward * (first @ inward)
        first /= np.linalg.norm(first)
        second = np.cross(inward, first)
        frame = np.column_stack((inward, first, second))
        columns = np.zeros((8, 3))
        for axis in range(3):
            local = coefficients[:, None] * frame[:, axis]
            columns[:3, axis] = local[body_mask].sum(axis=0)
            columns[3:6, axis] = (
                np.cross(system.rest[nodes[body_mask]] - body.com, local[body_mask]).sum(axis=0) / radius
            )
            for shoe in (1, 2):
                columns[5 + shoe, axis] = local[groups == shoe, 0].sum()
        wrench_columns.append(columns)
        actual_columns.append(magnitude)
        collision_rows.append(
            {
                "index": index,
                "nodes": nodes.tolist(),
                "groups": groups.tolist(),
                "current_normal_force_N": float(magnitude),
                "inward_normal": inward.tolist(),
                "reference_application_point_m": (coefficients[body_mask] @ system.rest[nodes[body_mask]]).tolist(),
            }
        )
    # __getitem__ returns a borrowed C++ collision. Release it while its owner is live.
    del collision
    complete = -system.mesh.to_full_dof(system.barrier.gradient(normal, system.mesh, current))
    np.testing.assert_allclose(reconstructed, complete, rtol=1e-10, atol=1e-12)
    matrix = np.concatenate(wrench_columns, axis=1)
    np.savez_compressed(
        args.output / "fixed_stencil.npz",
        wrench_matrix=matrix,
        normal_force_N=actual_columns,
        rest_positions=system.rest,
        displacement=displacement,
    )
    records = []
    for gravity in ((0, 0, 0), (0, -9.81 / 64, 0), (0, -9.81, 0), (0, 0, -9.81 / 64), (0, 0, -9.81)):
        target = np.r_[-body.total_mass * np.array(gravity), np.zeros(3), args.force, -args.force]
        for sides in (8, 16, 32):
            angles = np.arange(sides) * (2 * np.pi / sides)
            rays = np.column_stack((np.ones(sides), args.friction * np.cos(angles), args.friction * np.sin(angles)))
            inner_matrix = np.concatenate([column @ rays.T for column in wrench_columns], axis=1)
            inner = linprog(
                np.zeros(inner_matrix.shape[1]),
                A_eq=inner_matrix,
                b_eq=target,
                bounds=(0, None),
                options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
            )
            single_inequality = np.column_stack((-np.full(sides, args.friction), np.cos(angles), np.sin(angles)))
            single_inequality = np.vstack((single_inequality, [-1, 0, 0]))
            inequality = sparse.block_diag([single_inequality] * len(normal)).toarray()
            outer = linprog(
                np.zeros(matrix.shape[1]),
                A_eq=matrix,
                b_eq=target,
                A_ub=inequality,
                b_ub=np.zeros(len(inequality)),
                bounds=(None, None),
                options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
            )
            row = {
                "gravity_m_per_s2": list(gravity),
                "polygon_sides": sides,
                "inner_status": inner.status,
                "inner_message": inner.message,
                "outer_status": outer.status,
                "outer_message": outer.message,
            }
            if inner.success:
                weights = inner.x.reshape(len(normal), sides)
                forces = weights @ rays
                residual = np.linalg.norm(matrix @ forces.reshape(-1) - target)
                violation = np.linalg.norm(forces[:, 1:], axis=1) - args.friction * forces[:, 0]
                assert residual <= 1e-9 and violation.max() <= 1e-10 and forces[:, 0].min() >= -1e-10
                row.update(
                    classification="fixed_stencil_wrench_feasible_in_exact_coulomb_cones",
                    complete_scaled_wrench_residual_N=float(residual),
                    coefficients_N=forces.tolist(),
                )
            elif outer.status == 2:
                # Farkas certificate: Aeq.T*y + Aub.T*z = 0, z>=0, b.T*y<0.
                equality = np.column_stack((matrix.T, inequality.T))
                objective = np.r_[target, np.zeros(len(inequality))]
                certificate = linprog(
                    objective,
                    A_eq=equality,
                    b_eq=np.zeros(matrix.shape[1]),
                    bounds=[(-1, 1)] * 8 + [(0, None)] * len(inequality),
                    options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
                )
                assert certificate.success
                multipliers = certificate.x
                dual_residual = equality @ multipliers
                bounds = []
                for column in wrench_columns:
                    shoe = 6 if abs(column[6, 0]) > 0.5 else 7
                    sign = np.sign(target[shoe])
                    minimum_x = sign * column[shoe, 0] - args.friction / np.cos(np.pi / sides) * np.linalg.norm(
                        column[shoe, 1:]
                    )
                    assert minimum_x > 0, "Finite multiplier bound requires inward x force for every allowed cone"
                    normal_bound = abs(target[shoe]) / minimum_x
                    bounds.extend(
                        (
                            normal_bound,
                            args.friction / np.cos(np.pi / sides) * normal_bound,
                            args.friction / np.cos(np.pi / sides) * normal_bound,
                        )
                    )
                bound_error = abs(dual_residual) @ np.array(bounds)
                margin = -target @ multipliers[:8]
                assert multipliers[8:].min() >= -1e-10 and margin > bound_error + 1e-9
                row.update(
                    classification="fixed_stencil_outer_cone_infeasibility_with_bounded_dual_certificate",
                    separating_margin_N=float(margin),
                    bounded_dual_error_N=float(bound_error),
                    multipliers=multipliers.tolist(),
                )
            else:
                row["classification"] = "unresolved_inner_outer_bracket"
            records.append(row)
    report = {
        "scope": __doc__,
        "input_sha256": input_hashes,
        "normal_collision_count": len(normal),
        "body_mass_kg": float(body.total_mass),
        "body_com_m": body.com.tolist(),
        "moment_scale_m": float(radius),
        "collisions": collision_rows,
        "rows": records,
        "limitations": "The contact geometry and existing stencil are fixed. Positive feasibility supplies possible static wrenches, not deformation/contact convergence. Negative certificates exclude only this fixed stencil's admissible force distributions; additional contacts, deformation and pose changes can alter support. Force/moment checks use the linear FEM reference rigid modes, with two x-free guided backplates. No general grasp infeasibility or resolved-contact acceptance is claimed.",
        "binding_reference": "https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/python/src/potentials/potential.hpp",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"normal_collision_count": len(normal), "classifications": [r["classification"] for r in records]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
