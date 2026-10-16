"""Audit lagged versus updated friction equilibrium in a retained force-controlled grasp step."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .contact_reference import IndependentContactSystem
from .skfem_reference import SkfemElasticReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--body-mesh", type=Path, required=True)
    parser.add_argument("--coordinate-map", type=Path, required=True)
    parser.add_argument("--geometry", choices=("u", "egg"), required=True)
    parser.add_argument("--force", type=float, default=0.05)
    parser.add_argument("--force-steps", type=int, default=64)
    parser.add_argument("--failed-step", type=int, default=1)
    parser.add_argument("--gravity", nargs=3, type=float, default=(0, 0, -9.81))
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--dhat", type=float, default=2.5e-6)
    parser.add_argument("--coupled-jacobian", action="store_true")
    parser.add_argument("--newton-direction-check", action="store_true")
    parser.add_argument(
        "--rigid-schur", action="store_true", help="Audit the final six body-rigid and two backplate coordinates"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.newton_direction_check and not args.coupled_jacobian:
        parser.error("--newton-direction-check requires --coupled-jacobian")
    if args.rigid_schur and not args.newton_direction_check:
        parser.error("--rigid-schur requires --newton-direction-check")
    args.output.mkdir(parents=True, exist_ok=False)
    references = []
    inputs = {}
    for name, path, young, poisson, density in (
        ("body", args.body_mesh, 1e10, 0.3, 2000),
        ("left_shoe", args.failure.parent / "left_shoe_mesh.npz", 1e6, 0, 1000),
        ("right_shoe", args.failure.parent / "right_shoe_mesh.npz", 1e6, 0, 1000),
    ):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            references.append(
                (name, SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, density))
            )
    system = IndependentContactSystem(references, args.dhat, 1e6, args.friction, 1e-8)
    with np.load(args.failure) as saved:
        displacement, previous, lagged = (
            saved[key] for key in ("displacement", "previous_displacement", "lagged_displacement")
        )
        boundary = saved["boundary_dofs"]
        np.testing.assert_array_equal(saved["rest_positions"], system.rest)
        np.testing.assert_array_equal(saved["surface_triangles"], system.faces)
    coordinate_map = sparse.load_npz(args.coordinate_map)
    assert coordinate_map.shape[0] == len(displacement)
    assert coordinate_map[boundary].nnz == 0
    force, gravity = np.zeros_like(displacement), np.zeros_like(displacement)
    for column, drive in zip((-2, -1), ((-1, 1) if args.geometry == "u" else (1, -1))):
        back = coordinate_map[:, column].nonzero()[0]
        force[back] = drive * args.force / len(back)
    for part in system.parts:
        acceleration = np.zeros(part.reference.basis.N)
        acceleration[part.reference.component_dofs] = args.gravity
        section = slice(3 * part.first_node, 3 * part.first_node + part.reference.basis.N)
        gravity[section] = (part.reference.mass @ acceleration)[part.permutation]
    force += gravity * args.failed_step / args.force_steps

    def evaluate(state, stencil_state, hessian=False):
        normal, current = system.normal_collisions(stencil_state)
        tangential = ipctk.TangentialCollisions()
        tangential.build(system.mesh, current, normal, system.barrier, args.friction)
        return system.evaluate(state, previous, tangential, hessian=hessian, external_force=force)

    evaluations = {}
    for name, state in (("retained_lagged_stencil", lagged), ("updated_stencil", displacement)):
        energy, gradient, matrix, normal, contact_gradient = evaluate(displacement, state, hessian=True)
        part_rows = []
        for part in system.parts:
            section = slice(3 * part.first_node, 3 * part.first_node + part.reference.basis.N)
            positions = system.rest[part.first_node : part.first_node + len(part.reference.coordinates)]
            contact_force = -contact_gradient[section].reshape(-1, 3)
            resultant = contact_force.sum(axis=0)
            moment = np.cross(positions - part.reference.com, contact_force).sum(axis=0)
            part_rows.append(
                {
                    "name": part.name,
                    "contact_resultant_N": resultant.tolist(),
                    "contact_moment_about_com_Nm": moment.tolist(),
                }
            )
        evaluations[name] = {
            "energy_J": float(energy),
            "complete_constrained_residual_N": float(np.linalg.norm(coordinate_map.T @ gradient)),
            "complete_body_residual_N": float(np.linalg.norm(gradient[: references[0][1].basis.N])),
            "normal_collisions": len(normal),
            "parts": part_rows,
        }
        if name == "updated_stencil":
            updated_gradient, updated_matrix = gradient, matrix
    derivative_rows = []
    coupled_matrix = None
    force_api_relative_difference = None
    if args.coupled_jacobian:
        normal, positions = system.normal_collisions(displacement)
        tangential = ipctk.TangentialCollisions()
        tangential.build(system.mesh, positions, normal, system.barrier, args.friction)
        rest = system.mesh.vertices(system.rest)
        current_displacement = system.mesh.vertices(displacement.reshape(-1, 3))
        velocity = system.mesh.vertices((displacement - previous).reshape(-1, 3))
        potential_gradient = system.friction_potential.gradient(tangential, system.mesh, velocity)
        force_api = system.friction_potential.force(
            tangential, system.mesh, rest, current_displacement, velocity, system.barrier
        )
        force_api_relative_difference = float(
            np.linalg.norm(force_api + potential_gradient) / np.linalg.norm(potential_gradient)
        )
        velocity_jacobian = system.friction_potential.force_jacobian(
            tangential,
            system.mesh,
            rest,
            current_displacement,
            velocity,
            system.barrier,
            ipctk.TangentialPotential.DiffWRT.VELOCITIES,
        )
        geometry_jacobian = system.friction_potential.force_jacobian(
            tangential,
            system.mesh,
            rest,
            current_displacement,
            velocity,
            system.barrier,
            ipctk.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS,
        )
        frozen_shear_matrix = system.mesh.to_full_dof(
            system.friction_potential.hessian(tangential, system.mesh, velocity)
        )
        coupled_matrix = (
            updated_matrix - frozen_shear_matrix - system.mesh.to_full_dof(velocity_jacobian + geometry_jacobian)
        )
    body_count = references[0][1].basis.N
    for axis in range(3):
        direction = np.zeros_like(displacement)
        direction[:body_count].reshape(-1, 3)[:, axis] = 1
        for epsilon in (1e-9, 1e-10, 1e-11):
            first, second = displacement + epsilon * direction, displacement - epsilon * direction
            actual = (evaluate(first, first)[1] - evaluate(second, second)[1]) / (2 * epsilon)
            frozen = updated_matrix @ direction
            derivative_rows.append(
                {
                    "body_translation_axis": axis,
                    "epsilon_m": epsilon,
                    "updated_gradient_derivative_norm_N_per_m": float(np.linalg.norm(coordinate_map.T @ actual)),
                    "frozen_hessian_derivative_norm_N_per_m": float(np.linalg.norm(coordinate_map.T @ frozen)),
                    "frozen_hessian_relative_difference": float(
                        np.linalg.norm(coordinate_map.T @ (actual - frozen)) / np.linalg.norm(coordinate_map.T @ actual)
                    ),
                }
            )
            if coupled_matrix is not None:
                expected = coupled_matrix @ direction
                derivative_rows[-1]["coupled_jacobian_relative_difference"] = float(
                    np.linalg.norm(coordinate_map.T @ (actual - expected)) / np.linalg.norm(coordinate_map.T @ actual)
                )
    newton_checks = None
    rigid_schur = None
    if args.newton_direction_check:
        assert coupled_matrix is not None, "The coupled Jacobian is required"
        reduced_matrix = (coordinate_map.T @ coupled_matrix @ coordinate_map).tocsc()
        reduced_gradient = coordinate_map.T @ updated_gradient
        reduced_direction = splu(reduced_matrix).solve(-reduced_gradient)
        direction = coordinate_map @ reduced_direction
        unit_direction = direction / np.linalg.norm(direction)
        expected = coordinate_map.T @ coupled_matrix @ unit_direction
        checks = []
        for epsilon in (1e-8, 1e-9, 1e-10, 1e-11):
            plus, minus = displacement + epsilon * unit_direction, displacement - epsilon * unit_direction
            actual = coordinate_map.T @ (evaluate(plus, plus)[1] - evaluate(minus, minus)[1]) / (2 * epsilon)
            checks.append(
                {
                    "epsilon_m": epsilon,
                    "directional_derivative_relative_difference": float(
                        np.linalg.norm(actual - expected) / np.linalg.norm(actual)
                    ),
                    "actual_residual_merit_derivative": float(reduced_gradient @ actual),
                    "analytic_residual_merit_derivative": float(reduced_gradient @ expected),
                    "normal_collision_counts_plus_minus": [
                        len(system.normal_collisions(state)[0]) for state in (plus, minus)
                    ],
                }
            )
        newton_checks = {
            "relative_linear_solve_residual": float(
                np.linalg.norm(reduced_matrix @ reduced_direction + reduced_gradient) / np.linalg.norm(reduced_gradient)
            ),
            "maximum_newton_displacement_m": float(np.max(np.abs(direction))),
            "finite_differences": checks,
        }
        if args.rigid_schur:
            elastic_count = coordinate_map.shape[1] - 8
            block = reduced_matrix[:elastic_count, :elastic_count]
            elastic_factor = splu(block.tocsc())
            er = reduced_matrix[:elastic_count, elastic_count:].toarray()
            re = reduced_matrix[elastic_count:, :elastic_count]
            rr = reduced_matrix[elastic_count:, elastic_count:].toarray()
            schur = rr - re @ elastic_factor.solve(er)
            effective_rhs = reduced_gradient[elastic_count:] - re @ elastic_factor.solve(
                reduced_gradient[:elastic_count]
            )
            left_vectors, singular_values, _ = np.linalg.svd(schur)
            projections = left_vectors.T @ effective_rhs
            rigid_schur = {
                "coordinate_count": 8,
                "coordinate_definition": "Six free body rigid-mode coordinates with rotations scaled by body radius, then two force-controlled backplate translations",
                "singular_values_N_per_m": singular_values.tolist(),
                "smallest_over_largest_singular_value": float(singular_values[-1] / singular_values[0]),
                "effective_rhs_projections_N": projections.tolist(),
                "weakest_mode_linear_displacement_m": float(abs(projections[-1] / singular_values[-1])),
                "interpretation": "Local linear conditioning diagnostic; a weak mode does not alone establish physical wrench infeasibility",
            }
            np.savez_compressed(
                args.output / "rigid_schur.npz",
                matrix=schur,
                effective_rhs=effective_rhs,
                left_singular_vectors=left_vectors,
            )
    inputs.update(
        {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.failure, args.coordinate_map)}
    )
    body = references[0][1]
    result = {
        "scope": "Diagnostic of saved failed free-body force/gravity step; no converged-contact claim",
        "input_sha256": inputs,
        "failed_step": args.failed_step,
        "force_steps": args.force_steps,
        "gravity_m_per_s2": args.gravity,
        "body_mass_kg": body.total_mass,
        "body_com_m": body.com.tolist(),
        "evaluations": evaluations,
        "updated_force_derivative_checks": derivative_rows,
        "friction_force_api_relative_difference": force_api_relative_difference,
        "newton_direction_checks": newton_checks,
        "rigid_schur": rigid_schur,
        "friction_force_jacobian_api": getattr(system.friction_potential.force_jacobian, "__doc__", ""),
        "limitations": "A large updated-stencil residual does not prove physical infeasibility. Geometry, contact support and total-wrench feasibility require separate checks.",
    }
    np.savez_compressed(
        args.output / "diagnostic_fields.npz", applied_force_N=force, updated_gradient_N=updated_gradient
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(evaluations, indent=2), flush=True)


if __name__ == "__main__":
    main()
