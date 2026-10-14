"""Read a retained failed contact state and check energy, gradient and Newton directions."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .contact_reference import IndependentContactSystem, block_reference
from .skfem_reference import SkfemElasticReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=3)
    parser.add_argument("--grasp-body-mesh", type=Path)
    parser.add_argument("--friction", type=float, default=0)
    parser.add_argument("--dhat", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.grasp_body_mesh is None:
        lower = block_reference((-0.005, -0.005, -0.005), (0.005, 0.005, 0), args.resolution, 1e6, 0)
        upper = block_reference((-0.005, -0.005, 0.001), (0.005, 0.005, 0.006), args.resolution, 1e6, 0)
        references = (("lower", lower), ("upper", upper))
    else:
        references = []
        for name, path, young, poisson, density in (
            ("body", args.grasp_body_mesh, 1e10, 0.3, 2000),
            ("left_shoe", args.failure.parent / "left_shoe_mesh.npz", 1e6, 0, 1000),
            ("right_shoe", args.failure.parent / "right_shoe_mesh.npz", 1e6, 0, 1000),
        ):
            with np.load(path) as saved:
                reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, density)
            references.append((name, reference))
    system = IndependentContactSystem(references, args.dhat, 1e6, args.friction, 1e-8)
    with np.load(args.failure) as saved:
        displacement, previous, boundary = saved["displacement"], saved["previous_displacement"], saved["boundary_dofs"]
        lagged = saved["lagged_displacement"]
        np.testing.assert_array_equal(system.rest, saved["rest_positions"])
    free = np.setdiff1d(np.arange(len(displacement)), boundary)
    tangential = ipctk.TangentialCollisions()
    if args.friction > 0:
        normal, positions = system.normal_collisions(lagged)
        tangential.build(system.mesh, positions, normal, system.barrier, args.friction)
    energy, gradient, matrix, _, _ = system.evaluate(displacement, previous, tangential, hessian=True)
    directions = []
    scale = np.max(np.abs(matrix.diagonal()))
    for relative_shift in (0, 1e-8, 1e-6, 1e-4, 1e-2, 1):
        operator = matrix[free][:, free] + relative_shift * scale * sparse.eye(len(free), format="csc")
        direction = np.zeros_like(displacement)
        direction[free] = splu(operator.tocsc()).solve(-gradient[free])
        current = system.mesh.vertices(system.rest + displacement.reshape(-1, 3))
        proposed = system.mesh.vertices(system.rest + (displacement + direction).reshape(-1, 3))
        safe = ipctk.compute_collision_free_stepsize(system.mesh, current, proposed)
        rows = []
        for fraction in (1, 0.5, 0.1, 0.01, 1e-3, 1e-6):
            alpha = fraction * min(1.0, safe)
            candidate = displacement + alpha * direction
            evaluated = system.evaluate(candidate, previous, tangential, hessian=False)
            rows.append(
                {
                    "alpha": alpha,
                    "energy_change_J": evaluated[0] - energy,
                    "predicted_change_J": alpha * (gradient @ direction),
                    "free_residual_N": np.linalg.norm(evaluated[1][free]),
                    "maximum_displacement_change_m": np.max(np.abs(alpha * direction)),
                }
            )
        directions.append(
            {
                "relative_diagonal_shift": relative_shift,
                "directional_derivative_J": gradient @ direction,
                "collision_free_step": safe,
                "trial_steps": rows,
            }
        )
        if relative_shift == 0:
            exact_direction = direction
    vector = exact_direction / np.linalg.norm(exact_direction)
    derivatives = []
    for epsilon in (1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12):
        first = system.evaluate(displacement + epsilon * vector, previous, tangential, hessian=False)
        second = system.evaluate(displacement - epsilon * vector, previous, tangential, hessian=False)
        derivatives.append(
            {
                "epsilon_m": epsilon,
                "analytic_energy_derivative_N": gradient @ vector,
                "finite_difference_energy_derivative_N": (first[0] - second[0]) / (2 * epsilon),
                "hessian_vector_relative_error": np.linalg.norm(
                    (first[1] - second[1]) / (2 * epsilon) - matrix @ vector
                )
                / np.linalg.norm(matrix @ vector),
            }
        )
    result = {
        "initial_intersections": bool(ipctk.has_intersections(system.mesh, system.mesh.vertices(system.rest))),
        "failed_state_intersections": bool(
            ipctk.has_intersections(system.mesh, system.mesh.vertices(system.rest + displacement.reshape(-1, 3)))
        ),
        "energy_J": energy,
        "free_residual_N": np.linalg.norm(gradient[free]),
        "directions": directions,
        "finite_difference_checks": derivatives,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
