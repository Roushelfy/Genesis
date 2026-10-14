"""Read a retained failed contact state and check energy, gradient and Newton directions."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy.sparse.linalg import splu

from .contact_reference import IndependentContactSystem, block_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--dhat", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lower = block_reference((-0.005, -0.005, -0.005), (0.005, 0.005, 0), args.resolution, 1e6, 0)
    upper = block_reference((-0.005, -0.005, 0.001), (0.005, 0.005, 0.006), args.resolution, 1e6, 0)
    system = IndependentContactSystem((("lower", lower), ("upper", upper)), args.dhat, 1e6, 0, 1e-8)
    with np.load(args.failure) as saved:
        displacement, previous, boundary = saved["displacement"], saved["previous_displacement"], saved["boundary_dofs"]
    free = np.setdiff1d(np.arange(len(displacement)), boundary)
    tangential = ipctk.TangentialCollisions()
    energy, gradient, matrix, _, _ = system.evaluate(displacement, previous, tangential, hessian=True)
    direction = np.zeros_like(displacement)
    direction[free] = splu(matrix[free][:, free].tocsc()).solve(-gradient[free])
    rows = []
    for alpha in (1, 0.5, 0.1, 0.01, 1e-3, 1e-6):
        candidate = displacement + alpha * direction
        candidate_energy, candidate_gradient, _, _, _ = system.evaluate(candidate, previous, tangential, hessian=False)
        rows.append(
            {
                "alpha": alpha,
                "energy_change_J": candidate_energy - energy,
                "predicted_change_J": alpha * (gradient @ direction),
                "free_residual_N": np.linalg.norm(candidate_gradient[free]),
                "maximum_displacement_change_m": np.max(np.abs(alpha * direction)),
            }
        )
    vector = direction / np.linalg.norm(direction)
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
        "energy_J": energy,
        "free_residual_N": np.linalg.norm(gradient[free]),
        "directional_derivative_J": gradient @ direction,
        "trial_steps": rows,
        "finite_difference_checks": derivatives,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
