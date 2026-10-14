"""Validate a rigid backplate force coordinate in independently solved unilateral block contact."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import sparse

from .contact_reference import IndependentContactSystem, block_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=3)
    parser.add_argument("--dhat", type=float, default=2.5e-6)
    parser.add_argument("--force", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gap = 0.001
    lower = block_reference((-0.005, -0.005, -0.005), (0.005, 0.005, 0), args.resolution, 1e6, 0)
    upper = block_reference((-0.005, -0.005, gap), (0.005, 0.005, gap + 0.005), args.resolution, 1e6, 0)
    system = IndependentContactSystem((("lower", lower), ("upper", upper)), args.dhat, 1e6, 0, 1e-8)
    bottom = np.flatnonzero(np.isclose(system.rest[:, 2], -0.005))
    top = np.flatnonzero(np.isclose(system.rest[:, 2], gap + 0.005))
    nodes = np.concatenate((bottom, top))
    original_fixed = (nodes[:, None] * 3 + np.arange(3)).reshape(-1)
    values = np.zeros((len(nodes), 3))
    values[len(bottom) :, 2] = -(gap - 4 * args.dhat)
    initial, _, _, _, _ = system.solve_supported(
        original_fixed, values.reshape(-1), 12, phase="approach", failure_path=args.output / "approach_failure.npz"
    )
    values[len(bottom) :, 2] = -(gap + 1e-5)
    initial, gradient, _, _, _ = system.solve_supported(
        original_fixed,
        values.reshape(-1),
        int(np.ceil((1e-5 + 4 * args.dhat) / (args.dhat / 4))),
        initial_displacement=initial,
        phase="preload_initialization",
        failure_path=args.output / "preload_failure.npz",
    )
    full_count = len(initial)
    fixed = np.concatenate(
        ((bottom[:, None] * 3 + np.arange(3)).reshape(-1), (top[:, None] * 3 + np.arange(2)).reshape(-1))
    )
    back_normal = 3 * top + 2
    independent = np.setdiff1d(np.arange(full_count), np.concatenate((fixed, back_normal)))
    rows = np.concatenate((independent, back_normal))
    columns = np.concatenate((np.arange(len(independent)), np.full(len(back_normal), len(independent))))
    coordinates = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(full_count, len(independent) + 1))
    sparse.save_npz(args.output / "coordinate_map.npz", coordinates)
    applied = np.zeros(full_count)
    applied[back_normal] = -args.force / len(back_normal)
    initial_force = np.zeros(full_count)
    initial_force[back_normal] = gradient[back_normal].sum() / len(back_normal)
    displacement, gradient, contact_force, energy, _ = system.solve_supported(
        fixed,
        np.zeros(len(fixed)),
        12,
        initial_displacement=initial,
        phase="force_control",
        coordinate_map=coordinates,
        applied_force=applied,
        initial_force=initial_force,
        failure_path=args.output / "force_failure.npz",
    )
    residual = np.linalg.norm(coordinates.T @ gradient)
    assert residual <= 1e-9
    np.testing.assert_allclose(displacement[back_normal], displacement[back_normal[0]], rtol=0, atol=1e-14)
    force = contact_force[: lower.basis.N].reshape(-1, 3).sum(axis=0)
    np.testing.assert_allclose(force, [0, 0, -args.force], rtol=1e-8, atol=1e-9)
    part_fields = []
    for part in system.parts:
        vector = np.empty(part.reference.basis.N)
        section = slice(part.first_node * 3, part.first_node * 3 + part.reference.basis.N)
        vector[part.permutation] = displacement[section]
        tensor, vm = part.reference.corner_stress(vector)
        part_fields.append((tensor, vm))
    result = {
        "scope": "Supported full3D force-controlled contact with an exact rigid-backplate translation coordinate",
        "resolution": args.resolution,
        "dhat_m": args.dhat,
        "actuator_force_N": args.force,
        "solved_interface_force_N": force.tolist(),
        "solved_back_displacement_m": float(displacement[back_normal[0]]),
        "analytic_zero_gap_back_displacement_m": -(gap + args.force * 0.01 / (1e6 * 1e-4)),
        "analytic_uniform_peak_Pa": args.force / 1e-4,
        "raw_global_peak_Pa": max(float(field[1].max()) for field in part_fields),
        "complete_constrained_residual_N": float(residual),
        "total_potential_J": float(energy),
        "accepted_steps": [asdict(row) for row in system.accepted_steps],
        "protocol": "Displacement initialization before touch/preload; final stage specifies actuator force and solves the back displacement",
    }
    np.savez_compressed(
        args.output / "fields.npz",
        rest_positions=system.rest,
        displacement=displacement,
        contact_force_N=contact_force,
        total_gradient_N=gradient,
        applied_force_N=applied,
        lower_tensor_Pa=part_fields[0][0],
        lower_vm_Pa=part_fields[0][1],
        upper_tensor_Pa=part_fields[1][0],
        upper_vm_Pa=part_fields[1][1],
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "accepted_steps"}), flush=True)


if __name__ == "__main__":
    main()
