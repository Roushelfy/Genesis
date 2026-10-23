"""Diagnose a semismooth Newton solve of the unchanged radial friction projection."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .hertz_history_cone import line_weights


def projection_residual(dual, offset, matrix, widths):
    increment = offset - matrix @ dual
    argument = increment + dual
    norms = np.linalg.norm(argument, axis=1)
    factors = np.minimum(1, widths / np.maximum(norms, np.finfo(float).tiny))
    projected = factors[:, None] * argument
    return increment - projected, increment, argument, projected


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--capture", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.capture), file_id(args.reference)],
            "scope": "Same staged prior, force equality and sixteen slope disks. Eliminate the weighted quadratic and force equality analytically; solve only their projection fixed point. Offline numerical diagnostic, not a physical model change or native performance result.",
        },
    )
    with np.load(args.capture) as saved:
        selected = np.flatnonzero((saved["input_normal"] > 0) & (saved["input_friction"] > 0))
        prior = saved["state_prior"][selected, :16].copy()
        target = (
            saved["input_force"][selected]
            / (saved["input_normal"][selected] * saved["input_friction"][selected])[:, None]
        )
    with np.load(args.reference) as saved:
        np.testing.assert_array_equal(saved["selected"], selected)
        reference = saved["reference"]
    nodes = np.linspace(0, 1, 17)
    widths = np.diff(nodes**2)
    weights = line_weights(nodes)[:16]
    difference = np.eye(16) - np.eye(16, k=1)
    matrix = difference @ np.diag(1 / weights) @ difference.T
    matrix[-1, -1] -= 1 / weights.sum()
    rows, solutions = [], []
    for case, (old, force) in enumerate(zip(prior, target)):
        offset = difference @ (old + (force - weights @ old) / weights.sum())
        dual = np.zeros((16, 2))
        history = []
        converged = False
        for iteration in range(64):
            residual, increment, argument, projected = projection_residual(dual, offset, matrix, widths)
            error = np.max(np.linalg.norm(residual, axis=1))
            norms = np.linalg.norm(argument, axis=1)
            active = norms > widths
            history.append({"iteration": iteration, "residual": float(error), "active_disks": int(active.sum())})
            if error < 1e-12:
                converged = True
                break
            derivative = np.broadcast_to(np.eye(2), (16, 2, 2)).copy()
            direction = argument[active] / norms[active, None]
            derivative[active] = (widths[active] / norms[active])[:, None, None] * (
                np.eye(2) - direction[:, :, None] * direction[:, None, :]
            )
            jacobian = (np.eye(2)[None] - derivative)[:, None] * matrix[:, :, None, None]
            jacobian[np.arange(16), np.arange(16)] += derivative
            jacobian = jacobian.transpose(0, 2, 1, 3).reshape(32, 32)
            history[-1]["condition"] = float(np.linalg.cond(jacobian))
            step = np.linalg.solve(jacobian, residual.reshape(32)).reshape(16, 2)
            fraction = 1.0
            accepted = False
            for backtrack in range(32):
                trial = dual + fraction * step
                trial_residual = projection_residual(trial, offset, matrix, widths)[0]
                if np.linalg.norm(trial_residual) <= (1 - 1e-4 * fraction) * np.linalg.norm(residual):
                    dual = trial
                    accepted = True
                    break
                fraction *= 0.5
            history[-1]["fraction"] = fraction
            if not accepted:
                break
        residual, increment, argument, projected = projection_residual(dual, offset, matrix, widths)
        line = np.zeros((17, 2))
        line[:16] = np.cumsum(projected[::-1], axis=0)[::-1]
        row = {
            "slot": int(selected[case]),
            "converged": converged,
            "target_magnitude": float(np.linalg.norm(force)),
            "force_residual": float(np.linalg.norm(weights @ line[:16] - force)),
            "cone_bound": float(np.max(np.linalg.norm(projected, axis=1) / widths)),
            "reference_line_difference": float(np.max(np.linalg.norm(line - reference[case], axis=1))),
            "maximum_dual": float(np.linalg.norm(dual, axis=1).max()),
            "history": history,
        }
        rows.append(row)
        solutions.append(line)
        write_json(args.output / "progress.json", {"rows": rows})
    np.savez_compressed(args.output / "lines.npz", selected=selected, line=np.stack(solutions), reference=reference)
    write_json(
        args.output / "result.json",
        {"passed": True, "rows": rows, "all_converged": all(row["converged"] for row in rows)},
    )


if __name__ == "__main__":
    main()
