"""Replay captured radial-memory inputs and staged history at unchanged tolerances."""

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz_friction import (
    StressFrictionHistory,
    kernel_friction_cooperative,
    kernel_friction_info,
    kernel_friction_serial,
)

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array
from .hertz_history_cone import ConeFrictionHistory


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--capture", type=Path, required=True)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--penalties", type=float, nargs="+", default=[0.1, 0.3, 1, 3, 10, 30])
    cli.add_argument("--captured-penalty", type=float, default=0.1)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "capture": file_id(args.capture),
            "penalties": args.penalties,
            "captured_penalty": args.captured_penalty,
            "scope": "Replay the exact staged prior without rerunning history interpolation. Penalty changes alter only the numerical solver, with the original iteration limit and all acceptance gates. Independent SLSQP solves the same radial approximation; no physical accuracy claim.",
        },
    )
    with np.load(args.capture) as archive:
        saved = {key: archive[key] for key in archive.files}
    selected = np.flatnonzero((saved["input_normal"] > 0) & (saved["input_friction"] > 0))
    assert len(selected) and not saved["state_prior_is_lost"][selected].any()
    references = []
    for slot in selected:
        reference = ConeFrictionHistory(16)
        reference.line = saved["state_prior"][slot].copy()
        reference.radius = saved["input_radius"][slot]
        reference.scale = 0.75 * saved["input_friction"][slot] * saved["input_normal"][slot] / reference.radius
        reference.update(
            reference.radius,
            saved["input_normal"][slot],
            saved["input_friction"][slot],
            saved["input_force"][slot],
        )
        references.append(reference)
    expected = np.stack([reference.line for reference in references])
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    native = StressFrictionHistory(len(selected))
    for field in fields(native.input):
        vars(native.input)[field.name].from_numpy(saved["input_" + field.name][selected].copy())
    native.state.prior.from_numpy(saved["state_prior"][selected].transpose(1, 0, 2).copy())
    native.state.prior_is_lost.from_numpy(saved["state_prior_is_lost"][selected].copy())
    solve = kernel_friction_cooperative if args.backend == "gpu" else kernel_friction_serial
    rows, lines = [], []
    for penalty in args.penalties:
        kernel_friction_info(penalty, native.info)
        solve(native.maximum_iterations, penalty, native.input, native.state, native.info)
        line, status = array(native.state.line), array(native.state.status)
        errors = np.max(np.linalg.norm(line - expected, axis=2), axis=1)
        assert np.all(errors[status == 0] < 1e-5), (penalty, errors, status)
        if np.isclose(penalty, args.captured_penalty):
            np.testing.assert_array_equal(status, saved["state_status"][selected])
        rows.append(
            {
                "penalty": penalty,
                "status": status.tolist(),
                "maximum_line_error": errors.tolist(),
                **{
                    name: array(vars(native.state)[name]).tolist()
                    for name in ["iterations", "residual", "dual_residual", "line_residual", "traction_bound"]
                },
            }
        )
        lines.append(line)
        write_json(args.output / "progress.json", {"rows": rows})
    np.savez_compressed(args.output / "lines.npz", selected=selected, reference=expected, native=np.stack(lines))
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "selected": selected.tolist(),
            "target_magnitude": (
                np.linalg.norm(saved["input_force"][selected], axis=1)
                / (saved["input_friction"][selected] * saved["input_normal"][selected])
            ).tolist(),
            "reference_solvers": [reference.solver for reference in references],
            "rows": rows,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
