"""Check native radial memory, slot permutations and physical-state tags."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz_friction import StressFrictionHistory
from genesis.utils.array_class import DataKind, iter_data
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json
from .hertz_history_cone import ConeFrictionHistory


def array(value):
    return qd_to_numpy(value, transpose=True, copy=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"line": 1e-5, "traction": 1e-4, "force": 1e-9, "cone": 1 + 1e-10}
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"), "targets": targets},
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    native = StressFrictionHistory(4)
    kinds = {item.name: item.kind for item in iter_data(native.state, "state")}
    assert all(
        kinds["state." + name] == DataKind.STATE for name in ["line", "previous_radius", "previous_scale", "is_lost"]
    )
    assert kinds["state.prior"] == kinds["state.coefficients"] == DataKind.SCRATCH
    references = [ConeFrictionHistory(16) for _ in range(4)]
    waypoints = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.65, 0.0],
            [0.65, 0.2, 0.3],
            [1.2, -0.35, 0.25],
            [0.85, -0.25, -0.3],
            [1.05, 0.4, -0.2],
            [1.0, 0.0, 0.0],
        ]
    )
    trajectory = np.concatenate(
        [
            first + (second - first) * np.arange(1, 9)[:, None] / 8
            for first, second in zip(waypoints[:-1], waypoints[1:])
        ]
    )
    slots = np.arange(4)
    radial = np.linspace(0, 1, 257)
    nodes = np.linspace(0, 1, 17)
    widths = np.diff(nodes**2)
    weights = np.sqrt(np.maximum(nodes[None, 1:] ** 2 - radial[:, None] ** 2, 0)) - np.sqrt(
        np.maximum(nodes[None, :-1] ** 2 - radial[:, None] ** 2, 0)
    )
    rows, retained = [], []
    for step, values in enumerate(trajectory):
        previous = slots.copy()
        if step % 3 == 2:
            slots = np.roll(slots, 1)
        previous_indices = np.array([np.flatnonzero(previous == identity)[0] for identity in slots])
        normal = np.array([values[0], values[0], 0.0, 0.8 + 0.1 * np.sin(step * 0.37)])
        radius = normal ** (1 / 3)
        friction = np.array([0.4, 0.4, 0.4, 0.3])
        force = np.zeros((4, 2))
        force[0] = 0.4 * values[1:]
        angle = 0.13 * step
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        force[1] = rotation @ force[0]
        force[3] = friction[3] * normal[3] * 0.4 * np.array([np.cos(step * 0.23), np.sin(step * 0.23)])
        transport = np.array([0.0, 0.13 if step else 0.0, 0.0, 0.0])
        is_reset = np.array([False, step == 17, False, False])
        native.input.radius.from_numpy(radius[slots])
        native.input.normal.from_numpy(normal[slots])
        native.input.friction.from_numpy(friction[slots])
        native.input.force.from_numpy(force[slots])
        native.input.angle.from_numpy(transport[slots])
        native.input.previous_index.from_numpy(previous_indices.astype(gs.np_int))
        native.input.is_reset.from_numpy(is_reset[slots])
        native.update()
        line = array(native.state.line)
        status = array(native.state.status)
        assert (status[slots != 2] == 0).all() and (status[slots == 2] == 1).all()
        assert not array(native.state.is_lost).any()
        assert (array(native.state.residual) < targets["force"]).all()
        assert (array(native.state.traction_bound) <= targets["cone"]).all()
        maximum_line, maximum_traction = 0.0, 0.0
        for slot, identity in enumerate(slots):
            references[identity].update(
                radius[identity],
                normal[identity],
                friction[identity],
                force[identity],
                transport[identity],
                reset=bool(is_reset[identity]),
            )
            expected = references[identity].line
            maximum_line = max(maximum_line, float(np.max(np.linalg.norm(line[slot] - expected, axis=1))))
            traction = weights @ ((line[slot, :-1] - line[slot, 1:]) / widths[:, None])
            maximum_traction = max(
                maximum_traction,
                float(np.max(np.linalg.norm(traction - references[identity].traction(radial), axis=1))),
            )
        assert maximum_line < targets["line"] and maximum_traction < targets["traction"], (
            step,
            maximum_line,
            maximum_traction,
        )
        rows.append({"step": step, "maximum_line": maximum_line, "maximum_traction": maximum_traction})
        retained.append(line.copy())
        write_json(args.output / "progress.json", {"rows": rows})
    np.savez_compressed(args.output / "history.npz", line=np.stack(retained), final_slots=slots)
    # A failed projection must retain its physical loss flag until release.
    native.input.previous_index.from_numpy(np.arange(4, dtype=gs.np_int))
    native.input.is_reset.fill(False)
    native.input.angle.fill(0)
    native.input.normal.fill(1)
    native.input.radius.fill(1)
    native.input.friction.fill(0.4)
    bad = np.tile([0.1, 0.0], (4, 1))
    bad[0] = [0.8, 0.0]
    native.input.force.from_numpy(bad)
    native.update()
    assert array(native.state.status)[0] == 3 and array(native.state.is_lost)[0]
    np.testing.assert_array_equal(array(native.state.line)[0], 0)
    bad[0] = [0.1, 0.0]
    native.input.force.from_numpy(bad)
    native.update()
    assert array(native.state.status)[0] == 5
    native.input.normal.fill(0)
    native.input.radius.fill(0)
    native.input.force.fill(0)
    native.update()
    assert (array(native.state.status) == 1).all() and not array(native.state.is_lost).any()
    native.input.normal.fill(1)
    native.input.radius.fill(1)
    native.input.force.from_numpy(np.tile([0.1, 0.0], (4, 1)))
    native.update()
    assert (array(native.state.status) == 0).all()
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "targets": targets,
            "slot_permutation_and_selected_reset_passed": True,
            "physical_state_and_scratch_classification_passed": True,
            "loss_latch_release_and_recovery_passed": True,
            "scope": "Numerical fixed16 native friction history, covariance, changing normal/vector inputs, slot remapping, reset and failure/release controls. Actual rigid lifecycle, physical checkpoint restore and stress recovery are verified separately.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
