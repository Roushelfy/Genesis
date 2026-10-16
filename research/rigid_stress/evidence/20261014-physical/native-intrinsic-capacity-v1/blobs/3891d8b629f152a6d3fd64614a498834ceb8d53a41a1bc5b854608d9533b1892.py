"""Recompute one failed contact state with identical held rows in two origins."""

import argparse
import json
from pathlib import Path

import numpy as np

from .franka_contact_reference import build_from_capture
from .neohookean_reference import NeoHookeanBulk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    world, _ = build_from_capture(args.capture, 2.5e-6, centre_coordinates=False)
    centred, _ = build_from_capture(args.capture, 2.5e-6)
    with np.load(args.failed_run / "results/closure_failure.npz") as failure:
        u, previous, held = (failure[key].copy() for key in ("displacement", "previous_displacement", "boundary_dofs"))
        np.testing.assert_allclose(world.rest, failure["rest_positions"], rtol=0, atol=0)
    states = [
        json.loads(line)
        for line in (args.failed_run / "reference.log").read_text().splitlines()
        if line.startswith("{")
    ]
    step = sum(state["newton_iteration"] == 0 for state in states)
    target = (
        0.04
        + (0.018 - 0.04)
        * step
        / json.loads((args.failed_run / "results/configuration.json").read_text())["control_steps"]
    )
    count = world.body.basis.N
    columns = np.r_[np.setdiff1d(np.arange(count), held), count, count + 1]
    mapping = world.coordinate_map(False)[0][:, columns]
    np.testing.assert_allclose(u[held], 0, rtol=0, atol=0)
    results = {
        "scope": __doc__,
        "held_original_rows": held.tolist(),
        "target_step": step,
        "q_target_m": target,
        "origin_world_m": centred.world_coordinate_origin_m.tolist(),
    }
    world_stable = world.bulk
    for name, system, bulk in (
        ("original_world", world, NeoHookeanBulk(world.body)),
        ("stable_world", world, world_stable),
        ("stable_centred", centred, centred.bulk),
    ):
        system.bulk = bulk
        system.q_target[:] = target
        external = system.gravity_force([0, 0, -9.81])
        g = system.evaluate_updated_friction(u, previous, external, hessian=False)[1]
        residual = float(np.linalg.norm(mapping.T @ g))
        results[name] = {"same_saved_state_residual_N": residual, "jaw_residual_N": (mapping.T @ g)[-2:].tolist()}
    # This assertion prevents a different fixture from being called a same-state replay.
    original = states[-1]["updated_residual_N"]
    assert abs(results["original_world"]["same_saved_state_residual_N"] - original) < 2e-11, results
    (args.output / "same-state-diagnostic.json").write_text(json.dumps(results, indent=2) + "\n")
    u_new, g, _, _, records = centred.solve_updated_friction(
        held,
        np.zeros(len(held)),
        1,
        initial_displacement=u,
        initial_previous_displacement=previous,
        coordinate_map=mapping,
        applied_force=centred.gravity_force([0, 0, -9.81]),
        initial_force=centred.gravity_force([0, 0, -9.81]),
        phase="identical_original_hold_rows_centred",
        failure_path=args.output / "requalification_failure.npz",
    )
    results["solved_centred_residual_N"] = float(np.linalg.norm(mapping.T @ g))
    assert results["solved_centred_residual_N"] <= 1e-9
    results["newton_iterations"] = len(records)
    results["unchanged_tolerance_N"] = 1e-9
    np.savez_compressed(args.output / "requalified_fields.npz", displacement=u_new, original_held_rows=held)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
