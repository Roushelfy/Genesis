"""Check complete periodic physical-state files and their exact endpoint packing."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.run / "start.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    rows = [json.loads(line) for line in (args.run / "steps.jsonl").read_text().splitlines()]
    stride = start["args"]["checkpoint_stride"]
    expected = list(range(result["inherited_steps"] + 1, result["steps"] + 1))
    expected = [step for step in expected if step % stride == 0]
    checks = {"complete_own_history": result["completed"] and result["mechanical_history_passed"],
              "positive_explicit_stride": stride > 0}
    receipts = [file_id(args.run / name) for name in ("start.json", "result.json", "steps.jsonl")]
    files = sorted((args.run / "checkpoints").glob("step*.npz"), key=lambda p: int(p.stem.removeprefix("step")))
    checks["all_expected_physical_steps_saved"] = [int(p.stem.removeprefix("step")) for p in files] == expected
    for path in files:
        receipts.append(file_id(path))
        step = int(path.stem.removeprefix("step"))
        row = rows[step - 1]
        with np.load(path) as state:
            checks[f"step{step}/exact_clock_and_plate"] = (
                int(state["step"]) == row["step"] and int(state["level"]) == row["level"]
                and float(state["time_s"]) == row["time_s"] and float(state["load_n"]) == row["command_force_n"]
                and float(state["compression_coordinate_m"]) == row["compression_coordinate_m"]
                and float(state["plate_speed_m_per_s"]) == row["plate_speed_m_per_s"])
            checks[f"step{step}/complete_finite_fp64_body_state"] = (
                state["displacement_m"].shape == state["velocity_m_per_s"].shape == state["vertices_m"].shape
                and state["displacement_m"].size == start["body_dofs"]
                and all(state[key].dtype == np.float64 and np.isfinite(state[key]).all()
                        for key in ("displacement_m", "velocity_m_per_s", "source_forces_n")))
            force = state["source_forces_n"]
            checks[f"step{step}/recorded_physical_point_force"] = (
                force.shape == state["source_points_m"].shape
                and abs(force[:, 2].sum() - row["lower_reaction_n"] + row["upper_reaction_n"]) <= 1e-12
                and np.max(np.abs(force[:, :2].sum(axis=0) - row["tangential_resultant_n"])) <= 1e-12)
            current = state["vertices_m"] + state["displacement_m"]
            net = state["forces_n"] + state["body_forces_n"] - state["inertial_forces_n"]
            scale = max(row["command_force_n"], 1e-3)
            checks[f"step{step}/full_force_and_current_moment"] = (
                np.linalg.norm(net.sum(axis=0)) <= 1e-7 * scale
                and np.linalg.norm(np.cross(current, net).sum(axis=0)) <= 1e-7 * scale * 0.03)
            if step == result["steps"]:
                endpoint = args.run / "frames" / f"level{row['level']}.npz"
                receipts.append(file_id(endpoint))
                with np.load(endpoint) as full:
                    for key in ("vertices_m", "tetrahedra", "load_n", "displacement_m", "velocity_m_per_s",
                                "source_points_m", "source_forces_n", "projected_area_m2", "forces_n",
                                "body_forces_n", "inertial_forces_n", "young_pa", "poisson"):
                        np.testing.assert_array_equal(state[key], full[key])
                        checks["endpoint_identical/" + key] = True
    passed = bool(checks) and all(checks.values())
    write_json(args.output / "result.json", {
        "identity": runtime_identity(require_gpu_inventory=False), "passed": passed,
        "checks": checks, "saved_steps": expected, "receipts": receipts,
        "scope": "Complete periodic state packing, physical clock/plate/force/current-moment checks and exact final endpoint-array identity. Trajectory equivalence uses the separate full-array/common-field audit. Preparation and restoration of a non-endpoint checkpoint require their own terminal provenance controls.",
        "whole_goal_complete": False,
    })
    if not passed:
        raise SystemExit("Periodic physical-state checks failed")


if __name__ == "__main__":
    main()
