"""Compare solved inertial contact at exact common times and fixed physical laws."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .audit_objective_frame import finite_frame
from .franka_contact_reference import build_from_capture
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    frozen = None
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "results/configuration.json").read_text())
        physical = {key: value for key, value in config.items() if key not in ("dt", "steps", "output")}
        if frozen is None:
            frozen = physical
        assert physical == frozen and abs(config["dt"] * config["steps"] - 0.1) < 1e-14
        report = json.loads((run / "results/summary.json").read_text())
        assert report["friction_mollifier_velocity_m_s"] == 4e-6
        assert len(report["steps"]) == config["steps"]
        assert max(r["complete_dynamic_residual_N"] for r in report["steps"]) <= 1e-9
        assert max(r["complete_body_dynamic_residual_N"] for r in report["steps"]) <= 1e-9
        cases.append((config["dt"], run, config))
    cases.sort(reverse=True)
    coarse_dt = cases[0][0]
    original = json.loads((Path(frozen["closure_run"]) / "results/configuration.json").read_text())
    system, _ = build_from_capture(Path(original["capture"]), original["dhat"])
    count = system.body.basis.N
    time_values = []
    for dt, run, config in cases:
        rows = []
        for time_index in range(1, round(0.1 / coarse_dt) + 1):
            time = time_index * coarse_dt
            index = round(time / dt)
            assert abs(index * dt - time) < 1e-14
            with np.load(run / f"results/step{index:04d}.npz") as saved:
                u, v, f = (saved[key].copy() for key in ("displacement", "velocity", "contact_force_N"))
            vector = np.empty(count)
            vector[system.parts[0].permutation] = u[:count]
            fitted, translation, local = finite_frame(system.body, vector)
            tensor, vm, upper = system.bulk.stress_bounds(vector)
            body_energy = system.bulk.evaluate(vector, False)[0]
            material_v = np.empty(count)
            material_v[system.parts[0].permutation] = v[:count]
            kinetic_energy = 0.5 * material_v @ (system.body.mass @ material_v)
            for node, axis, mass in zip(system.finger_com_nodes, system.axes, system.masses, strict=True):
                kinetic_energy += 0.5 * (mass + 0.1) * (v[3 * node : 3 * node + 3] @ axis) ** 2
            rows.append(
                {
                    "time_s": time,
                    "displacement": u,
                    "velocity": v,
                    "force": f,
                    "tensor": tensor,
                    "peak_lower_Pa": float(vm.max()),
                    "peak_upper_Pa": float(upper.max()),
                    "rotation": fitted,
                    "translation": translation,
                    "objective_displacement": local,
                    "body_elastic_energy_J": float(body_energy),
                    "total_body_and_jaw_kinetic_energy_J": float(kinetic_energy),
                }
            )
        time_values.append((dt, run, rows))
    comparisons = []
    for old, new in pairwise(time_values):
        record = {"previous_dt_s": old[0], "current_dt_s": new[0], "common_physical_samples": len(new[2])}
        for key in (
            "displacement",
            "velocity",
            "force",
            "tensor",
            "objective_displacement",
            "body_elastic_energy_J",
            "total_body_and_jaw_kinetic_energy_J",
        ):
            a = np.stack([row[key] for row in old[2]])
            b = np.stack([row[key] for row in new[2]])
            record[key + "_common_time_l2_relative_change"] = float(np.linalg.norm(b - a) / np.linalg.norm(b))
            record[key + "_final_relative_change"] = float(np.linalg.norm(b[-1] - a[-1]) / np.linalg.norm(b[-1]))
        record["raw_corner_peak_final_relative_change"] = (
            abs(new[2][-1]["peak_lower_Pa"] - old[2][-1]["peak_lower_Pa"]) / new[2][-1]["peak_lower_Pa"]
        )
        record["final_common_time_s"] = new[2][-1]["time_s"]
        comparisons.append(record)
    summary = []
    for dt, run, rows in time_values:
        last = rows[-1]
        summary.append(
            {
                "run": str(run),
                "dt_s": dt,
                "final_peak_lower_Pa": last["peak_lower_Pa"],
                "final_peak_upper_Pa": last["peak_upper_Pa"],
                "final_objective_displacement_max_m": float(
                    np.linalg.norm(last["objective_displacement"][system.body.component_dofs], axis=1).max()
                ),
                "final_body_fit_translation_m": last["translation"].tolist(),
                "final_body_fit_rotation": last["rotation"].tolist(),
                "final_body_elastic_energy_J": last["body_elastic_energy_J"],
                "final_total_kinetic_energy_J": last["total_body_and_jaw_kinetic_energy_J"],
            }
        )
    output = {
        "scope": __doc__,
        "comparisons": comparisons,
        "cases": summary,
        "goal_completion": False,
        "limitations": "Three inertial time resolutions retain the same held-row initial state, gravity, material, actual rigid fingers, clippedPD/passive damping, physical friction mollifier velocity and0.1s horizon. All saved steps satisfy their original full mapped/body dynamic-force gates. Common-time fields are unsmoothed original corner tensors; reported L2 values are discrete common-grid diagnostics rather than continuum physical accuracy. No spatial/contact regularization or matching to the rigid weld/full arm is inferred. Raw corner lower peaks and whole-volume nonlinear upper peaks remain separate.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
