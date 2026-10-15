"""Measure initial-motion sensitivity at identical solved closures and physical controls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-audit", type=Path, required=True)
    parser.add_argument("--motion-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    roots = [args.zero_audit, args.motion_audit]
    reports = []
    for root in roots:
        closed_source(root)
        assert int((root / "returncode.txt").read_text()) == 0
        reports.append(json.loads((root / "results/summary.json").read_text()))
    assert all(len(report["cases"]) == 2 for report in reports)
    cases = []
    for case_index, (zero, motion) in enumerate(zip(*(report["cases"] for report in reports))):
        assert zero["rigid"] == motion["rigid"]
        assert zero["steps_checked"] == motion["steps_checked"]
        runs = [Path(case["run"]) for case in [zero, motion]]
        configs = [json.loads((run / "results/configuration.json").read_text()) for run in runs]
        assert not configs[0].get("captured_initial_velocity", False)
        assert configs[1]["captured_initial_velocity"] is True
        for key in ["capture", "dt", "steps", "control_steps", "dhat"]:
            assert configs[0][key] == configs[1][key], key
        assert len({hashlib.sha256(Path(c["mjcf"]).read_bytes()).hexdigest() for c in configs}) == 1
        summaries = [json.loads((run / "results/summary.json").read_text()) for run in runs]
        for key in ["controller_target", "controller_kp", "controller_kv", "controller_caps", "robot_solver_tolerance"]:
            np.testing.assert_array_equal(summaries[0][key], summaries[1][key])
        with np.load(runs[0] / "results/closure_state.npz") as first, np.load(
            runs[1] / "results/closure_state.npz"
        ) as second:
            assert first.files == second.files
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])
        with np.load(roots[0] / f"results/case{case_index:02d}/step0001.npz") as saved:
            vertices, tetrahedra = saved["body_mesh_vertices"], saved["body_mesh_tetrahedra"]
        body = SkfemElasticReference(vertices, tetrahedra, 1e10, 0.3, 2000)

        def tensor_norm(tensor, weights=body.basis.dx):
            return float(np.sqrt(np.sum(weights * np.sum(tensor * tensor, axis=(-2, -1)))))

        rows = []
        for old, new in zip(zero["rows"], motion["rows"]):
            assert old["step"] == new["step"] and old["time_s"] == new["time_s"]
            step = old["step"]
            fields = []
            for root in roots:
                with np.load(root / f"results/case{case_index:02d}/step{step:04d}.npz") as saved:
                    np.testing.assert_array_equal(saved["body_mesh_vertices"], vertices)
                    np.testing.assert_array_equal(saved["body_mesh_tetrahedra"], tetrahedra)
                    fields.append({k: saved[k].copy() for k in saved.files})
            row = {"step": step, "time_s": old["time_s"]}
            for label, key in [
                ("linear_replay", "objective_replay_quadrature_tensor_Pa"),
                ("nonlinear", "objective_nonlinear_quadrature_tensor_Pa"),
            ]:
                if key not in fields[0]:
                    continue
                absolute = tensor_norm(fields[0][key] - fields[1][key])
                row[label + "_tensor_absolute_volume_L2_Pa_m32"] = absolute
                row[label + "_tensor_relative_volume_L2"] = absolute / tensor_norm(fields[1][key])
            for key in ["raw_replayed_peak_Pa", "replayed_energy_J", "nonlinear_raw_peak_lower_Pa"]:
                if old[key] is not None:
                    row[key + "_absolute_change"] = abs(old[key] - new[key])
                    row[key + "_relative_change"] = abs(old[key] / new[key] - 1)
            for key in [
                "body_linear_velocity_world_m_s",
                "body_angular_velocity_world_rad_s",
                "body_angular_momentum_about_com_world_Nms",
                "body_contact_moment_world_Nm",
            ]:
                row[key + "_absolute_change"] = float(np.linalg.norm(np.array(old[key]) - new[key]))
            for label, section in [("force_N", slice(0, 3)), ("moment_Nm", slice(3, 6))]:
                old_wrench, new_wrench = [field["patch_wrenches_material_N_Nm"][:, section] for field in fields]
                absolute = float(np.linalg.norm(old_wrench - new_wrench))
                row["patch_" + label + "_absolute_change"] = absolute
                denominator = float(np.linalg.norm(new_wrench))
                row["patch_" + label + "_relative_change"] = absolute / denominator if denominator else None
            rotations = [field["rotation_world_from_material"] for field in fields]
            row["body_orientation_absolute_change_rad"] = float(
                np.linalg.norm(Rotation.from_matrix(rotations[0] @ rotations[1].T).as_rotvec())
            )
            rows.append(row)
        cases.append({"rigid": zero["rigid"], "runs": [str(run) for run in runs], "rows": rows})
    report = {
        "scope": __doc__,
        "zero_audit": str(args.zero_audit),
        "motion_audit": str(args.motion_audit),
        "cases": cases,
        "goal_completion": False,
        "limitations": "This isolates declared initial-velocity sensitivity at array-identical solved closures, fixed physical controls and one coarse spatial/contact setting. It does not match original approach/lift/support history or qualify physical spatial/contact accuracy. Original body/joint/fresh-step gates are established by the input independent audits; comparisons do not relax them.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "steps_per_case": [len(case["rows"]) for case in cases]}))


if __name__ == "__main__":
    main()
