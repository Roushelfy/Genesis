"""Inspect all saved moving probe fields and matching benchmark audits."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json
from .study_hertz_nearfield import stress_measures


def load(path, bindings):
    bindings.append(file_id(path))
    with np.load(path) as values:
        return {key: values[key] for key in values}


def relative(left, right):
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1))


def inspect_control(root, bindings):
    rows = []
    result = json.loads((root / "result.json").read_text())
    paths = sorted(root.glob("frame*.npz"), key=lambda p: int(p.stem[5:]))
    assert result["passed"] and len(paths) == len(result["rows"])
    for path in paths:
        data = load(path, bindings)
        for env in range(4):
            active = [patch for patch in [env, env + 4] if data["forces_n"][patch] > 0]
            numerical = relative(data["tensor_pa"][active], data["same_load_probe_tensor_pa"][active])
            corners = relative(data["corners_pa"][env], data["same_load_corners_pa"][env])
            higher = relative(data["tensor_pa"][active], data["higher_rule_probe_tensor_pa"][active])
            higher_corners = relative(data["corners_pa"][env], data["higher_rule_corners_pa"][env])
            observed = np.concatenate(
                [data["corners_pa"][env].reshape(-1, 3, 3), data["tensor_pa"][active].reshape(-1, 3, 3)]
            )
            reference = np.concatenate(
                [
                    data["higher_rule_corners_pa"][env].reshape(-1, 3, 3),
                    data["higher_rule_probe_tensor_pa"][active].reshape(-1, 3, 3),
                ]
            )
            tension, vm = stress_measures(observed)
            reference_t, reference_vm = stress_measures(reference)
            np.testing.assert_allclose(
                [data["vm_pa"][env], data["tension_pa"][env]], [vm.max(), tension.max()], rtol=1e-12, atol=1e-8
            )
            vm_change = abs(vm.max() - reference_vm.max()) / max(reference_vm.max(), 1)
            t_change = abs(tension.max() - reference_t.max()) / max(abs(reference_t.max()), 1)
            assert max(numerical, corners) < 1e-7
            assert max(higher, higher_corners) < 1e-3
            assert max(vm_change, t_change) < 1e-2
            if not active:
                assert data["status"][env] == 1
                assert np.array_equal(data["tensor_pa"][[env, env + 4]], np.zeros((2, 89, 3, 3)))
                assert (data["elements"][[env, env + 4]] == -1).all()
            rows.append(
                {
                    "frame": path.stem,
                    "env": env,
                    "numerical_probe": numerical,
                    "numerical_corners": corners,
                    "higher_rule_probe": higher,
                    "higher_rule_corners": higher_corners,
                    "higher_rule_vm_peak": float(vm_change),
                    "higher_rule_tensile_peak": float(t_change),
                }
            )
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--controls", type=Path, required=True)
    cli.add_argument("--prior", type=Path, required=True)
    cli.add_argument("--conditional", type=Path, required=True)
    cli.add_argument("--paired", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bindings, checks, conformance = [], [], []
    for campaign, cases in [
        (args.controls, ["moving", "rotated", "benchmark"]),
        (args.prior, ["moving", "rotated"]),
        (args.conditional, ["moving", "benchmark"]),
    ]:
        for group in [0, 1]:
            for case in cases:
                root = campaign / f"group{group}/{case}"
                checks.append({"root": str(root), "rows": inspect_control(root, bindings)})
        for case in cases:
            for path in sorted((campaign / f"group0/{case}").glob("frame*.npz")):
                cpu, gpu = load(path, bindings), load(campaign / f"group1/{case}" / path.name, bindings)
                assert np.array_equal(cpu["elements"], gpu["elements"])
                field, corner = (
                    relative(cpu["tensor_pa"], gpu["tensor_pa"]),
                    relative(cpu["corners_pa"], gpu["corners_pa"]),
                )
                assert max(field, corner) < 1e-7
                conformance.append(
                    {
                        "kind": "cpu_gpu",
                        "campaign": campaign.name,
                        "case": case,
                        "frame": path.stem,
                        "probe_relative": field,
                        "corner_relative": corner,
                        "same_material_elements": True,
                    }
                )
        for group in [0, 1]:
            for path in sorted((campaign / f"group{group}/moving").glob("frame*.npz")) if "rotated" in cases else []:
                base, rotated = load(path, bindings), load(campaign / f"group{group}/rotated" / path.name, bindings)
                rotation = rotated["frames"][0] @ base["frames"][0].T
                field = relative(rotated["tensor_pa"], rotation @ base["tensor_pa"] @ rotation.T)
                corner = relative(rotated["corners_pa"], rotation @ base["corners_pa"] @ rotation.T)
                assert max(field, corner) < 1e-7
                conformance.append(
                    {
                        "kind": "rotation",
                        "campaign": campaign.name,
                        "group": group,
                        "frame": path.stem,
                        "probe_relative": field,
                        "corner_relative": corner,
                    }
                )
    starts = [
        json.loads((args.paired / f"group0/{case}/start.json").read_text()) for case in ["exact", "far", "q6", "q4"]
    ]
    assert len({s["identity"]["environment"]["SLURM_JOB_ID"] for s in starts}) == 1
    assert len({s["identity"]["hostname"] for s in starts}) == 1
    timings = []
    summaries = {
        case: json.loads((args.paired / f"group0/{case}/result.json").read_text())
        for case in ["exact", "far", "q6", "q4"]
    }
    for batch in [1, 64, 256, 1024]:
        for mode in ["max", "full"]:
            raw = {case: load(args.paired / f"group0/{case}/{mode}-b{batch}-audit.npz", bindings) for case in summaries}
            for case, data in raw.items():
                expected = (np.arange(batch)[:, None] % 4 != 3) & (
                    (np.arange(batch)[:, None] + np.arange(24)) % 13 != 0
                )
                assert np.array_equal(data["loaded"], expected)
                assert (data["status"] <= 1).all() and data["elastic_valid"].all()
                assert np.isfinite(data["vm_pa"]).all() and np.isfinite(data["tensile_pa"]).all()
                assert np.max(data["wrench_error"]) < 1e-4
                assert np.max(np.linalg.norm(data["centers_m"] - data["centers_m"][:, :1], axis=2)) > 0
                other = load(
                    args.paired / f"group0/{case}/{'full' if mode == 'max' else 'max'}-b{batch}-audit.npz", bindings
                )
                for key in ["vm_pa", "tensile_pa"]:
                    np.testing.assert_allclose(data[key], other[key], rtol=1e-7, atol=1e-6)
            for case in ["exact", "q6", "q4"]:
                for key in ["vm_pa", "tensile_pa"]:
                    np.testing.assert_allclose(raw[case][key], raw["far"][key], rtol=1e-3, atol=1e-6)
            values = {
                case: next(row for row in summary["rows"] if row["batch"] == batch and row["mode"] == mode)
                for case, summary in summaries.items()
            }
            timings.append(
                {
                    "batch": batch,
                    "mode": mode,
                    "seconds": {case: row["median_seconds"] for case, row in values.items()},
                    "far_vs_exact_reduction": 1 - values["far"]["median_seconds"] / values["exact"]["median_seconds"],
                    "q4_vs_q8_far_reduction": 1 - values["q4"]["median_seconds"] / values["far"]["median_seconds"],
                    "q6_vs_q8_far_reduction": 1 - values["q6"]["median_seconds"] / values["far"]["median_seconds"],
                    "far_exact_repeat_ranges_separate": max(values["far"]["seconds_per_update"])
                    < min(values["exact"]["seconds_per_update"]),
                    "q4_q8_far_repeat_ranges_separate": max(values["q4"]["seconds_per_update"])
                    < min(values["far"]["seconds_per_update"]),
                    "q6_q8_far_repeat_ranges_separate": max(values["q6"]["seconds_per_update"])
                    < min(values["far"]["seconds_per_update"]),
                    "admitted_total_frames": int(raw["q4"]["status"].size),
                    "admitted_loaded_frames": int(raw["q4"]["loaded"].sum()),
                }
            )
    maxima = {
        key: max(row[key] for group in checks for row in group["rows"])
        for key in checks[0]["rows"][0]
        if key not in ["frame", "env"]
    }
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "bindings": bindings,
            "independent_reference_checks": checks,
            "maximum_control_changes": maxima,
            "complete_cpu_gpu_and_rotation_checks": conformance,
            "same_allocation_timings": timings,
            "automatic_collision_input_or_scene_qualified": False,
            "unknown_pressure_or_general_friction_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
