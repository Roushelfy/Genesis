"""Assess short-window cost and numerical contrasts at declared keyframes; no physical reference or long-run qualification."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import corotation


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--reactions", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reference-name", default="budget8192")
    parser.add_argument(
        "--no-observer-control",
        action="store_true",
        help="Declare no same-setting sparse/dense control was run; emit no observer-equivalence claim",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan, fields, reaction_receipt = read(args.plan), read(args.fields), read(args.reactions / "result.json")
    bands = plan["provisional_numerical_bands"]
    folders, rejected = {}, []
    for root in args.profiles:
        for group, tasks in enumerate(read(root / "tasks.json")):
            for task in tasks:
                assert task["module"].endswith(".contact_window")
                name = task["name"]
                folder = root / f"group{group}" / name
                receipt_path = folder / "result.json"
                receipt = read(receipt_path) if receipt_path.exists() else None
                if receipt is not None and receipt.get("completed_requested_history"):
                    assert name not in folders
                    folders[name] = folder
                else:
                    log = folder.parent / f"{name}.log"
                    rejected.append(
                        {
                            "name": name,
                            "folder": str(folder),
                            "receipt": receipt,
                            "log": file_id(log) if log.exists() else None,
                            "scope": "Missing/incomplete result is not an admitted short-window run; inspect true terminal status before final interpretation.",
                        }
                    )
    reference = folders[args.reference_name]
    receipts = {name: read(folder / "result.json") for name, folder in folders.items()}
    inputs = {name: read(folder / "inputs.json") for name, folder in folders.items()}
    histories = {
        name: [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        for name, folder in folders.items()
    }
    field_pairs = {Path(row["candidate_folder"]).name: row for row in fields["comparisons"]}
    with np.load(args.reactions / "reactions.npz") as saved:
        reactions = {key: saved[key] for key in saved.files}
    fingers = reaction_receipt["finger_body_ids"]
    reaction_index = {
        Path(row["source"]["path"]).parent.name: index for index, row in enumerate(reaction_receipt["comparisons"])
    }
    with np.load(reference / "rest.npz") as saved:
        rest = {key: saved[key] for key in saved.files}
    weight = float(rest["masses_kg"].sum() * 9.81)
    keys = {
        round(value / receipts[args.reference_name]["config"]["dt"])
        for value in plan["full_spatial_stress_keyframes_s"]
    }
    metrics = {}
    for name, rows in histories.items():
        metrics[name] = {
            round(row["time_s"] / receipts[name]["config"]["dt"]): row["sampled_stress_metrics"]
            for row in rows
            if "sampled_stress_metrics" in row
        }
        assert keys <= metrics[name].keys()
    records = []
    for name, folder in sorted(folders.items()):
        result, rows = receipts[name], histories[name]
        assert result["completed_requested_history"]
        expected_calls = len(rows) if "dense" in name else len(keys)
        assert result["stress_observer_calls"] == expected_calls
        by_tick = {round(row["time_s"] / result["config"]["dt"]): row for row in rows}
        key_results = []
        for key in sorted(keys):
            where = np.flatnonzero(np.abs(reactions["reference_time_s"] - key * result["config"]["dt"]) < 1e-12)
            assert len(where) == 1
            index = where[0]
            force_key = (
                "reference_link_force_world_n"
                if name == args.reference_name
                else f"candidate{reaction_index[name]}_link_force_world_n"
            )
            finger_force_norm = float(np.linalg.norm(reactions[force_key][index, fingers]))
            values, expected = metrics[name][key], metrics[args.reference_name][key]
            net = float(np.linalg.norm(values["force_residual_resultant_n"]) / weight)
            residual = values["force_residual_relative_contact"]
            peak = abs(values["raw_vm_peak_pa"] - expected["raw_vm_peak_pa"]) / max(abs(expected["raw_vm_peak_pa"]), 1)
            common, force_difference, moment_difference = 0.0, 0.0, 0.0
            if name != args.reference_name:
                pair = field_pairs[name]
                row = next(row for row in pair["history"] if abs(row["time_s"] - key * result["config"]["dt"]) < 1e-12)
                common = row["common_tangent"]["absolute_tensor_rms_error_pa"] / max(
                    row["common_tangent"]["reference_tensor_rms_pa"], 1
                )
                candidate = reaction_index[name]
                for kind, floor in (("force_world_n", 1e-3), ("moment_about_com_nm", 1e-7)):
                    expected_vector = reactions[f"reference_link_{kind}"][index, fingers]
                    actual_vector = reactions[f"candidate{candidate}_link_{kind}"][index, fingers]
                    relative = float(
                        np.linalg.norm(actual_vector - expected_vector) / max(np.linalg.norm(expected_vector), floor)
                    )
                    if kind == "force_world_n":
                        force_difference = relative
                    else:
                        moment_difference = relative
            numerical = (
                values["all_fields_finite"]
                and values["minimum_det_F"] > 0
                and net <= bands["keyframe_net_force_residual_over_weight"]
                and residual is not None
                and residual <= bands["keyframe_nodal_residual_relative_to_contact_norm"]
                and common <= bands["common_tangent_relative_contrast"]
                and force_difference <= bands["finger_force_vector_contrast"]
                and moment_difference <= bands["finger_moment_vector_contrast"]
                and peak <= bands["within_keyframe_spatial_peak_contrast"]
            )
            key_results.append(
                {
                    "time_s": key * result["config"]["dt"],
                    "net_residual_over_egg_weight": net,
                    "nodal_force_residual_relative_contact": residual,
                    "common_tangent_contrast_with_floor": common,
                    "finger_force_contrast_with_floor": force_difference,
                    "finger_force_vector_norm_n": finger_force_norm,
                    "finger_moment_contrast_with_floor": moment_difference,
                    "spatial_VM_peak_contrast_with_1Pa_floor": peak,
                    "provisional_keyframe_bands_passed": bool(numerical),
                    "newton": by_tick[key]["newton"],
                    "pcg_max": by_tick[key]["pcg_max"],
                }
            )
        window_rows = [row for row in rows if row["time_s"] > inputs[name]["window_s"][0] + 1e-12]
        distributions = {}
        for label in ("newton", "pcg_total", "pcg_max", "line_search_max", "ccd_maximum_iterations"):
            values = np.array([row[label] for row in window_rows])
            distributions[label] = {
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": int(values.max()),
                "sum": int(values.sum()),
            }
        no_native_failure = all(
            not row["invalid_step"] and row["edge_triangle_intersection_count"] == 0 for row in rows
        )
        records.append(
            {
                "name": name,
                "receipt": file_id(folder / "result.json"),
                "config": result["config"],
                "stress_observer_calls": result["stress_observer_calls"],
                "pcg_cap_steps": result["pcg_cap_steps"],
                "window_intrusive_seconds": result["window_intrusive_seconds"],
                "stage_summary": result["stage_summary"],
                "timing_mode": result.get("timing_mode", "segmented"),
                "field_IO_deferred": result.get("field_IO_deferred", False),
                "elastic_observer_cold_activation_seconds": result.get("elastic_observer_cold_activation_seconds"),
                "newton_aware_gate": result.get("newton_aware_gate"),
                "window_iteration_distributions": distributions,
                "keyframes": key_results,
                "keyframes_with_finger_reaction_above_1mN": sum(
                    row["finger_force_vector_norm_n"] > 1e-3 for row in key_results
                ),
                "provisional_numerical_bands_passed": bool(
                    no_native_failure and all(row["provisional_keyframe_bands_passed"] for row in key_results)
                ),
            }
        )
    invariance = []
    for key in () if args.no_observer_control else sorted(keys):
        tensors, points = [], []
        for name in ("baseline-sparse", "baseline-dense-compute"):
            with np.load(folders[name] / "frames" / f"step{key - 1:06d}.npz") as saved:
                points.append(saved["positions_m"])
                rotation = corotation(rest["authored_vertices_m"], saved["positions_m"], rest["masses_kg"])
                tensors.append(rotation.T @ saved["cauchy_pa"] @ rotation)
        numerator = np.einsum("t,tij,tij->", rest["volumes_m3"], tensors[1] - tensors[0], tensors[1] - tensors[0])
        denominator = np.einsum("t,tij,tij->", rest["volumes_m3"], tensors[0], tensors[0])
        relative = float(np.sqrt(numerator / max(denominator, rest["volumes_m3"].sum())))
        position = float(np.linalg.norm(points[1] - points[0], axis=1).max())
        invariance.append(
            {
                "time_s": key * receipts["baseline-sparse"]["config"]["dt"],
                "maximum_position_difference_m": position,
                "volume_weighted_corotated_3D_tensor_contrast_with_1Pa_floor": relative,
                "strict_observer_equivalence_passed": position <= 1e-9 and relative <= 1e-5,
            }
        )
    eligible = [
        row
        for row in records
        if row["provisional_numerical_bands_passed"]
        and "dense" not in row["name"]
        and row["timing_mode"] == "segmented"
    ]
    next_choice = min(eligible, key=lambda row: row["window_intrusive_seconds"])["name"] if eligible else None
    repeat_groups = {}
    for row in records:
        if row["timing_mode"] == "whole-window":
            signature = json.dumps(row["config"], sort_keys=True)
            repeat_groups.setdefault(signature, []).append(row)
    repeated_timing = []
    for group in repeat_groups.values():
        values = np.array([row["window_intrusive_seconds"] for row in group])
        repeated_timing.append(
            {
                "config": group[0]["config"],
                "repeats": len(group),
                "names": [row["name"] for row in group],
                "window_seconds": values.tolist(),
                "median_window_seconds": float(np.median(values)),
                "minimum_window_seconds": float(values.min()),
                "maximum_window_seconds": float(values.max()),
                "all_provisional_keyframe_bands_passed": all(
                    row["provisional_numerical_bands_passed"] for row in group
                ),
                "scope": "Short-window boundary timing, with declared3-keyframe computation/transfer and deferred compression/IO; not full-grasp throughput.",
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "declared-short-window-tolerance-cost-audit",
            "identity": runtime_identity(),
            "plan": file_id(args.plan),
            "fields": file_id(args.fields),
            "reactions": file_id(args.reactions / "result.json"),
            "numerical_comparison_reference": args.reference_name,
            "egg_weight_n": weight,
            "provisional_bands": bands,
            "configurations": records,
            "rejected_or_incomplete_profiles": rejected,
            "same_config_sparse_dense_invariance": invariance,
            "observer_equivalence_measured": not args.no_observer_control,
            "candidate_for_next_minimally_instrumented_timing_check": next_choice,
            "minimally_instrumented_repeat_summary": repeated_timing,
            "reference_qualified": False,
            "scope": "Numerical contrasts and intrusive stage costs for the entire declared short window with continuous physical history. All3 declared spatial keyframes contribute; no whole-history stress norm or temporal-peak claim. Reaction floors1mN/1e-7Nm and stress floor1Pa are explicit. PCG caps remain visible and are interpreted with final nonlinear force/field measurements. A passing band is relative to an unqualified stricter candidate, not physical accuracy. The named next candidate still requires minimally instrumented timing, capped-coupling qualification and full-grasp sparse controls before a practical primary-cost recommendation.",
        },
    )


if __name__ == "__main__":
    main()
