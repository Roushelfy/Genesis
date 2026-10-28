"""Compare independently fitted pressure-rule loads from immutable exports.

Only Slurm rtx-mid/image1_26 imports NumPy. This study performs no pressure,
elastic or rigid solve and never rewrites its source artifacts.
"""

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

from .artifacts import file_id, runtime_identity, write_json

IS_SLURM = (
    bool(os.environ.get("SLURM_JOB_ID"))
    and os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid"
    and os.environ.get("GENESIS_IMAGE_VER") == "1_26"
)
PAIRS = (("angle16", "angle8"), ("angle64", "angle16"), ("angle128", "angle64"))
PHYSICAL_FIELDS = (
    "valid",
    "position",
    "force",
    "normal",
    "friction",
    "radius",
    "radius_status",
    "center",
    "anchor_face",
)
if IS_SLURM:
    import numpy as np


@dataclasses.dataclass(frozen=True)
class RuleExport:
    record: dict
    moments: dict
    contacts: dict
    loaded: Any
    supported: Any
    eligible: Any
    mass: Any
    gradient_norm: Any
    artifacts: tuple


def validate_method(path, budgets):
    method = json.loads(path.read_text())
    if method["schema"] != "genesis.dpr-fast-independent-refit-admission-method.v3":
        raise ValueError("Use the frozen independently fitted v3 method.")
    pairs = tuple((row["base"], row["independently_fitted_checker"]) for row in method["pairs"])
    if pairs != PAIRS or method["scores"]["threshold"] != 1e-3 or method["scores"]["pass_operator"] != "<=":
        raise ValueError("The predeclared pairs and unchanged1e-3 gate are required.")
    if method["application_budgets"] != file_id(budgets):
        raise ValueError("The application budget identity differs from the predeclaration.")
    return method


def load_rule(record):
    artifacts = tuple(record[key] for key in ("moment_witness", "contacts"))
    if artifacts != tuple(file_id(row["path"]) for row in artifacts):
        raise ValueError("An exported contact or moment artifact differs from the terminal receipt.")
    with np.load(record["moment_witness"]["path"], allow_pickle=False) as saved:
        moments = {key: np.ascontiguousarray(saved[key]) for key in saved.files}
    with np.load(record["contacts"]["path"], allow_pickle=False) as saved:
        contacts = {key: saved[key] for key in saved.files}
    force_norm = np.linalg.norm(contacts["force"], axis=-1)
    loaded = contacts["valid"] & (force_norm > 1e-30)
    physical_finite = np.ones_like(contacts["valid"])
    for key in ("position", "force", "normal", "center"):
        physical_finite &= np.isfinite(contacts[key]).all(axis=-1)
    physical_finite &= np.isfinite(contacts["friction"]) & np.isfinite(contacts["radius"])
    physical_valid = np.all(physical_finite | ~contacts["valid"], axis=1)
    physical_valid &= np.all((contacts["radius"] > 0) | ~loaded, axis=1)
    first = moments["face_gradient"].sum(axis=2)
    mass = first[..., 0]
    supported = np.isfinite(mass) & (mass > 0)
    supported &= np.isfinite(moments["face_load_fractions"]).all(axis=(-2, -1))
    supported &= np.isfinite(moments["face_gradient"]).all(axis=(-2, -1))
    supported &= moments["face_statistics"][..., 6].sum(axis=2) == 0
    supported &= contacts["status"] == 0
    supported &= contacts["force_error"] / np.maximum(force_norm, 1e-30) <= 1e-8
    supported &= contacts["moment_error"] / np.maximum(force_norm * contacts["radius"], 1e-30) <= 1e-8
    eligible = np.array(record["source_and_elastic_eligible_per_environment"])
    eligible &= physical_valid & np.all(supported | ~loaded, axis=1)
    eligible &= np.array(record["K_residual"]) <= record["K_residual_budget"]
    gradient_norm = np.linalg.norm(first - np.array([1.0, 0.0, 0.0]), axis=-1)
    return RuleExport(record, moments, contacts, loaded, supported, eligible, mass, gradient_norm, artifacts)


def legacy_spatial(record, loaded):
    """Keep the stored-six spatial budget distinct from the historical volume parity gate."""
    error6 = record["legacy_errors"]["stress_tensor"]
    peak = record["legacy_errors"]["peak"]
    return [
        {
            "environment": env,
            "loaded": bool(is_loaded),
            "legacy_error6": error6[env],
            "global_VM_peak_relative": peak[env],
            "source_and_elastic_eligible": record["source_and_elastic_eligible_per_environment"][env],
            "reference_evaluable": record["truth_evaluable_per_environment"][env],
            "old_spatial_worst_and_peak_passed": bool(
                record["source_and_elastic_eligible_per_environment"][env]
                and record["truth_evaluable_per_environment"][env]
                and error6[env] is not None
                and peak[env] is not None
                and error6[env] <= 0.1
                and peak[env] <= 0.05
            ),
        }
        for env, is_loaded in enumerate(loaded.any(axis=1))
    ]


def compare_pair(base, checker, output):
    if any(
        base.contacts[key].shape != checker.contacts[key].shape
        or base.contacts[key].dtype != checker.contacts[key].dtype
        or base.contacts[key].tobytes() != checker.contacts[key].tobytes()
        for key in PHYSICAL_FIELDS
    ):
        raise ValueError("Independent fits do not have exactly the same frozen physical sources.")
    if base.moments["face_load_fractions"].shape != checker.moments["face_load_fractions"].shape:
        raise ValueError("The original face/P2 mappings differ between pressure rules.")
    raw_L1 = np.abs(base.moments["face_load_fractions"] - checker.moments["face_load_fractions"]).sum(axis=(-2, -1))
    base_mass = np.where(base.supported, base.mass, 1.0)
    check_mass = np.where(checker.supported, checker.mass, 1.0)
    normalized_L1 = np.abs(
        base.moments["face_load_fractions"] / base_mass[..., None, None]
        - checker.moments["face_load_fractions"] / check_mass[..., None, None]
    ).sum(axis=(-2, -1))
    contact_admitted = base.supported & checker.supported & (raw_L1 <= 1e-3)
    environment_admitted = np.all(contact_admitted | ~base.loaded, axis=1) & base.eligible & checker.eligible
    np.savez_compressed(
        output,
        loaded=base.loaded,
        base_eligible=base.eligible,
        checker_eligible=checker.eligible,
        base_total_pressure_mass=base.mass,
        checker_total_pressure_mass=checker.mass,
        base_gradient_norm=base.gradient_norm,
        checker_gradient_norm=checker.gradient_norm,
        raw_independent_fit_P2_L1=raw_L1,
        diagnostic_mass_normalized_independent_fit_P2_L1=normalized_L1,
        contact_admitted=contact_admitted,
        environment_admitted=environment_admitted,
    )
    truth = base.record["application_passed_per_environment"]
    admitted = environment_admitted.tolist()
    pair_eligible = base.eligible & checker.eligible
    return {
        "base": base.record["rule"],
        "independently_fitted_checker": checker.record["rule"],
        "base_source_artifacts": base.artifacts,
        "checker_source_artifacts": checker.artifacts,
        "physical_inputs_byte_exact_between_rules": True,
        "score_arrays": file_id(output),
        "loaded_contacts": int(base.loaded.sum()),
        "raw_P2_L1_max_loaded": float(raw_L1[base.loaded].max()) if base.loaded.any() else None,
        "diagnostic_normalized_P2_L1_max_loaded": float(normalized_L1[base.loaded].max())
        if base.loaded.any()
        else None,
        "base_gradient_max_loaded": float(base.gradient_norm[base.loaded].max()) if base.loaded.any() else None,
        "checker_gradient_max_loaded": float(checker.gradient_norm[base.loaded].max()) if base.loaded.any() else None,
        "base_eligible_per_environment": base.eligible.tolist(),
        "checker_eligible_per_environment": checker.eligible.tolist(),
        "pair_eligible_per_environment": pair_eligible.tolist(),
        "empirical_environment_admitted": admitted,
        "base_application_truth_per_environment": truth,
        "base_legacy_volume_parity_passed": base.record["legacy_passed_per_environment"],
        "checker_legacy_volume_parity_passed": checker.record["legacy_passed_per_environment"],
        "base_original_spatial_metrics": legacy_spatial(base.record, base.loaded),
        "checker_original_spatial_metrics": legacy_spatial(checker.record, checker.loaded),
        "false_accept_count": sum(admit and actual is False for admit, actual in zip(admitted, truth)),
        "false_reject_count": sum(not admit and actual is True for admit, actual in zip(admitted, truth)),
        "eligible_base_application_inaccurate_count": sum(
            eligible and actual is False for eligible, actual in zip(base.eligible.tolist(), truth)
        ),
        "eligible_pair_application_inaccurate_count": sum(
            eligible and actual is False for eligible, actual in zip(pair_eligible.tolist(), truth)
        ),
        "truth_unresolved_count": sum(actual is None for actual in truth),
        "existing_same_beta_diagnostics": {
            "base_raw_v1_environment_admitted": base.record["empirical_environment_admitted"],
            "base_normalized_v2_environment_admitted": base.record.get("mass_normalized_v2_environment_admitted"),
            "checker_raw_v1_environment_admitted": checker.record["empirical_environment_admitted"],
            "checker_normalized_v2_environment_admitted": checker.record.get("mass_normalized_v2_environment_admitted"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not IS_SLURM:
        parser.error("Scientific postprocessing requires Slurm rtx-mid/image1_26.")
    if len(args.runs) > 12 or len(set(args.runs)) != len(args.runs):
        parser.error("Use at most twelve unique existing frozen-input runs.")
    if not args.output.resolve().is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use an explicit fresh data-side output directory.")
    method = validate_method(args.method, args.budgets)
    expected = {row["path"]: row for row in method["corpus"]["original_results"]}
    inputs = [file_id(path / "result.json") for path in args.runs]
    if any(expected.get(row["path"]) != row for row in inputs):
        raise ValueError("Every run must match its exact predeclared terminal identity.")
    if args.source_revision != method["source_baseline"]:
        raise ValueError("The declared native source differs from the predeclaration.")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "genesis.dpr-fast-independent-refit-admission-study.v3",
        "complete": False,
        "passed": False,
        "application_detector_validated": False,
        "goal_complete": False,
        "primary_timing": False,
        "source_revision": args.source_revision,
        "identity": runtime_identity(require_gpu_inventory=False),
        "worker": file_id(Path(__file__)),
        "predeclaration": file_id(args.method),
        "budgets": file_id(args.budgets),
        "original_results": inputs,
        "scope": "Independent fitted-rule exported moments only. No pressure/elastic solve, physics replay, original artifact rewrite, online q128 screen or continuum certification.",
        "rows": [],
        "original_spatial_budget_by_rule": [],
    }
    write_json(args.output / "result.json", result)
    spatial = {}
    original_artifacts = []
    case_count = 0
    for run_index, directory in enumerate(args.runs):
        original = json.loads((directory / "result.json").read_text())
        if original["complete"] is not True or original["passed"] is not True:
            raise ValueError("Only trustworthy completed frozen-rule runs can be postprocessed.")
        if original["source_revision"] != args.source_revision or original["input_files_unchanged"] is not True:
            raise ValueError("A native source or original-input identity differs.")
        for case_index, case in enumerate(original["rows"]):
            case_count += 1
            records = {record["rule"]: record for record in case["rules"]}
            exports = {rule: load_rule(records[rule]) for rule in ("angle8", "angle16", "angle64", "angle128")}
            for rule, export in exports.items():
                original_artifacts.extend(export.artifacts)
            legacy_rows = []
            for rule, record in records.items():
                values = legacy_spatial(record, exports["angle16"].loaded)
                spatial.setdefault(rule, []).extend(values)
                legacy_rows.append(
                    {
                        "rule": rule,
                        "original_volume_parity_passed": record["legacy_passed_per_environment"],
                        "original_spatial_metrics": values,
                    }
                )
            pair_rows = [
                compare_pair(
                    exports[base],
                    exports[checker],
                    args.output / f"run{run_index:02d}-case{case_index:02d}-{base}-vs-{checker}.npz",
                )
                for base, checker in PAIRS
            ]
            result["rows"].append(
                {
                    "name": case["name"],
                    "original_result": file_id(directory / "result.json"),
                    "physical_inputs": case["physical_inputs"],
                    "whole_frame_qualification": case["whole_frame_qualification"],
                    "mechanical_scope": case["mechanical_scope"],
                    "pairs": pair_rows,
                    "separate_legacy_budget_comparisons": legacy_rows,
                }
            )
            write_json(args.output / "result.json", result)
    if case_count > 12:
        raise ValueError("The predeclared existing-input corpus is bounded at twelve cases.")
    for rule, rows in spatial.items():
        loaded_rows = [row for row in rows if row["loaded"]]
        error6 = [row["legacy_error6"] for row in loaded_rows]
        is_finite = bool(error6) and all(value is not None and math.isfinite(value) for value in error6)
        mean_error6 = sum(error6) / len(error6) if is_finite else None
        peak_errors = [row["global_VM_peak_relative"] for row in loaded_rows]
        result["original_spatial_budget_by_rule"].append(
            {
                "rule": rule,
                "loaded_environment_frames": len(loaded_rows),
                "legacy_error6_unweighted_mean": mean_error6,
                "legacy_error6_worst": max(error6) if is_finite else None,
                "global_peak_error_worst": max(peak_errors)
                if peak_errors and all(value is not None and math.isfinite(value) for value in peak_errors)
                else None,
                "mean_5percent_passed": mean_error6 is not None and mean_error6 <= 0.05,
                "all_frame_10percent_and_peak5percent_passed": bool(loaded_rows)
                and all(row["old_spatial_worst_and_peak_passed"] for row in loaded_rows),
                "passed": mean_error6 is not None
                and mean_error6 <= 0.05
                and all(row["old_spatial_worst_and_peak_passed"] for row in loaded_rows),
            }
        )
    pair_rows = [row for case in result["rows"] for row in case["pairs"]]
    result["pair_count"] = len(pair_rows)
    result["case_count"] = case_count
    result["false_accept_count"] = sum(row["false_accept_count"] for row in pair_rows)
    result["false_reject_count"] = sum(row["false_reject_count"] for row in pair_rows)
    result["converged_bad_rule_detection_exercised"] = any(
        row["eligible_pair_application_inaccurate_count"] for row in pair_rows
    )
    result["original_results_unchanged"] = inputs == [file_id(path / "result.json") for path in args.runs]
    result["original_artifacts_unchanged"] = original_artifacts == [file_id(row["path"]) for row in original_artifacts]
    result["complete"] = True
    result["passed"] = result["original_results_unchanged"] and result["original_artifacts_unchanged"]
    write_json(args.output / "result.json", result)
    if not result["passed"]:
        raise RuntimeError("An immutable source artifact changed during postprocessing.")


if __name__ == "__main__":
    main()
