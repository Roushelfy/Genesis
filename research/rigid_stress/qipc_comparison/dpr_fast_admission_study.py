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
                and isinstance(error6[env], (int, float))
                and isinstance(peak[env], (int, float))
                and error6[env] <= 0.1
                and peak[env] <= 0.05
            ),
        }
        for env, is_loaded in enumerate(loaded.any(axis=1))
    ]


def loaded_max(values, loaded):
    """Return null for empty or nonfinite diagnostics; admission still rejects nonfinite loads."""
    selected = values[loaded]
    return float(selected.max()) if selected.size and np.isfinite(selected).all() else None


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
        "raw_P2_L1_max_loaded": loaded_max(raw_L1, base.loaded),
        "diagnostic_normalized_P2_L1_max_loaded": loaded_max(normalized_L1, base.loaded),
        "base_gradient_max_loaded": loaded_max(base.gradient_norm, base.loaded),
        "checker_gradient_max_loaded": loaded_max(checker.gradient_norm, base.loaded),
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


def assemble_source_nodes(fractions, face_nodes, node_count):
    """Apply the original P2 face-node load map separately to every source."""
    nodes = np.zeros((*fractions.shape[:2], node_count))
    env_indices = np.arange(nodes.shape[0])[:, None, None]
    source_indices = np.arange(nodes.shape[1])[None, :, None]
    for local in range(6):
        np.add.at(nodes, (env_indices, source_indices, face_nodes[None, None, :, local]), fractions[..., local])
    return nodes


def nodal_pair(base, checker, model_identity, output):
    if model_identity != file_id(model_identity["path"]):
        raise ValueError("The original integer face-node mapping artifact changed.")
    with np.load(model_identity["path"], allow_pickle=False) as saved:
        face_nodes = saved["surface_nodes"]
        node_count = len(saved["vertices"])
    shape = base.moments["face_load_fractions"].shape
    if (
        face_nodes.shape != (shape[2], 6)
        or not np.issubdtype(face_nodes.dtype, np.integer)
        or face_nodes.min() < 0
        or face_nodes.max() >= node_count
    ):
        raise ValueError("The immutable original P2 connectivity does not match the face witness.")
    base_fraction = base.moments["face_load_fractions"]
    checker_fraction = checker.moments["face_load_fractions"]
    difference = assemble_source_nodes(base_fraction - checker_fraction, face_nodes, node_count)
    score = np.abs(difference).sum(axis=-1)
    normalized_difference = assemble_source_nodes(
        base_fraction / np.where(base.supported, base.mass, 1.0)[..., None, None]
        - checker_fraction / np.where(checker.supported, checker.mass, 1.0)[..., None, None],
        face_nodes,
        node_count,
    )
    arrays = {
        "loaded": base.loaded,
        "source_nodal_difference": difference,
        "raw_source_nodal_L1": score,
        "diagnostic_normalized_source_nodal_L1": np.abs(normalized_difference).sum(axis=-1),
    }
    artifacts = [model_identity]
    witness_passed = True
    witness_rows = []
    for label, export in (("base", base), ("checker", checker)):
        identity = export.record["fields"]
        if identity != file_id(identity["path"]):
            raise ValueError("An immutable native field/load artifact changed.")
        artifacts.append(identity)
        nodes = assemble_source_nodes(export.moments["face_load_fractions"], face_nodes, node_count)
        reconstructed = np.einsum("bcn,bcd->bnd", nodes, export.contacts["force"])
        with np.load(identity["path"], allow_pickle=False) as saved:
            actual = saved["nodal_force"]
        error = np.linalg.norm((reconstructed - actual).reshape((len(actual), -1)), axis=1)
        error /= np.maximum(np.linalg.norm(actual.reshape((len(actual), -1)), axis=1), 1e-30)
        passed = (error <= 1e-8) | ~export.eligible
        witness_passed &= bool(passed.all())
        arrays.update(
            {
                label + "_source_nodal_fractions": nodes,
                label + "_reconstructed_nodal_force": reconstructed,
                label + "_native_nodal_force": actual,
                label + "_load_reconstruction_relative": error,
            }
        )
        witness_rows.append(
            {
                "rule": export.record["rule"],
                "relative_error_per_environment": [float(value) if np.isfinite(value) else None for value in error],
                "passed": bool(passed.all()),
            }
        )
    admitted_contact = base.supported & checker.supported & np.isfinite(score) & (score <= 1e-3)
    admitted_environment = np.all(admitted_contact | ~base.loaded, axis=1) & base.eligible & checker.eligible
    arrays["contact_admitted"] = admitted_contact
    arrays["environment_admitted"] = admitted_environment
    np.savez_compressed(output, **arrays)
    if not witness_passed:
        write_json(output.with_suffix(".failure.json"), {"witness": witness_rows, "score_arrays": file_id(output)})
        raise RuntimeError("The sourcewise original P2 node witness does not reconstruct the accepted native load.")
    truth = base.record["application_passed_per_environment"]
    admitted = admitted_environment.tolist()
    return {
        "score_arrays": file_id(output),
        "model": model_identity,
        "source_artifacts": artifacts,
        "raw_source_nodal_L1_max_loaded": loaded_max(score, base.loaded),
        "diagnostic_normalized_source_nodal_L1_max_loaded": loaded_max(
            arrays["diagnostic_normalized_source_nodal_L1"], base.loaded
        ),
        "environment_admitted": admitted,
        "sourcewise_native_load_reconstruction": witness_rows,
        "false_accept_count": sum(admit and actual is False for admit, actual in zip(admitted, truth)),
        "false_reject_count": sum(not admit and actual is True for admit, actual in zip(admitted, truth)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=())
    parser.add_argument("--coarse-runs", nargs="+", type=Path, default=())
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--nodal-method", type=Path)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not IS_SLURM:
        parser.error("Scientific postprocessing requires Slurm rtx-mid/image1_26.")
    if len(args.runs) > 12 or len(set(args.runs)) != len(args.runs):
        parser.error("Use at most twelve unique existing frozen-input runs.")
    if not args.runs and not args.coarse_runs:
        parser.error("Provide existing completed normal or coarse frozen-input runs.")
    if args.coarse_runs and (
        args.nodal_method is None or len(args.coarse_runs) > 4 or len(set(args.coarse_runs)) != len(args.coarse_runs)
    ):
        parser.error("Coarse postprocessing requires the frozen nodal method and at most four existing manual runs.")
    if not args.output.resolve().is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use an explicit fresh data-side output directory.")
    method = validate_method(args.method, args.budgets)
    expected = {row["path"]: row for row in method["corpus"]["original_results"]}
    inputs = [file_id(path / "result.json") for path in args.runs]
    if any(expected.get(row["path"]) != row for row in inputs):
        raise ValueError("Every run must match its exact predeclared terminal identity.")
    if args.source_revision != method["source_baseline"]:
        raise ValueError("The declared native source differs from the predeclaration.")
    nodal_method = None
    if args.nodal_method is not None:
        nodal_method = json.loads(args.nodal_method.read_text())
        if (
            nodal_method["schema"] != "genesis.dpr-fast-source-nodal-admission-method.v4"
            or nodal_method["v3_face_method"] != file_id(args.method)
            or nodal_method["scores"]["threshold"] != 1e-3
            or nodal_method["scores"]["pass_operator"] != "<="
            or nodal_method["normal_pairs"] != method["pairs"]
        ):
            raise ValueError("Use the predeclared source-nodal mapping and unchanged1e-3 threshold.")
        control_faces = np.array([[0, 1, 2, 3, 4, 5], [0, 2, 6, 4, 7, 8]])
        control_fractions = np.zeros((1, 2, 2, 6))
        control_fractions[0, 0, 0, 0], control_fractions[0, 0, 1, 0] = 0.3, -0.3
        control_fractions[0, 0, 0, 1], control_fractions[0, 1, 0, 1] = 0.2, -0.2
        control_nodes = assemble_source_nodes(control_fractions, control_faces, 9)
        expected_nodes = np.zeros_like(control_nodes)
        expected_nodes[0, 0, 1], expected_nodes[0, 1, 1] = 0.2, -0.2
        if not np.array_equal(control_nodes, expected_nodes) or not np.all(np.abs(control_nodes).sum(axis=-1) > 1e-3):
            raise RuntimeError("The P2 operator must merge shared nodes and retain separate opposing sources.")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "genesis.dpr-fast-source-nodal-admission-study.v4"
        if nodal_method is not None
        else "genesis.dpr-fast-independent-refit-admission-study.v3",
        "complete": False,
        "passed": False,
        "application_detector_validated": False,
        "goal_complete": False,
        "primary_timing": False,
        "source_revision": args.source_revision,
        "identity": runtime_identity(require_gpu_inventory=False),
        "worker": file_id(Path(__file__)),
        "predeclaration": file_id(args.method),
        "nodal_predeclaration": file_id(args.nodal_method) if nodal_method is not None else None,
        "nodal_operator_software_controls": {
            "shared_node_face_cancellation_exact": True,
            "opposing_sources_cannot_cancel_admission_error": True,
        }
        if nodal_method is not None
        else None,
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
            if nodal_method is not None:
                for row, (base, checker) in zip(pair_rows, PAIRS):
                    row["nodal_v4"] = nodal_pair(
                        exports[base],
                        exports[checker],
                        original["model"],
                        args.output / f"run{run_index:02d}-case{case_index:02d}-{base}-vs-{checker}-nodal.npz",
                    )
                    original_artifacts.extend(row["nodal_v4"]["source_artifacts"])
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
    coarse_inputs = [file_id(path / "result.json") for path in args.coarse_runs]
    result["coarse_original_results"] = coarse_inputs
    for run_index, directory in enumerate(args.coarse_runs):
        original = json.loads((directory / "result.json").read_text())
        if (
            original["complete"] is not True
            or original["passed"] is not True
            or original["source_revision"] != args.source_revision
            or original["coarse_subset_predeclaration"] != nodal_method["coarse_predeclaration"]
            or original["reused_reference_result"] not in nodal_method["coarse_reference_results"]
            or original["reused_model_arrays_exact"] is not True
            or original["reused_reference_artifacts_unchanged"] is not True
        ):
            raise ValueError("Coarse results do not match the frozen same-model/reference study.")
        reference_identity = original["reused_reference_result"]
        if reference_identity != file_id(reference_identity["path"]):
            raise ValueError("The reused manual angle8 reference receipt changed.")
        reference = json.loads(Path(reference_identity["path"]).read_text())
        checker = load_rule(next(row for row in reference["rows"][0]["rules"] if row["rule"] == "angle8"))
        original_artifacts.extend((reference_identity, *checker.artifacts))
        if len(original["rows"]) != 1:
            raise ValueError("Each coarse run must retain its one existing manual physical input.")
        case = original["rows"][0]
        pair_rows = []
        for record in case["rules"]:
            if record["rule"] not in ("edge4", "angle2"):
                raise ValueError("Only the two predeclared coarse controls may be added.")
            base = load_rule(record)
            original_artifacts.extend(base.artifacts)
            prefix = args.output / f"coarse{run_index:02d}-{record['rule']}-vs-angle8"
            row = compare_pair(base, checker, prefix.with_suffix(".npz"))
            row["nodal_v4"] = nodal_pair(base, checker, original["model"], prefix.with_suffix(".nodal.npz"))
            original_artifacts.extend(row["nodal_v4"]["source_artifacts"])
            spatial.setdefault(record["rule"], []).extend(legacy_spatial(record, base.loaded))
            pair_rows.append(row)
        if {row["base"] for row in pair_rows} != {"edge4", "angle2"} or len(pair_rows) != 2:
            raise ValueError("Both unique predeclared coarse rows are required.")
        result["rows"].append(
            {
                "name": case["name"],
                "original_result": file_id(directory / "result.json"),
                "physical_inputs": case["physical_inputs"],
                "whole_frame_qualification": case["whole_frame_qualification"],
                "mechanical_scope": case["mechanical_scope"],
                "pairs": pair_rows,
                "coarse_negative_control": True,
            }
        )
        write_json(args.output / "result.json", result)
    if case_count > 12:
        raise ValueError("The predeclared existing-input corpus is bounded at twelve cases.")
    for rule, rows in spatial.items():
        loaded_rows = [row for row in rows if row["loaded"]]
        error6 = [row["legacy_error6"] for row in loaded_rows]
        is_finite = bool(error6) and all(isinstance(value, (int, float)) and math.isfinite(value) for value in error6)
        mean_error6 = sum(error6) / len(error6) if is_finite else None
        peak_errors = [row["global_VM_peak_relative"] for row in loaded_rows]
        result["original_spatial_budget_by_rule"].append(
            {
                "rule": rule,
                "loaded_environment_frames": len(loaded_rows),
                "legacy_error6_unweighted_mean": mean_error6,
                "legacy_error6_worst": max(error6) if is_finite else None,
                "global_peak_error_worst": max(peak_errors)
                if peak_errors
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in peak_errors)
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
    result["case_count"] = len(result["rows"])
    result["normal_case_count"] = case_count
    result["coarse_case_count"] = len(args.coarse_runs)
    result["false_accept_count"] = sum(row["false_accept_count"] for row in pair_rows)
    result["false_reject_count"] = sum(row["false_reject_count"] for row in pair_rows)
    if nodal_method is not None:
        result["nodal_v4_false_accept_count"] = sum(row["nodal_v4"]["false_accept_count"] for row in pair_rows)
        result["nodal_v4_false_reject_count"] = sum(row["nodal_v4"]["false_reject_count"] for row in pair_rows)
    result["converged_bad_rule_detection_exercised"] = any(
        row["eligible_pair_application_inaccurate_count"] for row in pair_rows
    )
    result["original_results_unchanged"] = inputs == [file_id(path / "result.json") for path in args.runs]
    result["original_artifacts_unchanged"] = original_artifacts == [file_id(row["path"]) for row in original_artifacts]
    result["coarse_original_results_unchanged"] = coarse_inputs == [
        file_id(path / "result.json") for path in args.coarse_runs
    ]
    result["complete"] = True
    result["passed"] = (
        result["original_results_unchanged"]
        and result["original_artifacts_unchanged"]
        and result["coarse_original_results_unchanged"]
    )
    write_json(args.output / "result.json", result)
    if not result["passed"]:
        raise RuntimeError("An immutable source artifact changed during postprocessing.")


if __name__ == "__main__":
    main()
