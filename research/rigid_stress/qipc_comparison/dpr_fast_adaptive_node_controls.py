"""Test the shared native source-node operator on saved positive and bad rules.

This is an aggregation primitive test, not a new pressure solve or an end-to-end
bad-field detector. Run only on Slurm rtx-mid/image1_26.
"""

import argparse
import dataclasses
import json
import os
import subprocess
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json

IS_SLURM = (
    bool(os.environ.get("SLURM_JOB_ID"))
    and os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid"
    and os.environ.get("GENESIS_IMAGE_VER") == "1_26"
)

if IS_SLURM:
    import numpy as np
    import quadrants as qd

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.adaptive_pressure import func_adaptive_node_error
    from genesis.engine.solvers.rigid.stress.scatter import StressScatterWorkspace, create_scatter_workspace
    from genesis.utils.array_class import V
    from genesis.utils.misc import qd_to_numpy

    from .dpr_fast_adaptive_replay import checked_identity, finite_json, read_npz

    @dataclasses.dataclass(frozen=True)
    class OperatorControl:
        name: str
        surface_nodes: np.ndarray
        differences: np.ndarray
        expected_scores: np.ndarray
        original_sources: list[tuple[int, int]]
        provenance: dict

    @qd.kernel
    def kernel_node_error(errors: qd.Tensor, surface_nodes: qd.Tensor, workspace: StressScatterWorkspace):
        for i_pair in range(errors.shape[0]):
            errors[i_pair] = func_adaptive_node_error(
                i_pair, workspace.source_heads[i_pair][1], surface_nodes, workspace
            )


def load_control(row, pair, negative):
    physical = read_npz(row["physical_inputs"])
    loaded = physical["valid"] & (np.linalg.norm(physical["force"], axis=-1) > 1e-30)
    source_indices = [tuple(index) for index in np.argwhere(loaded)]
    base = read_npz(pair["base_source_artifacts"][0])["face_load_fractions"]
    checker = read_npz(pair["checker_source_artifacts"][0])["face_load_fractions"]
    model = read_npz(pair["nodal_v4"]["model"])
    face_nodes = model["surface_nodes"]
    if base.shape != checker.shape or base.shape != (*loaded.shape, len(face_nodes), 6):
        raise ValueError("Saved P2 witness/model/source dimensions differ.")
    difference = (base - checker)[loaded]
    expected = np.zeros(len(source_indices))
    for index, values in enumerate(difference):
        nodal = np.zeros(len(model["vertices"]))
        for local in range(6):
            np.add.at(nodal, face_nodes[:, local], values[:, local])
        expected[index] = np.abs(nodal).sum()
    original = read_npz(pair["nodal_v4"]["score_arrays"])["raw_source_nodal_L1"][loaded]
    if not np.all(np.abs(expected - original) <= 1e-12):
        raise ValueError("Independent original-node assembly differs from the frozen source-node score.")
    if not source_indices or (negative and not np.any(expected > 1e-3)):
        raise ValueError("A declared true negative lacks a loaded rejected source.")
    if not negative and not np.all(expected <= 1e-3):
        raise ValueError("A declared positive contains a failed original score.")
    provenance = {
        "original_result": checked_identity(row["original_result"]),
        "physical_inputs": row["physical_inputs"],
        "model": pair["nodal_v4"]["model"],
        "base_artifacts": pair["base_source_artifacts"],
        "checker_artifacts": pair["checker_source_artifacts"],
        "frozen_operator_scores": pair["nodal_v4"]["score_arrays"],
        "base_rule": pair["base"],
        "checker_rule": pair["independently_fitted_checker"],
        "base_application_truth_per_environment": pair["base_application_truth_per_environment"],
        "pair_eligible_per_environment": pair["pair_eligible_per_environment"],
        "whole_frame_qualification": row["whole_frame_qualification"],
        "mechanical_scope": row["mechanical_scope"],
        "negative": negative,
    }
    for identity in [*pair["base_source_artifacts"], *pair["checker_source_artifacts"]]:
        checked_identity(identity)
    name = row["name"] + "-" + pair["base"] + "-vs-" + pair["independently_fitted_checker"]
    return OperatorControl(name, face_nodes, difference, expected, source_indices, provenance)


def run_control(control, order, directory):
    directory.mkdir()
    entries = [
        (source, face)
        for source, values in enumerate(control.differences)
        for face in np.flatnonzero(np.any(values != 0.0, axis=-1))
    ]
    # Vary the global task order, while retaining the same complete source-local
    # nonzero signed operator. Omitted exactly-zero face vectors contribute zero.
    entries.sort(key=lambda entry: (entry[1], entry[0]), reverse=order == "reverse")
    capacity = max(len(entries), len(control.expected_scores))
    workspace = create_scatter_workspace(1, capacity, adaptive=True)
    tasks = np.zeros((capacity, 2), dtype=gs.np_int)
    heads = np.zeros((capacity, 2), dtype=gs.np_int)
    heads[:, 0] = -1
    next_task = np.full((capacity, 1), -1, dtype=gs.np_int)
    differences = np.zeros((capacity, 6), dtype=gs.np_float)
    for task, (source, face) in enumerate(entries):
        tasks[task] = source, face
        next_task[task, 0] = heads[source, 0]
        heads[source, 0] = task
        heads[source, 1] += 1
        differences[task] = control.differences[source, face]
    workspace.tasks.from_numpy(tasks)
    workspace.source_heads.from_numpy(heads)
    workspace.next_task.from_numpy(next_task)
    workspace.load_difference.from_numpy(differences)
    nodes = V(dtype=gs.qd_int, shape=control.surface_nodes.shape)
    nodes.from_numpy(np.ascontiguousarray(control.surface_nodes))
    errors = V(dtype=gs.qd_float, shape=(len(control.expected_scores),))
    kernel_node_error(errors, nodes, workspace)
    qd.sync()
    actual = qd_to_numpy(errors, copy=True)
    difference = np.abs(actual - control.expected_scores)
    admitted = actual <= 1e-3
    expected_admitted = control.expected_scores <= 1e-3
    passed = bool(
        np.isfinite(actual).all() and (difference <= 1e-12).all() and np.array_equal(admitted, expected_admitted)
    )
    np.savez_compressed(
        directory / "operator.npz",
        surface_nodes=control.surface_nodes,
        original_signed_differences=control.differences,
        native_score=actual,
        independent_score=control.expected_scores,
        tasks=tasks,
        source_heads=heads,
        next_task=next_task,
        load_difference=differences,
    )
    return {
        "name": control.name,
        "order": order,
        "source_indices": control.original_sources,
        "provenance": control.provenance,
        "native_score": actual.tolist(),
        "independent_score": control.expected_scores.tolist(),
        "absolute_difference": difference.tolist(),
        "native_admitted": admitted.tolist(),
        "expected_admitted": expected_admitted.tolist(),
        "absolute_operator_parity_tolerance": 1e-12,
        "admission_threshold_unchanged": 1e-3,
        "task_count": len(entries),
        "original_physical_face_count": len(control.surface_nodes),
        "arrays": file_id(directory / "operator.npz"),
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-result", type=Path, required=True)
    parser.add_argument("--positive-results", type=Path, nargs=2, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not IS_SLURM:
        parser.error("Scientific execution requires Slurm rtx-mid/image1_26.")
    if Path(gs.__file__).resolve().parent.parent != args.git_source.resolve():
        raise ValueError("Imported native source differs from explicit --git-source.")
    saved = json.loads(args.admission_result.read_text())
    if saved["complete"] is not True or saved["passed"] is not True or saved["pair_count"] != 44:
        raise ValueError("Use the terminal frozen44-pair v4 operator study.")
    positive_ids = [file_id(path) for path in args.positive_results]
    if len({identity["sha256"] for identity in positive_ids}) != 2:
        raise ValueError("Use the two distinct actual221/606 reference results.")
    selected = []
    for row in saved["rows"]:
        for pair in row["pairs"]:
            negative = all(pair["pair_eligible_per_environment"]) and any(
                value is False for value in pair["base_application_truth_per_environment"]
            )
            positive = (
                row["original_result"] in positive_ids
                and pair["base"] == "angle16"
                and pair["independently_fitted_checker"] == "angle8"
            )
            if negative or positive:
                selected.append((row, pair, negative))
    if len(selected) != 5 or sum(negative for _row, _pair, negative in selected) != 3:
        raise ValueError("Require exactly three real bad controls and two actual positive controls.")
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu, precision="64", seed=510000, logging_level="warning")
    controls = [load_control(row, pair, negative) for row, pair, negative in selected]
    negative = next(control for control in controls if control.provenance["negative"])
    # Algebraic isolation control only: equal opposite signed operators share
    # topology, but retain two distinct source IDs. No physical input is edited.
    bad_index = np.flatnonzero(negative.expected_scores > 1e-3)[0]
    values = negative.differences[bad_index : bad_index + 1]
    sign_control = OperatorControl(
        "opposite-source-operators-do-not-cancel",
        negative.surface_nodes,
        np.concatenate((values, -values)),
        np.repeat(negative.expected_scores[bad_index : bad_index + 1], 2),
        [(0, 0), (0, 1)],
        {
            "algebraic_control": True,
            "derived_from": negative.provenance,
            "physical_source_replay": False,
            "aggregate_signed_difference_exact_zero": bool(np.all(values + (-values) == 0.0)),
            "per_source_score_still_rejects": bool(negative.expected_scores[bad_index] > 1e-3),
        },
    )
    controls.append(sign_control)
    bound = [file_id(args.admission_result), *positive_ids]
    for row, pair, _negative in selected:
        bound.extend(
            [
                row["original_result"],
                row["physical_inputs"],
                pair["nodal_v4"]["model"],
                pair["nodal_v4"]["score_arrays"],
                *pair["base_source_artifacts"],
                *pair["checker_source_artifacts"],
            ]
        )
    result = {
        "schema": "genesis.dpr-fast-native-source-node-controls.v1",
        "complete": False,
        "passed": False,
        "goal_complete": False,
        "end_to_end_detector_qualified": False,
        "source_revision": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "worker": file_id(Path(__file__)),
        "native_aggregation_source": file_id(
            args.git_source / "genesis/engine/solvers/rigid/stress/adaptive_pressure.py"
        ),
        "worker_identity": runtime_identity(require_gpu_inventory=args.backend == "cuda"),
        "frozen_admission_result": file_id(args.admission_result),
        "physical_inputs_changed": False,
        "reference_solves_performed_here": False,
        "pressure_bank_changed": False,
        "controls": [],
        "scope": "Native shared cached source-node aggregation only, on frozen independently fitted positive/bad operators and an algebraic source-isolation control. No pressure fitting, source modification, elastic solve or end-to-end bad-field detection is performed here.",
    }
    write_json(args.output / "result.json", result)
    for index, control in enumerate(controls):
        for order in ("forward", "reverse"):
            report = run_control(control, order, args.output / f"control{index:02d}-{order}")
            result["controls"].append(report)
            write_json(args.output / "result.json", finite_json(result))
    result["complete"] = True
    result["bound_inputs_unchanged"] = [checked_identity(identity) for identity in bound] == bound
    result["passed"] = result["bound_inputs_unchanged"] and all(row["passed"] for row in result["controls"])
    write_json(args.output / "result.json", finite_json(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
