"""Compare all saved paired-reference arrays and full stress fields without a new solve."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_articulated_contact import quadrature_fields
from .qualify_articulated_precision import peak, tensor_from_f
from .resume_interrupted_spatial_reference import compare_state_arrays
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--allow-resolved-initializer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, configs, audits, paths = {}, [], [], []

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[str(path)] = digest
        return digest

    for run, audit in [(args.baseline, args.baseline_audit), (args.candidate, args.candidate_audit)]:
        for directory in [run, audit]:
            closed_source(directory)
            assert int((directory / "returncode.txt").read_text()) == 0
        config_path, audit_path = run / "results/configuration.json", audit / "results/summary.json"
        retain(config_path)
        retain(audit_path)
        config = json.loads(config_path.read_text())
        report = json.loads(audit_path.read_text())
        assert Path(report["run"]).resolve() == run.resolve()
        assert report["states_checked"] == len(report["rows"]) == config["steps"]
        assert [r["step"] for r in report["rows"]] == list(range(1, config["steps"] + 1))
        for row in report["rows"]:
            assert row["complete_body_dynamic_residual_N"] <= 1e-9
            assert row["complete_mapped_dynamic_residual_N"] <= 1e-9
        configs.append({key: value for key, value in config.items() if key != "output"})
        audits.append(report)
        paths.append(sorted((run / "results").glob("step[0-9][0-9][0-9][0-9].npz")))
        assert len(paths[-1]) == config["steps"]
    assert configs[0] == configs[1]
    for name in ["reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz"]:
        assert retain(args.baseline / "results" / name) == retain(args.candidate / "results" / name)
    initializers = [run / "results/closure_state.npz" for run in [args.baseline, args.candidate]]
    initializer_hashes = [retain(path) for path in initializers]
    initializer_difference = 0.0
    if args.allow_resolved_initializer:
        with np.load(initializers[0]) as original, np.load(initializers[1]) as changed:
            assert original.files == changed.files
            for key in original.files:
                if key != "displacement":
                    np.testing.assert_array_equal(original[key], changed[key])
                    continue
                delta = np.asarray(changed[key], dtype=np.longdouble) - original[key]
                error = np.linalg.norm(delta)
                assert error <= 1e-5 * np.linalg.norm(original[key]) + 1e-12
                initializer_difference = float(abs(delta).max(initial=0))
    else:
        assert initializer_hashes[0] == initializer_hashes[1]
    static, _ = restore_static(args.baseline)
    bulk = WidePiolaBulk(static.body)
    rows = []
    for step, (baseline, candidate) in enumerate(zip(*paths, strict=True), 1):
        assert baseline.name == candidate.name == f"step{step:04d}.npz"
        retain(baseline)
        retain(candidate)
        with np.load(baseline) as original, np.load(candidate) as changed:
            arrays = compare_state_arrays(original, changed)
            fields = []
            for saved in [original, changed]:
                vector = np.empty(static.body.basis.N, dtype=np.longdouble)
                vector[static.parts[0].permutation] = saved["displacement"][: static.body.basis.N]
                corner = tensor_from_f(bulk.corner_deformation_gradients(vector), static.body.lame)
                quadrature = quadrature_fields(bulk, np.array(vector, dtype=float))[0]
                fields.append((corner, quadrature))
        corner_change = float(np.linalg.norm(fields[1][0] - fields[0][0]) / np.linalg.norm(fields[0][0]))
        quad_change = float(np.linalg.norm(fields[1][1] - fields[0][1]) / np.linalg.norm(fields[0][1]))
        peak_change = abs(peak(fields[1][0]) / peak(fields[0][0]) - 1)
        assert max(corner_change, quad_change, peak_change) <= 1e-6
        row = {
            "step": step,
            "time_s": step * configs[0]["dt"],
            "all_saved_array_changes": arrays,
            "all_corner_tensor_relative_change": corner_change,
            "all_quadrature_tensor_relative_change": quad_change,
            "raw_global_peak_relative_change": peak_change,
            "independent_force_audits": [report["rows"][step - 1] for report in audits],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "scope": __doc__,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "body_tetrahedra": static.body.mesh.nelements,
        "states_checked": len(rows),
        "rows": rows,
        "input_sha256": inputs,
        "initializer_comparison": {
            "independently_resolved_initializer_allowed": args.allow_resolved_initializer,
            "byte_identical": initializer_hashes[0] == initializer_hashes[1],
            "maximum_displacement_difference_m": initializer_difference,
            "other_initializer_arrays_identical": True,
        },
        "all_original_arrays_and_complete_fields_qualified": True,
        "declared_existing_budgets": {
            "state_relative": 1e-5,
            "state_absolute": 1e-12,
            "all_corner_and_quadrature_tensor_relative": 1e-6,
            "independently_audited_physical_body_and_mapped_force_N": 1e-9,
        },
        "goal_completion": False,
        "limitations": "Complete saved histories are compared at identical physical inputs and byte-identical geometry. By default initializer bytes must match; the explicit resolved-initializer option instead checks every initializer array at the original state budget and retains its measured difference. Independent force-audit results are checked and retained, not reassembled in this field comparison. This proves numerical equivalence only for these saved states; it establishes neither physical convergence nor a controlled timing improvement or warmed production cost.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
