"""Apply the predeclared exact, quadrature and history budgets to retained complete fields."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def per_environment(actual, expected):
    delta = (actual - expected).reshape((actual.shape[0], -1))
    reference = expected.reshape((expected.shape[0], -1))
    return np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--kind", choices=("exact", "quadrature", "history"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_budget = 1e-6 if args.kind == "quadrature" else 1e-10
    field_budget = {"exact": 1e-8, "quadrature": 1e-4, "history": 1e-5}[args.kind]
    result = dict(
        identity=runtime_identity(require_gpu_inventory=False),
        kind=args.kind,
        load_budget=load_budget,
        field_budget=field_budget,
        complete=False,
        passed=False,
        rows=[],
    )
    write_json(args.output / "result.json", result)
    starts = [json.loads((p / "start.json").read_text()) for p in (args.before, args.after)]
    completed = [json.loads((p / "result.json").read_text()) for p in (args.before, args.after)]
    assert all(d["passed"] for d in completed)
    assert starts[0]["conditions"]["sha256"] == starts[1]["conditions"]["sha256"]
    assert completed[0]["accepted_frames"] == completed[1]["accepted_frames"]
    result["accepted_frames"] = completed[0]["accepted_frames"]
    with np.load(args.before / "observations.npz") as before, np.load(args.after / "observations.npz") as after:
        np.testing.assert_array_equal(before["positions"], after["positions"])
        difference = np.abs(after["peaks"] - before["peaks"])
        budgets = field_budget * np.abs(before["peaks"]) + 1e-8
        result["peak_budget_passed"] = bool((difference <= budgets).all())
        result["peak_max_scaled_error"] = float(np.max(difference / np.maximum(np.abs(before["peaks"]), 1e-8)))
        result["peak_max_absolute_Pa"] = float(difference.max())
        result["peak_worst_frame_env"] = [
            int(i)
            for i in np.unravel_index(
                np.argmax(difference / np.maximum(np.abs(before["peaks"]), 1e-8)), difference.shape
            )
        ]
    before_files = {p.name for p in args.before.glob("field-*.npz")}
    assert before_files and before_files == {p.name for p in args.after.glob("field-*.npz")}
    for name in sorted(before_files):
        with np.load(args.before / name) as before, np.load(args.after / name) as after:
            for key in ["valid", "status", "force", "normal", "friction", "position"]:
                np.testing.assert_array_equal(after[key], before[key])
            radius_bitwise_equal = np.array_equal(after["radius"], before["radius"])
            radius_relative = np.abs(after["radius"] - before["radius"]) / np.maximum(np.abs(before["radius"]), 1e-30)
            # Match the established scene audit's radius contract and retain the stronger bitwise observation.
            np.testing.assert_allclose(after["radius"], before["radius"], rtol=1e-12, atol=0)
            errors = {}
            for key in ["nodal_force", "stress_tensor", "shell_membrane", "shell_bending", "shell_transverse"]:
                if key in before:
                    errors[key] = per_environment(after[key], before[key]).tolist()
            passed = all(
                max(values) <= (load_budget if key == "nodal_force" else field_budget) for key, values in errors.items()
            )
            result["rows"].append(
                dict(
                    file=name,
                    before=file_id(args.before / name),
                    after=file_id(args.after / name),
                    errors=errors,
                    radius_bitwise_equal=radius_bitwise_equal,
                    radius_max_relative=float(radius_relative.max()),
                    passed=passed,
                )
            )
        write_json(args.output / "result.json", result)
    result["complete"] = True
    result["passed"] = result["peak_budget_passed"] and all(row["passed"] for row in result["rows"])
    write_json(args.output / "result.json", result)
    assert result["passed"], result


if __name__ == "__main__":
    main()
