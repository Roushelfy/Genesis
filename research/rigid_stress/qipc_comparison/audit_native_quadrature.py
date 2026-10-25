"""Bind an admitted native trajectory to the fixed-fan high-order field controls."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .audit_gpu_candidates import per_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    completed = json.loads((args.native / "result.json").read_text())
    assert completed["passed"] and completed["accepted_frames"] == 600
    references = {p.name: p for p in args.references.glob("group*/frame*") if p.is_dir()}
    result = dict(
        identity=runtime_identity(require_gpu_inventory=False),
        native=file_id(args.native / "result.json"),
        load_budget=1e-6,
        field_budget=1e-4,
        complete=False,
        rows=[],
    )
    for frame in (129, 221, 400, 599):
        reference_path = references[f"frame{frame}"]
        report = json.loads((reference_path / "result.json").read_text())
        assert report["complete"] and report["force_fan_reference"] and report["reference_quadrature"] == 256
        native_path = args.native / f"field-{frame:04d}.npz"
        original_path = args.original / native_path.name
        with (
            np.load(native_path) as actual,
            np.load(original_path) as original,
            np.load(reference_path / "angle256.npz") as expected,
        ):
            for key in ("valid", "force", "normal", "position", "friction", "angular_velocity"):
                np.testing.assert_array_equal(actual[key], original[key])
            # Match the established trajectory audit's radius contract and record
            # the stronger bitwise observation instead of silently requiring it.
            radius_bitwise_equal = bool(np.array_equal(actual["radius"], original["radius"]))
            radius_relative = np.abs(actual["radius"] - original["radius"]) / np.maximum(
                np.abs(original["radius"]), 1e-30
            )
            np.testing.assert_allclose(actual["radius"], original["radius"], rtol=1e-12, atol=0.0)
            errors = {
                key: per_environment(actual[key], expected[key])
                for key in ("nodal_force", "displacement", "stress_tensor")
            }
            peak = actual["von_mises"].reshape((len(actual["valid"]), -1)).max(axis=1)
            errors["peak"] = per_environment(peak, expected["peak"])
        passed = all(
            float(values.max()) < (result["load_budget"] if key == "nodal_force" else result["field_budget"])
            for key, values in errors.items()
        )
        result["rows"].append(
            dict(
                frame=frame,
                native=file_id(native_path),
                reference=file_id(reference_path / "angle256.npz"),
                mechanical_inputs_bitwise_unchanged=True,
                radius_within_original_contract=True,
                radius_bitwise_equal=radius_bitwise_equal,
                radius_max_relative=float(radius_relative.max()),
                errors={key: float(values.max()) for key, values in errors.items()},
                per_environment_errors={key: values.tolist() for key, values in errors.items()},
                passed=passed,
            )
        )
        write_json(args.output / "result.json", result)
    result["complete"] = True
    result["passed"] = all(row["passed"] for row in result["rows"])
    write_json(args.output / "result.json", result)
    assert result["passed"]


if __name__ == "__main__":
    main()
