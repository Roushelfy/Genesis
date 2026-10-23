"""Check equivalent-input DPR optimizations using the existing numerical error budgets."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [args.before, args.after]
    starts = [json.loads((path / "start.json").read_text()) for path in paths]
    results = [json.loads((path / "result.json").read_text()) for path in paths]
    assert all(result["passed"] for result in results)
    assert results[0]["accepted_frames"] == results[1]["accepted_frames"]
    assert starts[0]["conditions"]["sha256"] == starts[1]["conditions"]["sha256"]
    with np.load(args.before / "observations.npz") as before, np.load(args.after / "observations.npz") as after:
        np.testing.assert_array_equal(after["positions"], before["positions"])
        np.testing.assert_allclose(after["peaks"], before["peaks"], rtol=1e-8, atol=1e-8)
    before_files = {path.name for path in args.before.glob("field-*.npz")}
    after_files = {path.name for path in args.after.glob("field-*.npz")}
    assert before_files and before_files == after_files
    rows, failures = [], []
    for name in sorted(before_files):
        with np.load(args.before / name) as before, np.load(args.after / name) as after:
            active = before["valid"]
            row = {"field": name, "before": file_id(args.before / name), "after": file_id(args.after / name)}
            delta = after["stress_tensor"] - before["stress_tensor"]
            row["tensor_max_absolute_difference_pa"] = float(np.abs(delta).max())
            row["tensor_relative_difference"] = float(np.linalg.norm(delta) / np.linalg.norm(before["stress_tensor"]))
            load_delta = np.linalg.norm(after["nodal_force"] - before["nodal_force"], axis=(1, 2))
            load_norm = np.linalg.norm(before["nodal_force"], axis=(1, 2))
            row["relative_load_difference_per_environment"] = np.divide(
                load_delta, load_norm, out=np.zeros_like(load_delta), where=load_norm > 0
            ).tolist()
            try:
                np.testing.assert_array_equal(after["valid"], active)
                for key in ("force", "normal", "position", "friction", "radius_status", "status"):
                    np.testing.assert_array_equal(after[key][active], before[key][active])
                np.testing.assert_allclose(after["radius"][active], before["radius"][active], rtol=1e-12, atol=0)
                np.testing.assert_allclose(after["stress_tensor"], before["stress_tensor"], rtol=1e-8, atol=1e-8)
                assert (load_delta <= 1e-9 * load_norm).all(), row
                loaded = active & (np.linalg.norm(after["force"], axis=-1) > 0)
                magnitude = np.linalg.norm(after["force"][loaded], axis=-1)
                assert (after["force_error"][loaded] <= 1e-8 * magnitude).all()
                assert (after["moment_error"][loaded] <= 1e-8 * magnitude * after["radius"][loaded]).all()
            except AssertionError as error:
                failures.append({"field": name, "error": str(error)})
            rows.append(row)
    write_json(
        args.output / "result.json",
        {
            "passed": not failures,
            "identity": runtime_identity(require_gpu_inventory=False),
            "accepted_frames": results[0]["accepted_frames"],
            "rigid_positions_exactly_equal": True,
            "rows": rows,
            "failures": failures,
            "scope": "Equivalent inputs and unchanged numerical budgets: load relative 1e-9, stress rtol/atol 1e-8, radius rtol 1e-12, original wrench 1e-8. Does not change or replace the historical bitwise audit.",
        },
    )
    assert not failures, failures


if __name__ == "__main__":
    main()
