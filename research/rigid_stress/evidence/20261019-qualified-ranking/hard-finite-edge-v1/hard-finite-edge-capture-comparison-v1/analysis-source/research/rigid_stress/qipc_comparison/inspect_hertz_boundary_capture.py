"""Compare actual scene fields before and after recording same-pose face bounds."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--before", type=Path, required=True)
    cli.add_argument("--after", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"field_relative": 1e-9, "wrench_absolute": 1e-12}
    write_json(
        args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False), "targets": targets}
    )
    floating = {
        "recovery_wrench_error",
        "vm_pa",
        "principal_pa",
        "probe_tensor_pa",
        "corner_tensor_pa",
        "regular_force_n",
    }
    maxima = {key: {"relative": 0.0, "absolute": 0.0} for key in floating}
    rows = []
    for step in range(109):
        before, after = args.before / f"frame{step:04d}.npz", args.after / f"frame{step:04d}.npz"
        with np.load(before) as old, np.load(after) as new:
            assert old.files == new.files
            for key in old.files:
                if key not in floating:
                    np.testing.assert_array_equal(old[key], new[key], err_msg=f"step {step}, field {key}")
                else:
                    np.testing.assert_array_equal(np.isfinite(old[key]), np.isfinite(new[key]))
                    is_finite = np.isfinite(old[key])
                    difference = old[key][is_finite] - new[key][is_finite]
                    absolute = np.max(np.abs(difference), initial=0.0)
                    relative = np.linalg.norm(difference) / max(np.linalg.norm(old[key][is_finite]), 1e-30)
                    maxima[key]["relative"] = max(maxima[key]["relative"], float(relative))
                    maxima[key]["absolute"] = max(maxima[key]["absolute"], float(absolute))
                    if key == "recovery_wrench_error":
                        assert absolute < targets["wrench_absolute"]
                    else:
                        assert relative < targets["field_relative"], (step, key, relative)
        rows.append({"step": step, "before": file_id(before), "after": file_id(after)})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "frames": rows,
            "maxima": maxima,
            "scope": "All 109 captured scene frames retain exact geometry, forces, histories and acceptance. Complete field arrays are compared with predeclared relative1e-9 and absolute wrench1e-12 gates. The first invalid observation remains at step108.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
