"""Compare actual scene fields before and after recording same-pose face bounds."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--before", type=Path, required=True)
    cli.add_argument("--after", type=Path, required=True)
    cli.add_argument("--numerical-inputs", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"field_relative": 1e-9, "curvature_relative": 1e-13, "wrench_absolute": 1e-12}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "targets": targets,
            "numerical_constitutive_inputs": args.numerical_inputs,
        },
    )
    floating = {
        "recovery_wrench_error",
        "vm_pa",
        "principal_pa",
        "probe_tensor_pa",
        "corner_tensor_pa",
        "regular_force_n",
        "region_curvature_per_m",
    }
    maxima = {key: {"relative": 0.0, "absolute": 0.0} for key in floating}
    exact_inputs = {
        "contact_force_n",
        "contact_position_m",
        "position_m",
        "quaternion",
        "frame_position_m",
        "frame_quaternion",
        "source_witness_m",
        "omega_per_s",
        "material_nodes_m",
        "material_face_normal",
        "material_surface_nodes",
        "probe_offsets",
    }
    rows = []
    for step in range(109):
        before, after = args.before / f"frame{step:04d}.npz", args.after / f"frame{step:04d}.npz"
        with np.load(before) as old, np.load(after) as new:
            assert old.files == new.files
            for key in old.files:
                if args.numerical_inputs and old[key].dtype.kind == "f" and key not in exact_inputs:
                    floating.add(key)
                    maxima.setdefault(key, {"relative": 0.0, "absolute": 0.0})
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
                    elif key == "region_curvature_per_m":
                        assert relative < targets["curvature_relative"], (step, key, relative)
                    else:
                        assert relative < targets["field_relative"], (step, key, relative)
        rows.append({"step": step, "before": file_id(before), "after": file_id(after)})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "frames": rows,
            "maxima": maxima,
            "scope": "All109 captures retain exact rigid positions, forces, material geometry and acceptance. Curvature uses relative1e-13 and fields relative1e-9, with absolute wrench1e-12. With numerical-inputs enabled, other constitutive floating fields use the same relative1e-9 budget; otherwise histories and law inputs remain exact. Every maximum is retained. These comparison budgets do not change production acceptance.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
