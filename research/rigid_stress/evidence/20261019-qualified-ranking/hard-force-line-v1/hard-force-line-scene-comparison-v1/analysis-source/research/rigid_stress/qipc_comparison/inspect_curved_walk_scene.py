"""Compare scene fields with physical scales for near-zero wrench residuals."""

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
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "scope": "Exact rigid inputs and acceptance; existing1e-9 stress-field and1e-13 curvature comparison budgets. Input moment residuals are measured in Nm and divided by Fn*a, using the unchanged native1e-8 gate; relative change of a roundoff residual is retained only as a diagnostic. Other floating intermediates are measured without adding acceptance claims.",
        },
    )
    exact = {
        "contact_force_n",
        "contact_position_m",
        "contact_normal",
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
    fields = {"vm_pa", "principal_pa", "probe_tensor_pa", "corner_tensor_pa", "regular_force_n"}
    before_files, after_files = sorted(args.before.glob("frame*.npz")), sorted(args.after.glob("frame*.npz"))
    assert [path.name for path in before_files] == [path.name for path in after_files]
    maxima, rows = {}, []
    moment_scaled, moment_absolute = [0.0, 0.0], [0.0, 0.0]
    for before, after in zip(before_files, after_files):
        with np.load(before) as old, np.load(after) as new:
            assert old.files == new.files
            for key in old.files:
                if old[key].dtype.kind != "f" or key in exact:
                    np.testing.assert_array_equal(old[key], new[key], err_msg=f"{before.name}: {key}")
                else:
                    np.testing.assert_array_equal(np.isfinite(old[key]), np.isfinite(new[key]))
                    is_finite = np.isfinite(old[key])
                    difference = old[key][is_finite] - new[key][is_finite]
                    absolute = np.max(np.abs(difference), initial=0.0)
                    relative = np.linalg.norm(difference) / max(np.linalg.norm(old[key][is_finite]), 1e-30)
                    item = maxima.setdefault(
                        key, {"relative": 0.0, "absolute": 0.0, "before_max_abs": 0.0, "after_max_abs": 0.0}
                    )
                    item["relative"] = max(item["relative"], float(relative))
                    item["absolute"] = max(item["absolute"], float(absolute))
                    item["before_max_abs"] = max(
                        item["before_max_abs"], float(np.max(np.abs(old[key][is_finite]), initial=0.0))
                    )
                    item["after_max_abs"] = max(
                        item["after_max_abs"], float(np.max(np.abs(new[key][is_finite]), initial=0.0))
                    )
                    if key in fields:
                        assert relative < 1e-9, (before.name, key, relative)
                    elif key == "region_curvature_per_m":
                        assert relative < 1e-13
                    elif key == "recovery_wrench_error":
                        assert absolute < 1e-12
            for index, data in enumerate([old, new]):
                residual = data["input_moment_error_nm"]
                scale = (data["law_force_n"] * data["law_axes_m"][:, 0]).reshape(residual.shape[::-1]).T
                is_loaded = scale > 0
                assert np.all(residual[~is_loaded] == 0)
                normalized = residual[is_loaded] / scale[is_loaded]
                moment_scaled[index] = max(moment_scaled[index], float(normalized.max(initial=0.0)))
                moment_absolute[index] = max(moment_absolute[index], float(np.abs(residual).max(initial=0.0)))
                assert np.all(normalized <= 1e-8)
        rows.append({"before": file_id(before), "after": file_id(after)})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "frames": rows,
            "maxima": maxima,
            "moment_residual_maximum_before_after_nm": moment_absolute,
            "moment_residual_maximum_before_after_over_Fn_a": moment_scaled,
            "rigid_inputs_and_acceptance_exact": True,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
