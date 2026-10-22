"""Read retained native route failures without rerunning the simulation."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json


def values(array):
    return [float(value) if np.isfinite(value) else str(value) for value in array.reshape(-1)]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for group in [0, 1]:
        root = args.campaign / f"group{group}/rotated"
        paths = sorted(root.glob("frame*.npz")) + sorted(root.glob("failed*.npz"))
        for path in paths:
            with np.load(path) as saved:
                data = {key: saved[key] for key in saved}
            loaded = np.flatnonzero(data["law_force_n"] > 0)
            rows.append(
                {
                    "input": file_id(path),
                    "input_status": data["input_status"].tolist(),
                    "contact_status": data["contact_status"].tolist(),
                    "region_status": data["region_status"].tolist(),
                    "recovery_status": data["recovery_status"].tolist(),
                    "wrench_error": values(data["recovery_wrench_error"]),
                    "step_valid": data["step_valid"].tolist(),
                    "elastic_valid": data["elastic_valid"].tolist(),
                    "vm_pa": values(data["vm_pa"]),
                    "principal_pa": values(data["principal_pa"]),
                    "loaded_patches": loaded.tolist(),
                    "invalid_probes": {
                        int(patch): np.flatnonzero(data["probe_elements"][patch] < 0).tolist() for patch in loaded
                    },
                    "normal_force_n": values(data["law_force_n"][loaded]),
                    "force_error_n": values(data["input_force_error_n"]),
                    "moment_error_nm": values(data["input_moment_error_nm"]),
                    "frame_position_m": data["frame_position_m"].tolist(),
                    "frame_quaternion": data["frame_quaternion"].tolist(),
                    "hertz_centers_m": data["hertz_centers_m"][loaded].tolist(),
                    "hertz_frames": data["hertz_frames"][loaded].tolist(),
                    "source_status": data["source_status"].tolist(),
                }
            )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "read_only": True,
            "failed_native_controls_admitted": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
