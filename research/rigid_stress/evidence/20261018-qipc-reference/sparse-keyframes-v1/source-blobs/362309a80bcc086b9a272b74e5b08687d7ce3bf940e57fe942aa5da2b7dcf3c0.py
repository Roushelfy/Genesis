"""Check selected-field invariance and actual CUDA recovery counts for native sampling profiles."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import voigt_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    bands = plan["strict_observer_equivalence_bands"]
    folders = {p.parent.name: p.parent for p in args.profiles.glob("group*/*/result.json")}
    records = []
    for method in ("dpr", "scr"):
        dense, sparse = (folders[f"{method}-{name}"] for name in ("every_step", "manual"))
        receipts = [json.loads((path / "result.json").read_text()) for path in (dense, sparse)]
        assert all(row["completed_requested_history"] and row["initial_mechanical_reset_exact"] for row in receipts)
        assert receipts[0]["conditions"]["sha256"] == receipts[1]["conditions"]["sha256"]
        with np.load(dense / "rest.npz") as saved:
            vertices, tets = saved["vertices"], saved["tetrahedra"]
        ds = vertices[tets[:, 1:]] - vertices[tets[:, :1]]
        volumes = np.abs(np.linalg.det(ds)) / 6
        states = []
        for time in plan["full_spatial_stress_keyframes_s"]:
            tensors, poses, quaternions, coordinates, peaks = [], [], [], [], []
            for path in (dense, sparse):
                frame = path / "frames" / f"step{round(time / 0.001) - 1:06d}.npz"
                with np.load(frame) as saved:
                    tensors.append(voigt_tensor(saved["stress_voigt_link_pa"]))
                    poses.append(saved["link_position_m"])
                    quaternions.append(saved["link_quaternion_wxyz"])
                    coordinates.append(saved["q"])
                    peaks.append(saved["vm_pa"].max().item())
            assert tensors[0].shape == tensors[1].shape == (len(volumes), 4, 3, 3)
            assert all(np.isfinite(tensor).all() for tensor in tensors)
            numerator = (volumes[:, None, None, None] * (tensors[1] - tensors[0]) ** 2).sum()
            denominator = (volumes[:, None, None, None] * tensors[0] ** 2).sum()
            contrast = np.sqrt(numerator / max(denominator, 4 * volumes.sum())).item()
            translation = np.linalg.norm(poses[1] - poses[0]).item()
            relative_rotation = float(
                (
                    Rotation.from_quat(quaternions[1][[1, 2, 3, 0]])
                    * Rotation.from_quat(quaternions[0][[1, 2, 3, 0]]).inv()
                ).magnitude()
            )
            states.append(
                {
                    "time_s": time,
                    "link_position_difference_m": translation,
                    "link_rotation_difference_rad": relative_rotation,
                    "maximum_joint_coordinate_difference": np.max(np.abs(coordinates[1] - coordinates[0])).item(),
                    "relative_authored_corner_tensor_with_1Pa_floor": contrast,
                    "raw_spatial_vm_peaks_pa": peaks,
                    "strict_observer_equivalence_passed": bool(
                        translation <= bands["maximum_link_position_difference_m"]
                        and contrast <= bands["relative_authored_corner_tensor_with_1Pa_floor"]
                    ),
                }
            )
        kernels = [json.loads((path / "kernels.json").read_text()) for path in (dense, sparse)]
        prefixes = (
            ("kernel_rigid_stress_post",)
            if method == "dpr"
            else ("kernel_resolve_stress_contacts", "kernel_initialize_intrinsic_contacts", "kernel_surface_apply")
        )
        selected = []
        for prefix in prefixes:
            for label, rows in zip(("every_step", "manual"), kernels):
                selected.append(
                    {
                        "prefix": prefix,
                        "sampling": label,
                        "kernels": [row for row in rows if row["name"].startswith(prefix)],
                    }
                )
        records.append(
            {
                "method": method,
                "dense": file_id(dense / "result.json"),
                "sparse": file_id(sparse / "result.json"),
                "keyframes": states,
                "all_strict_keyframe_bands_passed": all(row["strict_observer_equivalence_passed"] for row in states),
                "selected_recovery_kernel_counts": selected,
                "profiled_window_seconds": [row["window_seconds"][-1] for row in receipts],
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "native-sparse-field-and-CUDA-kernel-audit",
            "identity": runtime_identity(),
            "plan": file_id(args.plan),
            "comparisons": records,
            "reference_accuracy_qualified": False,
            "scope": "Same-method observer equivalence at all3 predeclared fields, using volume-weighted four-corner matrix norms with1Pa floor in the shared authored frame. This positive corner metric is an observation-mode check, not a new continuum accuracy norm. Exact CUDA kernel counts come from retained full CUPTI traces over the200-step contact window. Profiled costs remain intrusive single-run observations, separate from primary throughput. Physical-reference and full-grasp qualification remain open.",
        },
    )


if __name__ == "__main__":
    main()
