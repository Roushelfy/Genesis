"""Compare retained keyframe robot poses with independently initialized shared prescribed targets."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import corotation
from .reference_uncertainty import frame_index


def matrices(quaternion):
    return Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binding = json.loads((args.binding / "result.json").read_text())
    assert binding["kind"] == "initial-prescribed-target-binding" and binding["physics_steps_executed"] == 0
    with np.load(args.binding / "prescribed-targets.npz") as saved:
        times, ids = saved["time_s"], saved["public_body_indices"]
        target_com, target_quaternion = saved["target_com_world_m"], saved["target_material_quaternion_wxyz"]
        names = saved["body_names"].tolist()
    assert np.array_equal(np.sort(ids), np.arange(len(ids)))
    with np.load(args.layout) as saved:
        owners, local = saved["rigid_vertex_public_owner"], saved["rigid_local_points_m"]
    np.testing.assert_allclose(matrices(target_quaternion[0]), matrices(-target_quaternion[0]), rtol=0, atol=0)
    records = []
    for root in args.profiles:
        for receipt in sorted(root.glob("group*/*/result.json")):
            folder = receipt.parent
            inputs = json.loads((folder / "inputs.json").read_text())
            assert inputs["native"]["sha256"] == binding["native"]["sha256"]
            assert inputs["baseline_inputs"]["sha256"] == binding["baseline_inputs"]["sha256"]
            frames = frame_index(folder)
            with np.load(folder / "rest.npz") as saved:
                authored, masses = saved["authored_vertices_m"], saved["masses_kg"]
            rows = []
            for number, time in enumerate(times):
                tick = round(time / 0.01)
                with np.load(frames[tick]) as saved:
                    com, quaternion = saved["actual_robot_com_world_m"], saved["actual_robot_material_quaternion_wxyz"]
                    egg_positions = saved["positions_m"]
                egg_rotation = corotation(authored, egg_positions, masses)
                egg_tilt = float(np.arctan2(np.linalg.norm(egg_rotation[:2, 2]), egg_rotation[2, 2]))
                target_rotation = matrices(target_quaternion[number])
                rotation = matrices(quaternion)
                position_error = np.linalg.norm(com[ids] - target_com[number], axis=1)
                angle_error = Rotation.from_matrix(rotation[ids] @ target_rotation.transpose(0, 2, 1)).magnitude()
                expected_com, expected_rotation = np.empty_like(com), np.empty_like(rotation)
                expected_com[ids], expected_rotation[ids] = target_com[number], target_rotation
                expected_points = np.einsum("vij,vj->vi", expected_rotation[owners], local) + expected_com[owners]
                actual_points = np.einsum("vij,vj->vi", rotation[owners], local) + com[owners]
                vertex_error = float(np.linalg.norm(actual_points - expected_points, axis=1).max())
                rows.append(
                    {
                        "time_s": float(time),
                        "maximum_com_error_m": float(position_error.max()),
                        "maximum_material_rotation_error_rad": float(angle_error.max()),
                        "maximum_represented_rigid_vertex_tracking_error_m": vertex_error,
                        "vertex_error_over_contact_band": vertex_error / inputs["config"]["d_hat"],
                        "egg_mass_com_world_m": (masses @ egg_positions / masses.sum()).tolist(),
                        "egg_fitted_material_axis_tilt_rad": egg_tilt,
                        "per_body_com_error_m": position_error.tolist(),
                        "per_body_rotation_error_rad": angle_error.tolist(),
                    }
                )
            records.append({"name": folder.name, "source": file_id(receipt), "keyframes": rows})
    write_json(
        args.output / "result.json",
        {
            "kind": "declared-keyframe-prescribed-boundary-tracking-audit",
            "identity": runtime_identity(),
            "binding": file_id(args.binding / "result.json"),
            "layout": file_id(args.layout),
            "body_names": names,
            "quaternion_sign_invariance_passed": True,
            "profiles": records,
            "reference_qualified": False,
            "scope": "Exact shared target poses use the pinned original initialized material frames, with no dynamics restart or fitted actual motion. Errors describe retained keyframes only. Represented rigid-vertex errors use the independently verified static layout, not a continuous Hausdorff certificate. Target tracking, free-egg trajectory/contact-history differences and elasticity/discretization error remain separate. A small target error relative to the barrier band alone does not qualify contact stress.",
        },
    )


if __name__ == "__main__":
    main()
