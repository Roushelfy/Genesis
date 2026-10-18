"""Check recorded actual QIPC public-body poses against authored joint kinematics with immutable initial frame offsets."""

import argparse
import json
from pathlib import Path

import numpy as np

import mujoco
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import corotation
from .history_audit import returncode
from .reference_uncertainty import frame_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--position-band", type=float, default=1e-7)
    parser.add_argument("--rotation-band", type=float, default=1e-6)
    parser.add_argument("--retain-poses", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads((args.reference / "result.json").read_text())
    inputs = json.loads((args.reference / "inputs.json").read_text())
    binding = json.loads((args.binding / "start.json").read_text())
    binding_receipt = json.loads((args.binding / "result.json").read_text())
    assert binding_receipt["kind"] == "initial-QIPC-public-rigid-frame-binding"
    assert reference["completed_requested_history"]
    assert returncode(args.reference.parent / "returncode.txt") == 0
    assert returncode(args.binding.parent / "returncode.txt") == 0
    assert inputs["native"]["sha256"] == binding["native"]["sha256"]
    assert inputs["case"] == binding_receipt["case"]
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    assert model.nq == 9
    state = mujoco.MjData(model)
    names = binding["public_rigid_link_names"]
    body = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names])
    assert (body >= 0).all()
    state.qpos[:] = binding["initial_q"]
    mujoco.mj_kinematics(model, state)
    initial_rotation = state.xmat[body].reshape(-1, 3, 3).copy()
    material_rotation = Rotation.from_quat(
        np.array(binding["public_rigid_material_quaternion_wxyz"])[:, [1, 2, 3, 0]]
    ).as_matrix()
    initial_com = np.array(binding["public_rigid_com_world_m"])
    offset_rotation = initial_rotation.transpose(0, 2, 1) @ material_rotation
    offset_com = np.einsum("bij,bj->bi", initial_rotation.transpose(0, 2, 1), initial_com - state.xpos[body])
    np.savez_compressed(
        args.output / "initial-frame-offsets.npz",
        body_names=np.array(names),
        material_rotation_in_authored_body=offset_rotation,
        com_in_authored_body_m=offset_com,
    )
    pose_rows = []
    if args.retain_poses:
        with np.load(args.reference / "rest.npz") as saved:
            authored, masses = saved["authored_vertices_m"], saved["masses_kg"]
    times, positions, angles = [], [], []
    for tick, path in frame_index(args.reference).items():
        with np.load(path) as saved:
            state.qpos[:] = saved["q"]
            mujoco.mj_kinematics(model, state)
            expected_rotation = state.xmat[body].reshape(-1, 3, 3) @ offset_rotation
            expected_com = state.xpos[body] + np.einsum("bij,bj->bi", state.xmat[body].reshape(-1, 3, 3), offset_com)
            actual_com = saved["actual_robot_com_world_m"]
            actual_rotation = Rotation.from_quat(
                saved["actual_robot_material_quaternion_wxyz"][:, [1, 2, 3, 0]]
            ).as_matrix()
            assert actual_com.shape == expected_com.shape
            positions.append(np.linalg.norm(actual_com - expected_com, axis=1))
            angles.append(Rotation.from_matrix(actual_rotation @ expected_rotation.transpose(0, 2, 1)).magnitude())
            times.append(tick * 0.01)
            if args.retain_poses:
                egg_rotation = corotation(authored, saved["positions_m"], masses)
                pose_rows.append(
                    {
                        "time_s": tick * 0.01,
                        "q": saved["q"].tolist(),
                        "target_q": saved["target_q"].tolist() if "target_q" in saved.files else None,
                        "actual_robot_com_world_m": actual_com.tolist(),
                        "egg_mass_com_world_m": (
                            (saved["positions_m"] * masses[:, None]).sum(axis=0) / masses.sum()
                        ).tolist(),
                        "egg_fitted_material_axis_tilt_rad": np.arctan2(
                            np.linalg.norm(egg_rotation[:2, 2]), egg_rotation[2, 2]
                        ).item(),
                    }
                )
    times, positions, angles = np.array(times), np.array(positions), np.array(angles)
    assert np.isfinite(positions).all() and np.isfinite(angles).all()
    records = []
    for index, name in enumerate(names):
        records.append(
            {
                "body": name,
                "maximum_com_difference_m": positions[:, index].max().item(),
                "maximum_com_time_s": times[positions[:, index].argmax()].item(),
                "maximum_rotation_difference_rad": angles[:, index].max().item(),
                "maximum_rotation_time_s": times[angles[:, index].argmax()].item(),
            }
        )
    np.savez_compressed(
        args.output / "recorded-pose-errors.npz",
        time_s=times,
        body_names=np.array(names),
        com_difference_m=positions,
        rotation_difference_rad=angles,
    )
    write_json(
        args.output / "result.json",
        {
            "kind": "actual-recorded-capped-rigid-pose-vs-authored-joint-kinematics",
            "identity": runtime_identity(),
            "mujoco_version": mujoco.__version__,
            "reference": file_id(args.reference / "result.json"),
            "binding": file_id(args.binding / "start.json"),
            "kinematic_model": file_id(args.mjcf),
            "case": binding_receipt["case"],
            "recorded_control_observations": len(times),
            "last_recorded_time_s": times[-1].item(),
            "predeclared_bands": {"com_position_m": args.position_band, "material_rotation_rad": args.rotation_band},
            "per_public_body": records,
            "maximum_com_difference_m": positions.max().item(),
            "maximum_rotation_difference_rad": angles.max().item(),
            "recorded_capped_geometry_tracking_passed": bool(
                positions.max() <= args.position_band and angles.max() <= args.rotation_band
            ),
            "retained_poses": pose_rows,
            "physical_reference_qualified": False,
            "scope": "Actual saved body COM/material rotations versus the authored MJCF FK at each actual saved joint coordinate. Constant frame offsets come only from the independently initialized input geometry, before any dynamic/contact step; no contact footprint, force, compliance or trajectory is fitted. This checks recorded10ms geometry, not unobserved physical-step extrema or contact/stress accuracy.",
        },
    )


if __name__ == "__main__":
    main()
