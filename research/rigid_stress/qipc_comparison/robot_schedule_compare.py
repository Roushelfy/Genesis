"""Compare complete unloaded native/QIPC original capped-command schedules and reset."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .history_audit import returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--qipc", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--arm-band", type=float, default=1e-4)
    parser.add_argument("--finger-band", type=float, default=1e-5)
    parser.add_argument("--hand-position-band", type=float, default=5e-5)
    parser.add_argument("--hand-rotation-band", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, rows, q, velocity, epochs, times = [], [], [], [], [], []
    for folder in (args.native, args.qipc):
        receipt = json.loads((folder / "result.json").read_text())
        assert receipt["completed_requested_history"] and receipt["bank_schedule"]
        assert returncode(folder.parent / "returncode.txt") == 0
        history = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        assert len(history) == receipt["requested_steps"] == receipt["executed_steps"]
        receipts.append(receipt)
        rows.append(history)
        q.append(np.array([row["q"] for row in history]))
        velocity.append(np.array([row["velocity"] for row in history]))
        epochs.append(np.array([row["reset_epoch"] for row in history]))
        times.append(np.array([row["time_s"] for row in history]))
    assert receipts[0]["case"] == receipts[1]["case"]
    np.testing.assert_array_equal(times[0], times[1])
    np.testing.assert_array_equal(epochs[0], epochs[1])
    for key in ("target_q", "force_caps", "phase"):
        np.testing.assert_array_equal([row[key] for row in rows[0]], [row[key] for row in rows[1]])
    assert np.isfinite(q).all() and np.isfinite(velocity).all()
    difference = q[1] - q[0]
    arm = np.abs(difference[:, :7]).max(axis=1)
    finger = np.abs(difference[:, 7:]).max(axis=1)
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    assert model.nq == 9
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    assert hand >= 0
    states = [mujoco.MjData(model), mujoco.MjData(model)]
    position_error, rotation_error = [], []
    for first, second in zip(*q):
        for state, coordinates in zip(states, (first, second)):
            state.qpos[:] = coordinates
            mujoco.mj_kinematics(model, state)
        position_error.append(np.linalg.norm(states[1].xpos[hand] - states[0].xpos[hand]))
        rotation_error.append(
            Rotation.from_matrix(states[1].xmat[hand].reshape(3, 3) @ states[0].xmat[hand].reshape(3, 3).T).magnitude()
        )
    position_error, rotation_error = np.array(position_error), np.array(rotation_error)
    statistics = {}
    for name, value in (
        ("arm_coordinate_rad", arm),
        ("finger_coordinate_m", finger),
        ("authored_hand_position_m", position_error),
        ("authored_hand_rotation_rad", rotation_error),
    ):
        statistics[name] = {
            "maximum": value.max().item(),
            "maximum_time_s": times[0][value.argmax()].item(),
            "rms": np.sqrt(np.mean(value**2)).item(),
        }
    bands = {
        "arm_coordinate_rad": args.arm_band,
        "finger_coordinate_m": args.finger_band,
        "authored_hand_position_m": args.hand_position_band,
        "authored_hand_rotation_rad": args.hand_rotation_band,
    }
    statuses = [row.get("status", {}) for row in rows[1]]
    status_observed = all("newton_budget_exhausted" in row and "line_search_exhausted" in row for row in statuses)
    bad_statuses = (
        sum(row["newton_budget_exhausted"] or row["line_search_exhausted"] for row in statuses)
        if status_observed
        else None
    )
    max_pcg = receipts[1]["config"]["max_pcg"]
    pcg_observed = all("pcg_max" in row for row in rows[1])
    pcg_caps = sum(row["pcg_max"] >= max_pcg for row in rows[1]) if pcg_observed else None
    passed = (
        bad_statuses == 0 and pcg_caps == 0 and all(statistics[key]["maximum"] <= band for key, band in bands.items())
    )
    np.savez_compressed(
        args.output / "trajectories.npz",
        time_s=times[0],
        native_q=q[0],
        qipc_q=q[1],
        native_velocity=velocity[0],
        qipc_velocity=velocity[1],
        reset_epoch=epochs[0],
        arm_difference_rad=arm,
        finger_difference_m=finger,
        hand_position_difference_m=position_error,
        hand_rotation_difference_rad=rotation_error,
    )
    write_json(
        args.output / "result.json",
        {
            "kind": "complete-unloaded-native-QIPC-capped-schedule-comparison",
            "identity": runtime_identity(),
            "case": receipts[0]["case"],
            "sources": [
                file_id(folder / name)
                for folder in (args.native, args.qipc)
                for name in ("result.json", "steps.jsonl", "start.json")
            ],
            "kinematic_model": file_id(args.mjcf),
            "mujoco_version": mujoco.__version__,
            "physical_observations": len(times[0]),
            "duration_s": times[0][-1].item(),
            "resets": receipts[0]["resets"],
            "commands_caps_phases_and_reset_epochs_exactly_equal": True,
            "statistics": statistics,
            "predeclared_bands": bands,
            "qipc_status_flags_observed": status_observed,
            "qipc_failed_status_steps": bad_statuses,
            "qipc_pcg_flags_observed": pcg_observed,
            "qipc_pcg_cap_steps": pcg_caps,
            "no_contact_schedule_bands_passed": passed,
            "physical_contact_reference_qualified": False,
            "scope": "Every physical-step coordinate and authored hand FK is compared over the complete original command schedule, including reset. No egg/contact, imposed actual q or fitted reaction force is present. Predeclared position/rotation bands test unloaded control/integration matching; passing cannot qualify loaded contact, friction, FEM stress or continuum peaks. Unknown numerical flags remain unknown and cannot pass.",
        },
    )


if __name__ == "__main__":
    main()
