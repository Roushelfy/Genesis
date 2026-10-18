"""Freeze a common joint-linear motion and exact authored-link FK, independent of held-out reference stresses."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .robot_inputs import prepare_mjcf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--sample-dt", type=float, default=0.00025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.conditions) as bank:
        initial = bank["initial_joints"][0]
    rows = [json.loads(line) for line in (args.rollout / "steps.jsonl").read_text().splitlines()]
    assert rows[-1]["time_s"] >= args.duration
    coarse_time = np.array([0.0] + [row["time_s"] for row in rows])
    coarse_q = np.vstack((initial, np.array([row["q"] for row in rows])))
    time = np.arange(round(args.duration / args.sample_dt) + 1) * args.sample_dt
    q = np.stack([np.interp(time, coarse_time, coarse_q[:, axis]) for axis in range(9)], axis=1)
    generated = prepare_mjcf(args.mjcf, initial, args.output)
    model = mujoco.MjModel.from_xml_path(str(generated))
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(model.nbody)]
    positions = np.empty((len(time), model.nbody, 3))
    quaternions = np.empty((len(time), model.nbody, 4))
    for index in range(len(time)):
        data.qpos[:] = q[index]
        mujoco.mj_kinematics(model, data)
        positions[index], quaternions[index] = data.xpos, data.xquat
    path = args.output / "motion.npz"
    np.savez_compressed(
        path, time_s=time, q=q, link_names=names, link_origin_world_m=positions, link_quaternion_wxyz=quaternions
    )
    write_json(
        args.output / "result.json",
        {
            "kind": "common-prescribed-motion-input",
            "identity": runtime_identity(),
            "source_rollout": file_id(args.rollout / "steps.jsonl"),
            "condition_bank": file_id(args.conditions),
            "motion": file_id(path),
            "duration_s": args.duration,
            "sampling_dt_s": args.sample_dt,
            "mujoco_version": mujoco.__version__,
            "actuator_caps_overridden_in_this_protocol": True,
            "contract": "Piecewise-linear joint coordinates from one complete rigid capped rollout; exact MuJoCo FK at every fine sample. Contains the entire first approach/squeeze/lift/lateral/weak/release/recontact history before the first episode reset. Input chosen without held-out QIPC stress or footprint fitting.",
        },
    )


if __name__ == "__main__":
    main()
