"""Untimed live comparison of rigid-only and intrinsic recovery dynamics."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import tensor_to_array


def state(workload):
    return np.concatenate(
        [
            tensor_to_array(value).reshape(workload.n_envs, -1)
            for value in (
                workload.robot.get_dofs_position(),
                workload.robot.get_dofs_velocity(),
                workload.egg.get_pos(),
                workload.egg.get_quat(),
                workload.egg.get_vel(),
                workload.egg.get_ang(),
            )
        ],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=640)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--egg-geometry", choices=("urdf", "mesh64"), default="mesh64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [*Path("genesis/engine/solvers/rigid/stress").glob("*.py"), Path("examples/rigid/franka_egg_stress.py")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    kwargs = {
        "n_envs": args.envs,
        "seed": args.seed,
        "varied": True,
        "egg_geometry": args.egg_geometry,
        "fused_rigid_post": False,
    }
    rigid = FrankaEgg(stress=False, **kwargs)
    recovered = FrankaEgg(stress=False, **kwargs)
    inertial = ET.parse("examples/rigid/assets/hollow_egg/level1/egg_shell.urdf").find("link/inertial")
    expected_mass = float(inertial.find("mass").get("value"))
    expected_com = np.array([float(value) for value in inertial.find("origin").get("xyz").split()])
    values = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
    expected_inertia = np.array(
        [
            [values["ixx"], values["ixy"], values["ixz"]],
            [values["ixy"], values["iyy"], values["iyz"]],
            [values["ixz"], values["iyz"], values["izz"]],
        ]
    )
    for workload in (rigid, recovered):
        np.testing.assert_allclose(tensor_to_array(workload.link.get_mass()), expected_mass, rtol=0, atol=1e-16)
        np.testing.assert_allclose(
            tensor_to_array(workload.egg.get_links_COM()).reshape(3), expected_com, rtol=0, atol=1e-16
        )
        rotation = Rotation.from_quat(np.asarray(workload.link.desc.inertial_quat)[[1, 2, 3, 0]]).as_matrix()
        authored_inertia = rotation @ tensor_to_array(workload.egg.get_links_inertia()).reshape(3, 3) @ rotation.T
        np.testing.assert_allclose(authored_inertia, expected_inertia, rtol=0, atol=1e-16)
    records, completed = [], 0
    initial = [state(rigid), state(recovered)]
    np.testing.assert_allclose(*initial, rtol=0, atol=1e-12)
    try:
        for tick in range(args.steps):
            rigid.step()
            recovered.step()
            recovered.scene.rigid_solver.check_errno()
            actual, reference = state(recovered), state(rigid)
            per_env = np.abs(actual - reference).max(axis=1)
            records.append(
                {"tick": tick, "maximum_state_error_per_env": per_env.tolist(), "reset_count": recovered.reset_count}
            )
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
            completed += 1
            if tick % 100 == 0:
                print(json.dumps(records[-1]), flush=True)
    finally:
        np.savez_compressed(args.output / "final-state.npz", rigid=state(rigid), recovered=state(recovered))
        summary = {
            "scope": __doc__,
            "arguments": vars(args) | {"output": str(args.output)},
            "completed_steps": completed,
            "records": records,
            "initial_state_maximum_error": float(np.max(np.abs(initial[0] - initial[1]))),
            "source_sha256": hashes,
            "source_sha256_after": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
