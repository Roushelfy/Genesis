"""Actual native Franka inertia and PD responses for comparison with the pinned QIPC forest."""

import argparse
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import file_id, write_json
from .robot_inputs import CAP, JOINT_NAMES, KP, KV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, gravity=(0, 0, -args.gravity)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=args.mjcf))
    scene.build(n_envs=1)
    with np.load(args.conditions) as bank:
        initial = bank["initial_joints"][args.case]
    robot.set_qpos(initial)
    robot.set_dofs_kp(KP)
    robot.set_dofs_kv(KV)
    robot.set_dofs_force_range(-CAP, CAP)
    write_json(
        args.output / "start.json",
        {
            "genesis_import": gs.__file__,
            "xml": file_id(args.mjcf),
            "bank": file_id(args.conditions),
            "armature": tensor_to_array(robot.get_dofs_armature()).tolist(),
            "damping": tensor_to_array(robot.get_dofs_damping()).tolist(),
            "kp": tensor_to_array(robot.get_dofs_kp()).tolist(),
            "kv": tensor_to_array(robot.get_dofs_kv()).tolist(),
            "force_range": [tensor_to_array(value).tolist() for value in robot.get_dofs_force_range()],
            "link_names": [link.name for link in robot.links],
            "mass_kg": tensor_to_array(robot.get_links_mass()).tolist(),
            "inertia": tensor_to_array(robot.get_links_inertia()).tolist(),
            "link_origin_world_m": tensor_to_array(robot.get_links_pos()).tolist(),
            "link_origin_quat": tensor_to_array(robot.get_links_quat()).tolist(),
            "inertial_local_com": tensor_to_array(robot.get_links_COM()).tolist(),
            "geometry_to_link": [
                {"geometry": geom.idx, "link": link.name} for link in robot.links for geom in link.geoms
            ],
            "admitted_geometry_pairs": scene.rigid_solver.collider._valid_collision_pairs.tolist(),
            "dofs_inverse_weight": qd_to_numpy(scene.rigid_solver.dyn_info.dofs.invweight, transpose=True).tolist(),
            "equality_solver_parameters": qd_to_numpy(
                scene.rigid_solver.dyn_info.equalities.sol_params, transpose=True
            ).tolist(),
        },
    )
    target = initial + np.array([0.001, -0.001, 0.002, -0.002, 0.001, 0.001, -0.001, -0.06, -0.03])
    robot.control_dofs_position(target)
    history = []
    for step in range(args.steps):
        scene.step()
        history.append(
            {
                "step": step,
                "q": tensor_to_array(robot.get_qpos())[0].tolist(),
                "velocity": tensor_to_array(robot.get_dofs_velocity())[0].tolist(),
                "clipped_pd_force": tensor_to_array(robot.get_dofs_control_force())[0].tolist(),
            }
        )
        write_json(args.output / "progress.json", {"history": history})
    torch.cuda.synchronize()
    write_json(
        args.output / "result.json",
        {
            "kind": "native-Franka-no-contact-servo-control",
            "dt": args.dt,
            "target": target.tolist(),
            "joint_names": JOINT_NAMES,
            "history": history,
            "gravity_m_per_s2": args.gravity,
        },
    )


if __name__ == "__main__":
    main()
