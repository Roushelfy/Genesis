"""Actual native Franka inertia and PD responses for comparison with the pinned QIPC forest."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .robot_inputs import CAP, JOINT_NAMES, KP, KV, robot_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--bank-schedule", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    rigid_options = {"enable_collision": False}
    if args.bank_schedule:
        rigid_options.update(
            batch_dofs_info=True,
            friction_cone=gs.friction_cone.elliptic,
            contact_resolution=gs.contact_resolution.convex,
            iterations=100,
            tolerance=1e-12,
            contact_pruning_tolerance=None,
            enable_torsional_friction=False,
            enable_rolling_friction=False,
            use_hibernation=False,
            use_gjk_collision=False,
            enable_contact_patch=False,
        )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, gravity=(0, 0, -args.gravity)),
        rigid_options=gs.options.RigidOptions(**rigid_options),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=args.mjcf))
    scene.build(n_envs=1)
    with np.load(args.conditions) as bank:
        bank = {key: bank[key] for key in bank.files}
    initial = bank["initial_joints"][args.case]
    robot.set_qpos(initial)
    robot.set_dofs_kp(KP)
    robot.set_dofs_kv(KV)
    robot.set_dofs_force_range(-CAP, CAP)
    write_json(
        args.output / "start.json",
        {
            "genesis_import": gs.__file__,
            "identity": runtime_identity(),
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
    if not args.bank_schedule:
        robot.control_dofs_position(target)
    reset_state = scene.get_state() if args.bank_schedule else None
    control_stride = round(0.01 / args.dt)
    assert control_stride > 0 and abs(control_stride * args.dt - 0.01) < 1e-12
    resets = 0
    history = []
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for step in range(args.steps):
            phase = None
            cap = CAP
            if args.bank_schedule:
                tick = step // control_stride
                if step % control_stride == 0:
                    if tick > bank["delays"][args.case] and (tick - bank["delays"][args.case]) % 600 == 0:
                        scene.reset(state=reset_state)
                        resets += 1
                    target, cap, phase = robot_target(bank, args.case, tick)
                    robot.control_dofs_position(target[:7], np.arange(7))
                    robot.control_dofs_position(target[7:], np.array([7, 8]))
                    robot.set_dofs_force_range(-cap[7:], cap[7:], np.array([7, 8]))
                else:
                    target, cap, phase = robot_target(bank, args.case, tick)
            scene.step()
            row = {
                "step": step,
                "time_s": (step + 1) * args.dt,
                "reset_epoch": resets,
                "phase": float(phase) if phase is not None else None,
                "target_q": target.tolist(),
                "force_caps": cap.tolist(),
                "q": tensor_to_array(robot.get_qpos())[0].tolist(),
                "velocity": tensor_to_array(robot.get_dofs_velocity())[0].tolist(),
                "clipped_pd_force": tensor_to_array(robot.get_dofs_control_force())[0].tolist(),
            }
            assert np.isfinite(row["q"]).all() and np.isfinite(row["velocity"]).all()
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            history.append(row)
            if not args.bank_schedule:
                write_json(args.output / "progress.json", {"history": history})
    torch.cuda.synchronize()
    if args.bank_schedule:
        scene.rigid_solver.check_errno()
    write_json(
        args.output / "result.json",
        {
            "kind": "native-Franka-no-contact-servo-control",
            "dt": args.dt,
            "case": args.case,
            "target": target.tolist(),
            "joint_names": JOINT_NAMES,
            **({"history": history} if not args.bank_schedule else {}),
            "gravity_m_per_s2": args.gravity,
            "bank_schedule": args.bank_schedule,
            "completed_requested_history": len(history) == args.steps,
            "requested_steps": args.steps,
            "executed_steps": len(history),
            "resets": resets,
            "journal": file_id(args.output / "steps.jsonl"),
            "scope": "No egg/contact. Bank-schedule mode uses the original gains, limits,10ms commands, weak/release schedule, reset and matched native rigid options. It qualifies robot dynamics/control only; it cannot qualify robot-egg contact or stress.",
        },
    )


if __name__ == "__main__":
    main()
