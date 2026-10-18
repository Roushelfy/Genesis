"""Audit the pinned integration candidate's actual Franka inertia and capped servo without contact."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import qipc
import qipc._src.native.solver.qipc_solver_ext as extension

from .artifacts import file_id, runtime_identity, write_json
from .reference import ReferenceConfig, solver_statistics
from .robot_inputs import CAP, JOINT_NAMES, prepare_mjcf, robot_target
from .servo import CappedServo, EqualityParameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--equality-constants", type=Path)
    parser.add_argument("--bank-schedule", action="store_true")
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.conditions) as bank:
        bank = {key: bank[key] for key in bank.files}
    initial = bank["initial_joints"][args.case]
    generated = prepare_mjcf(args.mjcf, initial, args.output)
    config = ReferenceConfig(
        dt=args.dt,
        mas_precision="double",
        velocity_tolerance=1e-7 if args.bank_schedule else 1e-6,
        linear_tolerance=1e-5 if args.bank_schedule else 1e-9,
    )
    scene = config.scene(gravity=(0, 0, -args.gravity), contact=False)
    robot = scene.add_mjcf(
        generated, link_type="rigid", minimal_coordinates=True, authored_passive_terms=True, enable_controller=True
    )
    equality = EqualityParameters.from_native(args.equality_constants) if args.equality_constants else None
    if equality is not None:
        equality.attach(robot, config.dt)
    assert robot.joint_names == JOINT_NAMES
    scene.init()
    with torch.cuda.stream(scene.solver.cuda_stream):
        servo = CappedServo.allocate(robot, config.dt, equality)
    torch.cuda.synchronize()
    np.testing.assert_allclose(servo.q.cpu().numpy(), initial, rtol=0, atol=1e-12)
    authored = [
        {
            "name": slot.name,
            "armature": slot.geometry.vertices["armature"].cpu().tolist(),
            "passive_damping": slot.geometry.vertices["passive_damping"].cpu().tolist(),
        }
        for slot in robot.joints
    ]
    body_names = [None] * scene.rigid_body.n_public_bodies
    for slot in robot.links:
        index = slot.geometry.meta["rigid_body_offset"].cpu()[0].item()
        assert 0 <= index < len(body_names) and body_names[index] is None
        body_names[index] = slot.name
    assert all(name is not None for name in body_names)
    write_json(
        args.output / "start.json",
        {
            "qipc_import": qipc.__file__,
            "identity": runtime_identity(),
            "native": file_id(extension.__file__),
            "condition_bank": file_id(args.conditions),
            "config": asdict(config),
            "case": args.case,
            "joint_names": robot.joint_names,
            "initial_q": servo.q.tolist(),
            "public_rigid_link_names": body_names,
            "authored": authored,
            "equality": asdict(equality) if equality is not None else None,
            "public_rigid_mass_kg": scene.rigid_body.mass[: scene.rigid_body.n_public_bodies].tolist(),
            "public_rigid_com_world_m": scene.rigid_body.t[: scene.rigid_body.n_public_bodies].tolist(),
            "public_rigid_material_quaternion_wxyz": scene.rigid_body.quat[: scene.rigid_body.n_public_bodies].tolist(),
            "public_rigid_inertia_tensor": scene.rigid_body.inertia_tensor[: scene.rigid_body.n_public_bodies].tolist(),
            "servo_adapter": "External clip(kp*(target-q)-kv*v, caps) + kv*v and original implicit kv; mirrors Genesis approximate_implicitfast frozen-start PD force plus its unclipped damping Jacobian. Passive damping and armature remain separate authored terms.",
        },
    )
    if args.layout_only:
        write_json(
            args.output / "result.json",
            {
                "kind": "initial-QIPC-public-rigid-frame-binding",
                "case": args.case,
                "start": file_id(args.output / "start.json"),
                "scope": "Original authored geometry at the input initial configuration; no physical step, contact or fitted dynamic response. This binds coordinate frames for independently checking saved actual body poses against joint kinematics.",
            },
        )
        return
    target = servo.q.new_tensor(initial + np.array([0.001, -0.001, 0.002, -0.002, 0.001, 0.001, -0.001, -0.06, -0.03]))
    cap = servo.q.new_tensor(CAP)
    control_stride = round(0.01 / args.dt)
    assert control_stride > 0 and abs(control_stride * args.dt - 0.01) < 1e-12
    resets = 0
    history = []
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for step in range(args.steps):
            phase = None
            if args.bank_schedule:
                tick = step // control_stride
                if (
                    step % control_stride == 0
                    and tick > bank["delays"][args.case]
                    and (tick - bank["delays"][args.case]) % 600 == 0
                ):
                    scene.reset()
                    with torch.cuda.stream(scene.solver.cuda_stream):
                        servo = CappedServo.allocate(robot, args.dt, equality)
                    torch.cuda.synchronize()
                    resets += 1
                command, limits, phase = robot_target(bank, args.case, tick)
                target, cap = servo.q.new_tensor(command), servo.q.new_tensor(limits)
            with torch.cuda.stream(scene.solver.cuda_stream):
                servo.push(target, cap)
            scene.step()
            with torch.cuda.stream(scene.solver.cuda_stream):
                servo.update()
            torch.cuda.synchronize()
            row = {
                "step": step,
                "time_s": (step + 1) * args.dt,
                "reset_epoch": resets,
                "phase": float(phase) if phase is not None else None,
                "target_q": target.tolist(),
                "force_caps": cap.tolist(),
                "q": servo.q.tolist(),
                "velocity": servo.velocity.tolist(),
                "clipped_pd_force": servo.clipped_force.tolist(),
                "effective_pd_force_including_jacobian": servo.effective_actuator_force.tolist(),
                "equality_inertia_coefficient_relative_difference": servo.equality_coefficient_difference.item(),
                **solver_statistics(scene),
                "status": scene.solver.last_step_status,
            }
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            history.append(row)
    write_json(
        args.output / "result.json",
        {
            "kind": "Franka-no-contact-servo-control",
            "case": args.case,
            "config": asdict(config),
            "target": target.tolist(),
            **({"history": history} if not args.bank_schedule else {}),
            "contact_qualified": False,
            "gravity_m_per_s2": args.gravity,
            "bank_schedule": args.bank_schedule,
            "completed_requested_history": len(history) == args.steps,
            "requested_steps": args.steps,
            "executed_steps": len(history),
            "resets": resets,
            "journal": file_id(args.output / "steps.jsonl"),
            "scope": "No egg/contact. Bank-schedule mode retains original targets/gains/caps,10ms cadence, passive/rotor inertia, equality mapping and reset. This extends the earlier20ms control but does not itself qualify contact/stress.",
        },
    )


if __name__ == "__main__":
    main()
