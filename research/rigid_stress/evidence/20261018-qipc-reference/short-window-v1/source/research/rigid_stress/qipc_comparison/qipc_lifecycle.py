"""Test public QIPC checkpoint/continuation and full reset with the actual capped reference inputs."""

import argparse
import json
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from qipc import tetmesh
from qipc.geometry import ground

from .artifacts import file_id, runtime_identity, write_json
from .reference import ReferenceConfig, SolidObserver, solver_statistics
from .reference_grasp import configure_contact
from .robot_inputs import robot_target
from .servo import CappedServo, EqualityParameters


def difference(expected, actual):
    result = {}
    for name, first in expected.items():
        second = actual[name]
        finite = np.isfinite(first) & np.isfinite(second)
        delta = first[finite] - second[finite]
        denominator = np.linalg.norm(first[finite])
        result[name] = {
            "maximum_absolute_difference": np.abs(delta).max().item() if delta.size else None,
            "relative_l2_difference": np.linalg.norm(delta).item() / denominator if denominator else None,
            "nonfinite_expected": int((~np.isfinite(first)).sum()),
            "nonfinite_actual": int((~np.isfinite(second)).sum()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint-time", type=float, default=1.2)
    parser.add_argument("--continuation-time", type=float, default=0.02)
    parser.add_argument("--reset-comparison-time", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    inputs = json.loads((args.reference / "inputs.json").read_text())
    config = ReferenceConfig(**inputs["config"])
    case = inputs["case"]
    with np.load(inputs["conditions"]["path"]) as saved:
        bank = {key: saved[key] for key in saved.files}
    with np.load(args.reference / "rest.npz") as saved:
        world, tets = saved["vertices_world_m"], saved["tetrahedra"]
    scene = config.scene()
    scene.workspace = args.output / "workspace"
    robot = scene.add_mjcf(
        args.reference / "panda-bank-initial.xml",
        link_type="rigid",
        minimal_coordinates=True,
        authored_passive_terms=True,
        enable_controller=True,
    )
    equality = EqualityParameters.from_native(inputs["equality_constants"]["path"])
    equality.attach(robot, config.dt)
    egg = tetmesh(world, tets)
    config.material().apply_to(
        egg, youngs_modulus=config.young, poissons_ratio=config.poisson, mass_density=config.density
    )
    scene.geometries.create("egg", egg)
    floor = ground(height=0.0, N=(0, 0, 1))
    scene.geometries.create("ground", floor)
    configure_contact(scene, robot, floor, config.friction)
    scene.init()
    with torch.cuda.stream(scene.solver.cuda_stream):
        observer = SolidObserver.allocate(scene, config)
        servo = CappedServo.allocate(robot, config.dt, equality)
    torch.cuda.synchronize()
    ticks_per_control = round(0.01 / config.dt)

    def capture():
        with torch.cuda.stream(scene.solver.cuda_stream):
            field = observer.update()
        torch.cuda.synchronize()
        return {
            "q": servo.q.cpu().numpy().copy(),
            "q_velocity": servo.velocity.cpu().numpy().copy(),
            "positions_m": scene.finite_element.x.cpu().numpy().copy(),
            "velocities_m_per_s": scene.finite_element.velocities.cpu().numpy().copy(),
            "cauchy_pa": field.cauchy_pa.cpu().numpy().copy(),
        }

    def advance(step):
        target, cap, _ = robot_target(bank, case, step // ticks_per_control)
        with torch.cuda.stream(scene.solver.cuda_stream):
            previous = scene.finite_element.velocities.clone()
            servo.push(servo.q.new_tensor(target), servo.q.new_tensor(cap))
        scene.step()
        with torch.cuda.stream(scene.solver.cuda_stream):
            servo.update()
        state = observer.read_step(previous)
        metrics = state.metrics()
        stats = solver_statistics(scene)
        assert metrics["all_fields_finite"] and stats["newton"] < config.max_newton
        return {"simulation_step": step, **metrics, **stats}

    checkpoint = round(args.checkpoint_time / config.dt)
    length = round(args.continuation_time / config.dt)
    reset_length = round(args.reset_comparison_time / config.dt)
    assert 0 < reset_length < checkpoint and 0 < length
    checks, payload = [], {}
    completed = False
    try:
        with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
            for step in range(checkpoint):
                row = advance(step)
                journal.write(json.dumps({"case": "first-history", **row}, allow_nan=False) + "\n")
                if step + 1 == reset_length:
                    initial_prefix = capture()
            expected_checkpoint = capture()
            dump = scene.dump()
            servo_state = {
                item.name: getattr(servo, item.name).clone()
                for item in fields(servo)
                if isinstance(getattr(servo, item.name), torch.Tensor)
            }
            for step in range(checkpoint, checkpoint + length):
                row = advance(step)
                journal.write(json.dumps({"case": "first-continuation", **row}, allow_nan=False) + "\n")
            expected_continuation = capture()
            payload.update({f"checkpoint_{name}": value for name, value in expected_checkpoint.items()})
            payload.update({f"continuation_{name}": value for name, value in expected_continuation.items()})
            for repeat in range(2):
                scene.recover(checkpoint)
                for name, value in servo_state.items():
                    setattr(servo, name, value.clone())
                checks.append(
                    {
                        "case": "immediate-public-recover",
                        "repeat": repeat,
                        "differences": difference(expected_checkpoint, capture()),
                    }
                )
                for step in range(checkpoint, checkpoint + length):
                    row = advance(step)
                    journal.write(json.dumps({"case": f"recover{repeat}-continuation", **row}, allow_nan=False) + "\n")
                candidate = capture()
                checks.append(
                    {
                        "case": "continuation-after-public-recover",
                        "repeat": repeat,
                        "differences": difference(expected_continuation, candidate),
                    }
                )
                payload.update({f"recover{repeat}_{name}": value for name, value in candidate.items()})
            scene.reset()
            with torch.cuda.stream(scene.solver.cuda_stream):
                servo = CappedServo.allocate(robot, config.dt, equality)
            torch.cuda.synchronize()
            for step in range(reset_length):
                row = advance(step)
                journal.write(json.dumps({"case": "full-reset-continuation", **row}, allow_nan=False) + "\n")
            checks.append({"case": "full-reset-continuation", "differences": difference(initial_prefix, capture())})
            completed = True
    finally:
        np.savez_compressed(args.output / "states.npz", **payload)
        write_json(
            args.output / "result.json",
            {
                "kind": "capped-qipc-public-lifecycle",
                "identity": runtime_identity(),
                "inputs": file_id(args.reference / "inputs.json"),
                "config": inputs["config"],
                "completed": completed,
                "checks": checks,
                "checkpoint_time_s": args.checkpoint_time,
                "continuation_time_s": args.continuation_time,
                "reset_comparison_time_s": args.reset_comparison_time,
                "intrusive_seconds": perf_counter() - started,
                "reference_qualified": False,
                "scope": "A single capped scene uses the same authored controller/equality/passives and actual Scene.dump/recover/reset. The Python controller's tensor state and clock are restored explicitly; no missing friction state is reconstructed from a trajectory snapshot. This bounded contact-onset lifecycle check is separate from complete-grasp uncertainty and native/Shell near-slip batch checks. The scene has no replicated native ownership, so no QIPC batch/partial-reset claim is made.",
            },
        )


if __name__ == "__main__":
    main()
