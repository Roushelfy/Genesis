"""Measure fresh-process, duplicate, restore and partial-restore continuation at near-slip."""

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..shell_comparison.rollout import controller_step, make_parser
from .artifacts import file_id, runtime_identity, write_json


def capture(workload, shell):
    data = {
        "robot_q": tensor_to_array(workload.robot.get_qpos()).copy(),
        "robot_v": tensor_to_array(workload.robot.get_dofs_velocity()).copy(),
    }
    if shell:
        data.update(
            position=tensor_to_array(workload.egg.get_verts_pos()).copy(),
            velocity=tensor_to_array(workload.egg.get_verts_vel()).copy(),
            stress=np.moveaxis(tensor_to_array(workload.observation.tensors).copy(), 1, 0),
        )
    else:
        entry = workload.scene.rigid_solver.stress_recovery.links[0]
        data.update(
            position=tensor_to_array(workload.egg.get_pos()).copy(),
            quaternion=tensor_to_array(workload.egg.get_quat()).copy(),
            velocity=tensor_to_array(workload.egg.get_vel()).copy(),
            stress=tensor_to_array(workload.link.get_stress_field(copy=True)[0]).copy(),
            invalid_steps=qd_to_numpy(entry.state.invalid_steps, transpose=True).copy(),
            step_valid=qd_to_numpy(entry.state.step_valid, transpose=True).copy(),
        )
    return data


def difference(reference, candidate, selection=slice(None)):
    rows = {}
    for name in reference:
        left, right = reference[name][selection], candidate[name][selection]
        finite = np.isfinite(left) & np.isfinite(right)
        delta = left[finite].astype(float) - right[finite].astype(float)
        denominator = np.linalg.norm(left[finite].astype(float))
        rows[name] = {
            "maximum_absolute_difference": np.abs(delta).max().item() if delta.size else None,
            "relative_l2_difference": np.linalg.norm(delta).item() / denominator if denominator > 0 else None,
            "reference_nonfinite_values": int((~np.isfinite(left)).sum()),
            "candidate_nonfinite_values": int((~np.isfinite(right)).sum()),
        }
    return rows


def main():
    parser = make_parser()
    parser.add_argument("--checkpoint-step", type=int, default=427)
    parser.add_argument("--continuation-steps", type=int, default=61)
    args = parser.parse_args()
    assert args.model in ("pressure", "intrinsic", "shell") and not args.instrumented
    assert args.output_mode == "full" and args.sim_dt == 0.01
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    shell = args.model == "shell"
    if shell:
        from ..shell_comparison.application import ShellEgg

        workload = ShellEgg(args)
    else:
        from examples.rigid.franka_egg_stress import FrankaEgg

        workload = FrankaEgg(
            args.envs,
            conditions=args.conditions,
            substeps=args.substeps,
            output_mode="full",
            native_controller=False,
            egg_geometry="mesh64",
            use_gjk_collision=False,
            enable_contact_patch=False,
            contact_model=args.model,
            intrinsic_load_partition="body_source_prior",
        )
    for _ in range(args.checkpoint_step):
        controller_step(workload, shell)
    state = workload.scene.get_state()
    checkpoint = capture(workload, shell)
    payload = {f"checkpoint_{name}": value for name, value in checkpoint.items()}
    for step in range(args.continuation_steps):
        controller_step(workload, shell)
        if step == 0:
            checkpoint_plus_one = capture(workload, shell)
    continuation = capture(workload, shell)
    controller_step(workload, shell)
    continuation_plus_one = capture(workload, shell)
    rows = []
    for repeat in range(3):
        workload.scene.reset(state=state)
        workload.tick = args.checkpoint_step
        if shell:
            workload.observation.update(workload.scene.sim.shell_solver)
        rows.append(
            {
                "case": "immediate-restore",
                "repeat": repeat,
                "differences": difference(checkpoint, capture(workload, shell)),
            }
        )
        for _ in range(args.continuation_steps):
            controller_step(workload, shell)
        candidate = capture(workload, shell)
        rows.append(
            {"case": "continuation-after-restore", "repeat": repeat, "differences": difference(continuation, candidate)}
        )
        payload.update({f"restore{repeat}_{name}": value for name, value in candidate.items()})
    if args.envs > 1:
        before = capture(workload, shell)
        workload.scene.reset(state=state, envs_idx=[0])
        if shell:
            workload.observation.update(workload.scene.sim.shell_solver)
        after = capture(workload, shell)
        rows.extend(
            [
                {"case": "partial-restore-selected", "differences": difference(checkpoint, after, slice(0, 1))},
                {"case": "partial-restore-unselected", "differences": difference(before, after, slice(1, None))},
            ]
        )
        # Scene state does not own this example's controller clock. Shift only
        # the restored environment's delay to replay its checkpoint target;
        # leave the continuing environments' clocks/targets unchanged.
        workload.delays[0] += args.continuation_steps
        workload.device_delays = torch.as_tensor(workload.delays, dtype=gs.tc_float, device=gs.device)
        controller_step(workload, shell)
        after = capture(workload, shell)
        rows.extend(
            [
                {
                    "case": "partial-restore-selected-continuation",
                    "differences": difference(checkpoint_plus_one, after, slice(0, 1)),
                },
                {
                    "case": "partial-restore-unselected-continuation",
                    "differences": difference(continuation_plus_one, after, slice(1, None)),
                },
            ]
        )
        payload.update({f"partial_continuation_{name}": value for name, value in after.items()})
    with np.load(args.conditions) as bank:
        period = len(bank["position"])
    if args.envs >= 2 * period:
        rows.append(
            {
                "case": "duplicate-inputs-within-batch",
                "differences": difference(
                    {n: v[:period] for n, v in checkpoint.items()},
                    {n: v[period : 2 * period] for n, v in checkpoint.items()},
                ),
            }
        )
    np.savez_compressed(args.output / "states.npz", **payload)
    write_json(
        args.output / "result.json",
        {
            "kind": "near-slip-public-lifecycle",
            "identity": runtime_identity(),
            "args": {key: str(value) if hasattr(value, "as_posix") else value for key, value in vars(args).items()},
            "conditions": file_id(args.conditions),
            "checks": rows,
            "scope": "Original capped controller and public Scene state. Native stress fields intentionally invalidate immediately after restore and must recover on continuation; NaNs at that boundary are counted explicitly. Shell coarse/warm state is not repaired. Partial continuation aligns the selected controller clock independently of unselected environments. Each process preserves actual results, including continuation defects; no accuracy qualification is inferred.",
        },
    )


if __name__ == "__main__":
    main()
