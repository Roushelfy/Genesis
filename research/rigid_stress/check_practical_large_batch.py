"""Intrusive large-batch field, checkpoint, partial-reset and release checks.

Use Slurm and an immutable engine snapshot. This is numerical/lifecycle evidence,
not throughput. The final zero-load/recontact stage is a frozen mapper check;
the live controller trajectory is complete before any raw contact is changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark.rigid_stress import prepare_recovery_snapshot
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array


def fields(workload, envs):
    tensor, vm = workload.link.get_stress_field(envs_idx=envs)
    return tensor_to_array(tensor), tensor_to_array(vm), tensor_to_array(workload.link.get_max_stress(envs_idx=envs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=16384)
    parser.add_argument("--contact-block-size", type=int, default=16)
    parser.add_argument("--history", type=int, choices=(0, 4), default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.envs >= 16 and args.envs % 8 == 0
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    bank_hash = hashlib.sha256(args.conditions.read_bytes()).hexdigest()
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    workload = FrankaEgg(
        args.envs,
        1,
        True,
        varied=True,
        seed=623016,
        conditions=args.conditions,
        history_size=args.history,
        contact_model="intrinsic",
        intrinsic_support="contact",
        intrinsic_load_partition="body_source_prior",
        contact_block_size=args.contact_block_size,
        output_mode="full",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
    )
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    assert (entry.history is None) == (args.history == 0)
    reset_ids = np.array([0, 1, args.envs // 2, args.envs - 1])
    observed = np.unique(np.concatenate((np.arange(8), np.arange(args.envs - 8, args.envs), reset_ids)))
    untouched = np.ones(args.envs, dtype=bool)
    untouched[reset_ids] = False
    for _ in range(900):
        workload.step()
    solver.check_errno()
    assert qd_to_numpy(entry.state.invalid_steps, transpose=True).sum() == 0
    saved_tick, saved_resets = workload.tick, workload.reset_count
    checkpoint = solver.__getstate__()
    workload.step()
    next_peak = tensor_to_array(workload.link.get_max_stress())
    next_position = tensor_to_array(workload.egg.get_pos())
    next_fields = fields(workload, observed)
    workload.step()
    following_peak = tensor_to_array(workload.link.get_max_stress())
    following_position = tensor_to_array(workload.egg.get_pos())
    solver.__setstate__(checkpoint)
    workload.tick, workload.reset_count = saved_tick, saved_resets
    del checkpoint
    np.testing.assert_array_equal(tensor_to_array(workload.link.get_max_stress()), 0)
    assert all(np.isnan(value).all() for value in fields(workload, observed)[:2])
    workload.step()
    solver.check_errno()
    np.testing.assert_allclose(tensor_to_array(workload.egg.get_pos()), next_position, rtol=0, atol=1e-15)
    np.testing.assert_allclose(tensor_to_array(workload.link.get_max_stress()), next_peak, rtol=1e-4, atol=1e-3)
    for actual, expected in zip(fields(workload, observed), next_fields):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
    peak_before = tensor_to_array(workload.link.get_max_stress())
    displacement_before = qd_to_numpy(entry.state.displacement, transpose=True, copy=True)
    history_before = None
    if entry.history is not None:
        history_before = qd_to_numpy(entry.history.state.count, transpose=True, copy=True)
    tensor_before, vm_before, _ = fields(workload, observed)
    workload.reset(reset_ids)
    peak_after = tensor_to_array(workload.link.get_max_stress())
    np.testing.assert_array_equal(peak_after[reset_ids], 0)
    np.testing.assert_array_equal(peak_after[untouched], peak_before[untouched])
    displacement_after = qd_to_numpy(entry.state.displacement, transpose=True)
    np.testing.assert_array_equal(displacement_after[reset_ids], 0)
    np.testing.assert_array_equal(displacement_after[untouched], displacement_before[untouched])
    if entry.history is not None:
        history_after = qd_to_numpy(entry.history.state.count, transpose=True)
        np.testing.assert_array_equal(history_after[reset_ids], 0)
        np.testing.assert_array_equal(history_after[untouched], history_before[untouched])
    selected_reset = np.isin(observed, reset_ids)
    tensor_after, vm_after, _ = fields(workload, observed)
    for actual, expected in ((tensor_after, tensor_before), (vm_after, vm_before)):
        assert np.isnan(actual[selected_reset]).all()
        np.testing.assert_array_equal(actual[~selected_reset], expected[~selected_reset])
    del displacement_before, displacement_after
    workload.step()
    solver.check_errno()
    np.testing.assert_allclose(
        tensor_to_array(workload.egg.get_pos())[untouched], following_position[untouched], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        tensor_to_array(workload.link.get_max_stress())[untouched], following_peak[untouched], rtol=1e-4, atol=1e-3
    )
    observations = []
    for step in range(601):
        if step:
            workload.step()
        if step % 100 == 0:
            solver.check_errno()
            tensor, vm, peak = fields(workload, observed)
            assert np.isfinite(tensor).all() and np.isfinite(vm).all()
            np.testing.assert_array_equal(vm.max(axis=(1, 2)), peak)
            von_mises = np.sqrt(
                0.5
                * (
                    (tensor[..., 0] - tensor[..., 1]) ** 2
                    + (tensor[..., 1] - tensor[..., 2]) ** 2
                    + (tensor[..., 2] - tensor[..., 0]) ** 2
                )
                + 3 * np.sum(tensor[..., 3:] ** 2, axis=-1)
            )
            np.testing.assert_allclose(von_mises, vm, rtol=1e-4, atol=1e-3)
            residual = np.sqrt(qd_to_numpy(entry.state.residual_norm_squared, transpose=True))
            rhs = np.sqrt(qd_to_numpy(entry.state.rhs_norm_squared, transpose=True))
            fraction = residual / np.maximum(entry.options.absolute_tolerance, entry.options.tolerance * rhs)
            assert np.isfinite(fraction).all() and (fraction <= 1).all()
            np.savez_compressed(
                args.output / f"step{workload.tick:05d}.npz",
                envs=observed,
                tensor_Pa=tensor,
                von_mises_Pa=vm,
                peak_Pa=peak,
                phase=tensor_to_array(workload.phase)[observed],
            )
            observations.append({"tick": workload.tick, "maximum_complete_budget_fraction": float(fraction.max())})
    invalid = int(qd_to_numpy(entry.state.invalid_steps, transpose=True).sum())
    assert invalid == 0 and qd_to_numpy(entry.state.step_valid, transpose=True).all()
    frozen_replay = prepare_recovery_snapshot(workload)
    valid = qd_to_numpy(entry.contacts.valid, transpose=True, copy=True)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(args.envs,))
    omega.fill(0)
    kernel_begin_step(entry.state)
    entry.intrinsic.scatter(
        entry.contacts,
        entry.state,
        entry.model.info,
        entry.surface.info,
        entry.options,
        solver.stress_recovery.source_epsilon,
        solver._errno,
    )
    entry.model.recover(omega, entry.state, entry.options)
    kernel_accept(entry.state, entry.contacts, solver._errno)
    solver.check_errno()
    frozen_peak = tensor_to_array(workload.link.get_max_stress())
    frozen_tensor, frozen_vm, _ = fields(workload, observed)
    frozen_force = qd_to_numpy(entry.state.force, transpose=True, copy=True)
    entry.contacts.valid.fill(False)
    kernel_begin_step(entry.state)
    entry.intrinsic.scatter(
        entry.contacts,
        entry.state,
        entry.model.info,
        entry.surface.info,
        entry.options,
        solver.stress_recovery.source_epsilon,
        solver._errno,
    )
    entry.model.recover(omega, entry.state, entry.options)
    kernel_accept(entry.state, entry.contacts, solver._errno)
    solver.check_errno()
    for value in (entry.state.force, entry.state.stress_tensor, entry.state.von_mises, entry.state.step_peak):
        assert torch.count_nonzero(qd_to_torch(value, transpose=True, copy=False)).item() == 0
    entry.contacts.valid.from_numpy(valid.T.copy())
    kernel_begin_step(entry.state)
    entry.intrinsic.scatter(
        entry.contacts,
        entry.state,
        entry.model.info,
        entry.surface.info,
        entry.options,
        solver.stress_recovery.source_epsilon,
        solver._errno,
    )
    entry.model.recover(omega, entry.state, entry.options)
    kernel_accept(entry.state, entry.contacts, solver._errno)
    solver.check_errno()
    np.testing.assert_allclose(qd_to_numpy(entry.state.force, transpose=True), frozen_force, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(tensor_to_array(workload.link.get_max_stress()), frozen_peak, rtol=1e-4, atol=1e-3)
    for actual, expected in zip(fields(workload, observed)[:2], (frozen_tensor, frozen_vm)):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
    assert hashlib.sha256(args.conditions.read_bytes()).hexdigest() == bank_hash
    result = {
        "scope": __doc__,
        "envs": args.envs,
        "initial_contact_block_size": args.contact_block_size,
        "effective_contact_block_size": entry.intrinsic.contacts.valid.shape[0],
        "output_mode": "full",
        "history_size": args.history,
        "conditions_sha256": bank_hash,
        "observed_envs": observed.tolist(),
        "reset_envs": reset_ids.tolist(),
        "warmup_steps": 900,
        "following_live_steps": 600,
        "observations": observations,
        "frozen_replay": frozen_replay,
        "invalid_live_steps": invalid,
        "checkpoint_passed": True,
        "partial_reset_and_unselected_following_step_passed": True,
        "complete_frozen_release_and_recontact_passed": True,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid),
        "passed": True,
        "goal_completion": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"envs": args.envs, "passed": True, "invalid_live_steps": invalid}))


if __name__ == "__main__":
    main()
