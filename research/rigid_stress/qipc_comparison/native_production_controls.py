"""Record native scene conformance and replay changing contacts from device-resident inputs.

This is an intrusive control, separate from primary scene timing. Replay includes
restoring recorded device buffers and the unchanged native recovery/acceptance
path, but excludes physical dynamics and host field export.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark.rigid_stress import restore_frame
from genesis.engine.solvers.base_solver import StateChange
from genesis.utils.array_class import DataKind
from genesis.utils.misc import qd_to_torch, tensor_to_array

from .artifacts import file_id, runtime_identity, write_json


@dataclass
class ReplayBuffer:
    name: str
    live: torch.Tensor
    frames: torch.Tensor


def replay_buffers(workload, steps):
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    collider = solver.collider.collider_state
    contact = collider.contact_data
    witness = contact.witness
    links = solver.dyn_state.links
    fields = (
        ("n_contacts", collider.n_contacts),
        ("contact_sort_idx", collider.contact_sort_idx),
        ("link_a", contact.link_a),
        ("link_b", contact.link_b),
        ("geom_a", contact.geom_a),
        ("geom_b", contact.geom_b),
        ("pair_idx", contact.pair_idx),
        ("position", contact.pos),
        ("force", contact.force),
        ("normal", contact.normal),
        ("friction", contact.friction),
        ("witness_method", witness.method),
        ("witness_vertex_a", witness.vertex_a),
        ("witness_vertex_b", witness.vertex_b),
        ("witness_point_a", witness.point_a),
        ("witness_point_b", witness.point_b),
        ("radius", entry.contacts.radius),
        ("frame_position", entry.contacts.frame_position),
        ("frame_quaternion", entry.contacts.frame_quaternion),
        ("omega", entry.omega),
    )
    buffers = []
    for name, field in fields:
        view = qd_to_torch(field, transpose=True, copy=False)
        buffers.append(ReplayBuffer(name, view, view.new_empty((steps, *view.shape))))
    for name, field in (
        ("applied_angular", links.cfrc_applied_ang),
        ("applied_linear", links.cfrc_applied_vel),
        ("coupling_angular", links.cfrc_coupling_ang),
        ("coupling_linear", links.cfrc_coupling_vel),
    ):
        view = qd_to_torch(field, transpose=True, copy=False)[:, entry.link.idx]
        buffers.append(ReplayBuffer(name, view, view.new_empty((steps, *view.shape))))
    return buffers


def motion(workload):
    return torch.cat(
        (
            workload.egg.get_pos(),
            workload.egg.get_quat(),
            workload.egg.get_vel(),
            workload.egg.get_ang(),
            workload.robot.get_dofs_position(),
            workload.robot.get_dofs_velocity(),
        ),
        dim=1,
    )


def restore(workload, buffers, tick, resets):
    if tick in resets:
        workload.scene.rigid_solver.stress_recovery.reset(StateChange.GEOMETRY, resets[tick])
    for buffer in buffers:
        buffer.live.copy_(buffer.frames[tick])
    # Association reads the saved pre-integration body frame, not the final
    # post-integration pose. This is the existing frozen-snapshot restoration.
    restore_frame(workload)


def capture_physics(workload, path):
    solver = workload.scene.rigid_solver
    arrays, inventory, configs = {}, [], {}
    for item in solver.data:
        if item.name.startswith("stress.") or item.kind == DataKind.SCRATCH:
            continue
        if item.kind == DataKind.CONFIG:
            configs[item.name] = {
                key: value.item() if isinstance(value, np.generic) else value for key, value in vars(item.value).items()
            }
        elif 0 not in item.value.shape:
            array = tensor_to_array(qd_to_torch(item.value, copy=False))
            arrays[item.name] = array
            inventory.append(
                {"name": item.name, "kind": item.kind.name, "shape": list(array.shape), "dtype": str(array.dtype)}
            )
    np.savez_compressed(path, **arrays)
    write_json(
        path.with_suffix(".json"),
        {
            "arrays": inventory,
            "configs": configs,
            "gs_use_ndarray": gs.use_ndarray,
            "gs_use_zerocopy": gs.use_zerocopy,
            "gs_use_deterministic_algorithms": gs.use_deterministic_algorithms,
            "rigid_options": solver._options.model_dump(mode="json"),
            "observed_link_index": workload.link.idx,
            "snapshot": file_id(path),
        },
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--method", choices=("rigid", "dpr", "scr"), required=True)
    cli.add_argument("--envs", type=int, required=True)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--output-mode", choices=("max", "full"), default="max")
    cli.add_argument("--steps", type=int, default=1200)
    cli.add_argument("--warmup", type=int, default=900)
    cli.add_argument("--field-stride", type=int, default=120)
    cli.add_argument("--replay", action="store_true")
    cli.add_argument("--mechanics-only", action="store_true", help="Build stress but leave manual updates unrequested")
    cli.add_argument("--rigid-post-fusion", choices=("auto", "on", "off"), default="auto")
    cli.add_argument("--repetitions", type=int, default=3)
    cli.add_argument("--contact-block-size", type=int, default=16)
    cli.add_argument("--deterministic", choices=("auto", "on", "off"), default="auto")
    cli.add_argument("--construction-snapshots", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if min(args.envs, args.steps, args.field_stride, args.repetitions) < 1 or args.warmup < 0:
        cli.error("Counts must be positive, with nonnegative warmup.")
    if args.replay and (args.method == "rigid" or args.mechanics_only):
        cli.error("Recovery replay requires a stress method.")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "args": vars(args) | {"conditions": str(args.conditions), "output": str(args.output)},
            "conditions": file_id(args.conditions),
        },
    )
    gs.init(
        backend=gs.gpu,
        precision="64",
        seed=623016,
        logging_level="warning",
        use_deterministic_algorithms={"auto": None, "on": True, "off": False}[args.deterministic],
    )
    workload = FrankaEgg(
        args.envs,
        level=1,
        stress=args.method != "rigid",
        seed=623016,
        varied=True,
        conditions=args.conditions,
        history_size=0,
        output_mode=args.output_mode,
        stress_update_mode="manual" if args.mechanics_only else "every_step",
        fused_rigid_post={"auto": None, "on": True, "off": False}[args.rigid_post_fusion],
        contact_model="intrinsic" if args.method == "scr" else "pressure",
        contact_block_size=args.contact_block_size,
        intrinsic_load_partition="body_source_prior",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
    )
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0] if workload.stress else None
    observed_stress = workload.stress and not args.mechanics_only
    if args.construction_snapshots:
        capture_physics(workload, args.output / "constructed.npz")
    with torch.inference_mode():
        for _ in range(args.warmup):
            workload.step()
        qd.sync()
        solver.check_errno()
        workload.restart()
        if args.construction_snapshots:
            capture_physics(workload, args.output / "restarted.npz")
        buffers = replay_buffers(workload, args.steps) if args.replay else []
        resets = {0: torch.arange(args.envs, dtype=gs.tc_int, device=gs.device)}
        for tick in range(1, args.steps):
            ids = np.flatnonzero((tick > workload.delays) & ((tick - workload.delays) % 600 == 0))
            if len(ids):
                resets[tick] = torch.as_tensor(ids, dtype=gs.tc_int, device=gs.device)
        states = motion(workload).new_empty((args.steps, args.envs, 31))
        assert motion(workload).shape == states.shape[1:]
        contact_counts = qd_to_torch(solver.collider.collider_state.n_contacts, copy=False)
        counts = contact_counts.new_empty((args.steps, args.envs))
        peaks = states.new_empty((args.steps, args.envs)) if observed_stress else None
        validity = torch.empty((args.steps, args.envs), dtype=gs.tc_bool, device=gs.device) if observed_stress else None
        errno_view = qd_to_torch(solver._errno, copy=False)
        errors = errno_view.new_empty((args.steps, args.envs))
        field_ticks = sorted({0, args.steps - 1, *range(args.field_stride - 1, args.steps, args.field_stride)})
        tensors, mises = [], []
        for tick in range(args.steps):
            workload.step()
            states[tick].copy_(motion(workload))
            if args.construction_snapshots and tick + 1 in (108, 109, 110, 120, args.steps):
                capture_physics(workload, args.output / f"step{tick + 1}.npz")
            counts[tick].copy_(contact_counts)
            errors[tick].copy_(errno_view)
            if observed_stress:
                peaks[tick].copy_(workload.link.get_max_stress(copy=False))
                validity[tick].copy_(qd_to_torch(entry.state.step_valid, copy=False))
                if args.output_mode == "full" and tick in field_ticks:
                    tensor, vm = workload.link.get_stress_field(copy=False)
                    assert tensor.is_cuda and vm.is_cuda
                    tensors.append(tensor.clone())
                    mises.append(vm.clone())
            for buffer in buffers:
                buffer.frames[tick].copy_(buffer.live)
        qd.sync()
        solver.check_errno()
        assert torch.isfinite(states).all() and (errors == 0).all()
        if observed_stress:
            assert validity.all() and torch.isfinite(peaks).all() and (peaks >= 0).all()
        arrays = {
            "motion": tensor_to_array(states),
            "contact_counts": tensor_to_array(counts),
            "errno": tensor_to_array(errors),
            "field_ticks": np.array(field_ticks),
        }
        checks = {"finite_motion": True, "every_step_errno_zero": True, "all_environments_retained": True}
        if observed_stress:
            arrays.update(peak_pa=tensor_to_array(peaks), valid=tensor_to_array(validity))
            checks.update(every_step_valid=True, finite_nonnegative_peaks=True)
        if tensors:
            tensor, vm = torch.stack(tensors), torch.stack(mises)
            assert tensor.shape == (len(field_ticks), args.envs, 480, 4, 6)
            assert torch.isfinite(tensor).all() and torch.isfinite(vm).all()
            xx, yy, zz, xy, xz, yz = tensor.unbind(dim=-1)
            independent_vm = torch.sqrt(
                0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + xz**2 + yz**2)
            )
            torch.testing.assert_close(vm, independent_vm, rtol=1e-10, atol=1e-7)
            torch.testing.assert_close(peaks[field_ticks], vm.amax(dim=(-2, -1)), rtol=1e-10, atol=1e-7)
            arrays.update(stress_pa=tensor_to_array(tensor), mises_pa=tensor_to_array(vm))
            checks.update(full_tensor_shape=True, full_vm_independent=True, peak_matches_full=True)
        np.savez_compressed(args.output / "trajectory.npz", **arrays)
        write_json(
            args.output / "conformance.json",
            {
                "passed": True,
                "checks": checks,
                "trajectory": file_id(args.output / "trajectory.npz"),
                "steps": args.steps,
                "envs": args.envs,
                "scope": "Complete recorded scene control before the separately admitted module replay.",
            },
        )
        replay_result = None
        if args.replay:
            # An untimed complete history checks the standalone native route
            # before timing it. Neither failures nor unloaded frames are removed.
            replay_peaks = torch.empty_like(peaks)
            replay_valid = torch.empty_like(validity)
            for tick in range(args.steps):
                restore(workload, buffers, tick, resets)
                solver.stress_recovery.recover(0)
                replay_peaks[tick].copy_(workload.link.get_max_stress(copy=False))
                replay_valid[tick].copy_(qd_to_torch(entry.state.step_valid, copy=False))
                if tensors and tick in field_ticks:
                    tensor, vm = workload.link.get_stress_field(copy=False)
                    i = field_ticks.index(tick)
                    torch.testing.assert_close(tensor, tensors[i], rtol=1e-4, atol=1e-3)
                    torch.testing.assert_close(vm, mises[i], rtol=1e-4, atol=1e-3)
            qd.sync()
            solver.check_errno()
            assert replay_valid.all()
            torch.testing.assert_close(replay_peaks, peaks, rtol=1e-4, atol=1e-3)
            history = args.output / "replay-inputs.npz"
            np.savez_compressed(history, **{b.name: tensor_to_array(b.frames) for b in buffers})
            history_identity = file_id(history)
            rows = []
            for _ in range(args.repetitions):
                before = qd_to_torch(entry.state.invalid_steps, copy=True)
                qd.sync()
                started = perf_counter()
                for tick in range(args.steps):
                    restore(workload, buffers, tick, resets)
                    solver.stress_recovery.recover(0)
                qd.sync()
                elapsed = perf_counter() - started
                solver.check_errno()
                invalid = qd_to_torch(entry.state.invalid_steps, copy=False) - before
                assert (invalid == 0).all()
                rows.append(
                    {
                        "updates": args.steps,
                        "seconds": elapsed,
                        "env_updates_per_second": args.envs * args.steps / elapsed,
                        "invalid_environment_updates": int(invalid.sum()),
                    }
                )
            replay_result = {
                "repeats": rows,
                "input_history": history_identity,
                "stress_reset_schedule": {str(k): tensor_to_array(v).tolist() for k, v in resets.items()},
                "partial_resets_replayed": True,
                "peak_max_absolute_difference_pa": float((replay_peaks - peaks).abs().max()),
                "complete_history_check_passed": True,
                "recorded_device_bytes": sum(b.frames.numel() * b.frames.element_size() for b in buffers),
                "buffers": [
                    {"name": b.name, "shape": list(b.frames.shape), "dtype": str(b.frames.dtype)} for b in buffers
                ],
                "scope": "Changing recorded contact history, native association/reconstruction/recovery/acceptance and device input restore; no dynamics or host field copies. Elastic history0; existing traction warm start retained. Warm repeated histories, not independent process repeats.",
            }
        write_json(
            args.output / "result.json",
            {
                "identity": runtime_identity(),
                "passed": True,
                "checks": checks,
                "method": args.method,
                "observed_stress": observed_stress,
                "effective_rigid_post_fusion": solver.stress_recovery.post_kernel is not None
                if entry is not None
                else False,
                "output_mode": args.output_mode,
                "envs": args.envs,
                "steps": args.steps,
                "warmup": args.warmup,
                "dt_s": 0.01,
                "substeps": 1,
                "resets": workload.reset_count,
                "conditions": file_id(args.conditions),
                "trajectory": file_id(args.output / "trajectory.npz"),
                "stress_configuration": entry.options.model_dump(mode="json") if entry is not None else None,
                "intrinsic_workspace_contacts": entry.intrinsic.contacts.valid.shape[0]
                if entry is not None and entry.intrinsic is not None
                else None,
                "replay": replay_result,
                "physical_reference_qualified": False,
                "scope": "Intrusive motion/output control; recording cost is not primary production throughput.",
            },
        )


if __name__ == "__main__":
    main()
