"""Intrusive changing-trajectory stage latencies and environment work; never a headline rate."""
import argparse
import json
import os
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark.rigid_stress_metrics import RolloutMetrics
from genesis.engine.solvers.rigid import rigid_solver
from genesis.engine.solvers.rigid.stress import (
    compatibility, cooperative_fit, cooperative_scatter, factor, intrinsic,
    inverse, model, recovery, surface_frontier, surface_inverse, traction,
)
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy, tensor_to_array


@dataclass(frozen=True)
class StepWork:
    values: qd.Tensor


@qd.kernel
def kernel_step_work(contacts: StressContactState, state: StressState, work: StepWork):
    for i_b in range(contacts.valid.shape[1]):
        n_contacts, evaluations, maximum_evaluations, samples, maximum_samples, retries = 0, 0, 0, 0, 0, 0
        for i_c in range(contacts.valid.shape[0]):
            if contacts.valid[i_c, i_b]:
                n_contacts += 1
                evaluations += contacts.evaluations[i_c, i_b]
                maximum_evaluations = qd.max(maximum_evaluations, contacts.evaluations[i_c, i_b])
                samples += contacts.candidate_count[i_c, i_b]
                maximum_samples = qd.max(maximum_samples, contacts.candidate_count[i_c, i_b])
                retries += gs.qd_int(contacts.is_refined[i_c, i_b])
        work.values[0, i_b], work.values[1, i_b], work.values[2, i_b] = n_contacts, evaluations, maximum_evaluations
        work.values[3, i_b], work.values[4, i_b], work.values[5, i_b] = samples, maximum_samples, retries
        work.values[6, i_b], work.values[7, i_b] = state.corrections[i_b], state.fallbacks[i_b]
        work.values[8, i_b] = gs.qd_int(not state.step_valid[i_b])


@dataclass
class ProfileState:
    is_active: bool = False
    tick: int = 0
    calls: list = field(default_factory=list)
    fit_attempts: list = field(default_factory=list)


@dataclass
class SynchronizedCall:
    original: Callable
    name: str
    state: ProfileState

    def __call__(self, *args, **kwargs):
        if not self.state.is_active:
            return self.original(*args, **kwargs)
        is_sample_retry = self.name == "fit_cooperative" and kwargs.get("sample_lanes", 32) == 8
        if (self.name == "fit_serial_retry" or is_sample_retry) and self.state.tick in (468, 934, 959, 1068):
            fitted = args[2]
            statuses = qd_to_numpy(fitted.status, transpose=True)
            evaluations = qd_to_numpy(fitted.evaluations, transpose=True)
            margin = qd_to_numpy(fitted.certificate_margin, transpose=True)
            failed = np.argwhere(statuses != 0)
            self.state.fit_attempts.append({
                "tick": self.state.tick, "retry_mode": "cooperative8" if is_sample_retry else args[7],
                "before": [{"env": int(i_b), "contact": int(i_c), "status": int(statuses[i_b, i_c]),
                            "evaluations": int(evaluations[i_b, i_c]), "certificate_margin": float(margin[i_b, i_c])}
                           for i_b, i_c in failed],
            })
        # The pre-fence belongs to preceding work, not to this kernel latency.
        qd.sync()
        start = time.perf_counter()
        result = self.original(*args, **kwargs)
        qd.sync()
        name = self.name
        if name == "fit_cooperative" and kwargs.get("is_retry", False):
            name = "fit_cooperative_cold_retry"
            if kwargs.get("sample_lanes", 32) == 8:
                name = "fit_cooperative_sample_retry8"
        if name == "fit_serial_retry":
            name += "_" + str(args[7])
        self.state.calls.append((self.state.tick, name, time.perf_counter() - start))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=623102)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--steps", type=int, default=1280)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    build_start = time.perf_counter()
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True, contact_model="intrinsic",
                        intrinsic_load_partition="body_source_prior", contact_block_size=4,
                        egg_geometry="mesh64", use_gjk_collision=True, enable_contact_patch=True)
    workload.save_conditions(args.output.with_suffix(".conditions.npz"))
    metrics = RolloutMetrics(workload)
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    work = StepWork(V(dtype=gs.qd_int, shape=(9, args.envs)))
    profile_state = ProfileState()
    bindings = (
        ("rigid_dynamics", rigid_solver, "kernel_substep_dynamics"),
        ("rigid_collision", rigid_solver, "kernel_substep_collision"),
        ("rigid_constraints", rigid_solver, "func_solve_body"),
        ("rigid_contact_resolve", rigid_solver, "kernel_resolve_stress_contacts"),
        ("rigid_integration", rigid_solver, "kernel_step_2"),
        ("begin_step", recovery, "kernel_begin_step"),
        ("association", recovery, "kernel_associate"),
        ("source_capture", recovery, "kernel_capture_rigid_sources"),
        ("contact_pack", intrinsic, "kernel_pack_intrinsic_contacts"),
        ("contact_load", intrinsic, "kernel_load_intrinsic_contacts"),
        ("vertex_binding", intrinsic, "kernel_bind_source_vertices"),
        ("source_resolution", intrinsic, "kernel_resolve_sources"),
        ("footprint_ports", intrinsic, "kernel_footprint_ports"),
        ("fan_frontier", surface_frontier, "kernel_fan_frontier"),
        ("fan_samples", surface_frontier, "kernel_fan_frontier_samples"),
        ("bind_support", intrinsic, "kernel_bind_traction_support_impl"),
        ("source_prior_snapshot", intrinsic, "kernel_snapshot_source_priors"),
        ("source_prior_partition", intrinsic, "kernel_partition_source_priors"),
        ("fit_cooperative", cooperative_fit, "kernel_fit_tractions"),
        ("fit_serial_retry", traction, "kernel_fit_tractions_impl"),
        ("disk_refinement", intrinsic, "kernel_refine_failed_disks"),
        ("prior_refresh", intrinsic, "kernel_refresh_refined_source_priors"),
        ("select_failed_fans", intrinsic, "kernel_select_failed_fans"),
        ("fan_refinement", intrinsic, "kernel_refine_failed_fans"),
        ("force_certificate", intrinsic, "kernel_certify_continuous_force"),
        ("wrench_certificate_search", compatibility, "kernel_search_wrench_certificate_impl"),
        ("continuous_wrench_certificate", compatibility, "kernel_certify_continuous_wrench_impl"),
        ("final_traction_scatter", cooperative_scatter, "kernel_scatter_cached_tractions"),
        ("final_traction_commit", cooperative_scatter, "kernel_commit_intrinsic_contacts"),
        ("body_partition_commit", intrinsic, "kernel_commit_body_partition"),
        ("failure_capture", intrinsic, "kernel_capture_contact_failure"),
        ("inertia_relief", model, "kernel_balance"),
        ("rhs_reduce", model, "kernel_direct_init"),
        ("surface_load_pack", surface_inverse, "kernel_surface_pack"),
        ("surface_inverse", surface_inverse, "kernel_surface_apply_packed"),
        ("surface_inverse_blocked", surface_inverse, "kernel_surface_apply_packed_block"),
        ("elastic_residual", model, "kernel_full_residual"),
        ("elastic_correction", inverse, "kernel_apply_inverse"),
        ("elastic_fallback", factor, "kernel_solve_cooperative"),
        ("global_peak", model, "kernel_peak"),
        ("final_accept", recovery, "kernel_accept"),
    )
    build_seconds = time.perf_counter() - build_start
    with torch.inference_mode(), ExitStack() as stack:
        for name, module, attribute in bindings:
            original = vars(module)[attribute]
            stack.enter_context(patch.object(module, attribute, SynchronizedCall(original, name, profile_state)))
        for _ in range(args.warmup):
            workload.step()
            metrics.record()
        qd.sync()
        workload.scene.rigid_solver.check_errno()
        warmup = metrics.summary()
        passes = []
        for is_instrumented in (True,):
            workload.restart()
            metrics.reset()
            profile_state.is_active = is_instrumented
            work_rows = np.empty((args.steps, args.envs, 9), dtype=gs.np_int)
            latencies = np.empty(args.steps)
            before_invalid = qd_to_numpy(entry.state.invalid_steps)
            for tick in range(args.steps):
                profile_state.tick = tick
                qd.sync()
                start = time.perf_counter()
                workload.step()
                metrics.record()
                qd.sync()
                latencies[tick] = time.perf_counter() - start
                # Work collection and host export are outside measured step latency.
                kernel_step_work(entry.contacts, entry.state, work)
                work_rows[tick] = qd_to_numpy(work.values, transpose=True)
                if tick in (40, 180, 330, 459, 649, 999, 1200) or np.max(work_rows[tick, :, 5]) > 0:
                    ids = np.unique(np.concatenate((np.arange(min(8, args.envs)), np.argsort(work_rows[tick, :, 2])[-8:])))
                    sample_path = args.output.parent / f"fields-{'instrumented' if is_instrumented else 'boundary'}-{tick}.npz"
                    np.savez_compressed(sample_path, envs=ids,
                        force=qd_to_numpy(entry.state.force, transpose=True)[ids],
                        rhs=qd_to_numpy(entry.state.rhs, transpose=True)[ids],
                        displacement=qd_to_numpy(entry.state.displacement, transpose=True)[ids],
                        peak=qd_to_numpy(entry.state.step_peak)[ids],
                        residual_N=np.sqrt(qd_to_numpy(entry.state.residual_norm_squared))[ids],
                        rhs_N=np.sqrt(qd_to_numpy(entry.state.rhs_norm_squared))[ids])
            invalid = qd_to_numpy(entry.state.invalid_steps) - before_invalid
            workload.scene.rigid_solver.check_errno()
            assert np.max(invalid) == 0 and np.max(work_rows[:, :, 8]) == 0
            position = tensor_to_array(workload.egg.get_pos())
            residual = np.sqrt(qd_to_numpy(entry.state.residual_norm_squared))
            rhs = np.sqrt(qd_to_numpy(entry.state.rhs_norm_squared))
            assert np.max(residual / np.maximum(1e-11, 1e-8 * rhs)) <= 1
            pass_path = args.output.with_suffix(f".{'instrumented' if is_instrumented else 'boundary'}.npz")
            np.savez_compressed(pass_path, work=work_rows, step_seconds=latencies, positions=position,
                                residual_N=residual, rhs_N=rhs, invalid=invalid)
            names = sorted({name for _, name, _ in profile_state.calls}) if is_instrumented else []
            stage_seconds = {name: np.zeros(args.steps) for name in names}
            for tick, name, seconds in profile_state.calls:
                if is_instrumented:
                    stage_seconds[name][tick] += seconds
            retry_mask = np.max(work_rows[:, :, 5], axis=1) > 0
            evaluations_max = np.max(work_rows[:, :, 2], axis=1)
            top_indices = np.argsort(np.max(work_rows[:, :, 2], axis=0))[-16:][::-1]
            row = {"is_instrumented": is_instrumented, "raw_npz": pass_path.name,
                   "resets": workload.reset_count, "invalid_environment_steps": int(invalid.sum()),
                   "metrics": metrics.summary(), "step_ms_percentiles": np.percentile(latencies * 1000, [0, 50, 90, 99, 100]).tolist(),
                   "mean_step_ms": float(latencies.mean() * 1000), "retry_steps": np.flatnonzero(retry_mask).tolist(),
                   "fit_evaluations_max": int(evaluations_max.max()),
                   "top_work_environments": [{"environment": int(i_b), "fit_evaluations_max": int(np.max(work_rows[:, i_b, 2])),
                                               "retry_frames": int(np.count_nonzero(work_rows[:, i_b, 5]))} for i_b in top_indices],
                   "stage_mean_ms": {name: float(seconds.mean() * 1000) for name, seconds in stage_seconds.items()},
                   "tail_steps": [{"tick": int(tick), "step_ms": float(latencies[tick] * 1000),
                                   "fit_evaluations_max": int(evaluations_max[tick]),
                                   "retry_environments": np.flatnonzero(work_rows[tick, :, 5]).tolist(),
                                   "stage_ms": {name: float(seconds[tick] * 1000) for name, seconds in stage_seconds.items()}}
                                  for tick in sorted(set(np.flatnonzero(retry_mask)) | set(np.argsort(latencies)[-16:]))]}
            passes.append(row)
            print(json.dumps({key: value for key, value in row.items() if key not in {"metrics", "tail_steps"}}, indent=2), flush=True)
        profile_state.is_active = False
    result = {"contract": __doc__, "envs": args.envs, "seed": args.seed, "dt": .01, "substeps": 1,
              "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"], "build_seconds_including_JIT": build_seconds,
              "warmup": warmup, "gpu": torch.cuda.get_device_name(),
              "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid), "quadrants": qd.__version__,
              "passes": passes, "calls": profile_state.calls, "fit_attempts": profile_state.fit_attempts,
              "limitations": "Kernel fences serialize and perturb dispatch; these are latency measurements, not overlapping sums or headline throughput. The boundary pass fences once per step, includes metrics, and exports work outside timing. Raw work proxies identify contributing environments; individual GPU wall times cannot be attributed from parallel work counts."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
