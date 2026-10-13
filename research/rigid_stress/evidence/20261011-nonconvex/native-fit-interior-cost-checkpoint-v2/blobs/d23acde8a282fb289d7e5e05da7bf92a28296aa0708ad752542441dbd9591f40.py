"""Measure applicability of a constraint-interior fit shortcut; do not change recovery."""

import argparse
import json
from pathlib import Path

import numpy as np
import quadrants as qd
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark.rigid_stress_metrics import RolloutMetrics
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionInfo,
    StressTractionState,
    StressTractionStatus,
    func_traction_constraint_active,
    func_traction_geometry,
    func_traction_reference,
    func_traction_source_friction,
    func_traction_support_range,
)

import genesis as gs
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def kernel_interior_applicability(
    contacts: StressContactState,
    tractions: StressTractionState,
    info: StressTractionInfo,
    counters: qd.Tensor,
    epsilon: float,
):
    for i_a, i_b in qd.ndrange(counters.shape[0], counters.shape[1]):
        counters[i_a, i_b] = 0
    qd.loop_config(block_dim=128)
    for i_thread in range(contacts.valid.shape[0] * contacts.valid.shape[1] * 32):
        lane = i_thread % 32
        i_pair = i_thread // 32
        i_c, i_b = i_pair // contacts.valid.shape[1], i_pair % contacts.valid.shape[1]
        if (
            contacts.valid[i_c, i_b]
            and func_traction_constraint_active(i_c, i_b, tractions)
            and tractions.status[i_c, i_b] == StressTractionStatus.SUCCESS
            and tractions.scale[i_c, i_b] > 0.0
        ):
            samples, outside = gs.qd_int(0), gs.qd_int(0)
            start, end = func_traction_support_range(i_c, i_b, tractions)
            force = contacts.force[i_c, i_b] / tractions.scale[i_c, i_b]
            dual = tractions.dual[i_c, i_b]
            for i_s in range(start, end):
                reference = func_traction_reference(i_s, i_c, i_b, force, tractions.weight_sum[i_c, i_b], tractions)
                friction = func_traction_source_friction(i_s, i_b, contacts.friction[i_c, i_b], tractions)
                for i_chunk in range((tractions.sample_count[i_s, i_b] + 31) // 32):
                    i_q = i_chunk * 32 + lane
                    if i_q < tractions.sample_count[i_s, i_b] and tractions.profile[i_q, i_s, i_b] > 0.0:
                        position, normal = func_traction_geometry(i_q, i_s, i_b, tractions, info)
                        relative = (position - contacts.center[i_c, i_b]) / contacts.radius[i_c, i_b]
                        value = reference - dual[0:3] - dual[3:6].cross(relative)
                        compression = value.dot(normal)
                        tangential = (value - compression * normal).norm()
                        roundoff = 128.0 * epsilon * value.norm()
                        samples += 1
                        if compression < -roundoff or tangential > friction * qd.max(compression, 0.0) + roundoff:
                            outside += 1
            samples = qd.simt.subgroup.reduce_all_add(samples)
            outside = qd.simt.subgroup.reduce_all_add(outside)
            if lane == 0:
                qd.atomic_add(counters[0, i_b], 1)
                qd.atomic_add(counters[1, i_b], gs.qd_int(outside == 0))
                qd.atomic_add(counters[2, i_b], samples)
                qd.atomic_add(counters[3, i_b], outside)
                work = samples * tractions.evaluations[i_c, i_b]
                qd.atomic_add(counters[4, i_b], work)
                if outside == 0:
                    qd.atomic_add(counters[5, i_b], work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.cuda, precision="64", seed=623102, logging_level="warning")
    workload = FrankaEgg(
        2048, seed=623102, varied=True, conditions=args.conditions, contact_model="intrinsic",
        intrinsic_load_partition="body_source_prior", contact_block_size=4, egg_geometry="mesh64",
        use_gjk_collision=True, enable_contact_patch=True,
    )
    for _ in range(600):
        workload.step()
    workload.restart()
    metrics = RolloutMetrics(workload)
    counters = V(dtype=gs.qd_int, shape=(6, 2048))
    samples = []
    for _ in range(1280):
        workload.step()
        metrics.record()
        entry = workload.scene.rigid_solver.stress_recovery.links[0]
        kernel_interior_applicability(
            entry.intrinsic.contacts, entry.intrinsic.tractions, entry.intrinsic.traction_info, counters,
            float(np.finfo(np.float64).eps),
        )
        samples.append(qd_to_numpy(counters, transpose=True, copy=True))
    workload.scene.rigid_solver.check_errno()
    raw = np.stack(samples)
    assert raw.shape == (1280, 2048, 6)
    np.savez_compressed(args.output.with_suffix(".npz"), counters=raw)
    totals = raw.sum(axis=(0, 1))
    result = {
        "envs": 2048, "seed": 623102, "warmup": 600, "steps": 1280,
        "counter_columns": ["loaded_constraints", "interior_constraints", "positive_samples", "projected_samples",
                            "sample_evaluations", "interior_sample_evaluations"],
        "totals": totals.tolist(), "interior_loaded_fraction": float(totals[1] / totals[0]),
        "interior_sample_evaluation_fraction": float(totals[5] / totals[4]),
        "trajectory_metrics": metrics.summary(),
        "limitations": "Diagnostic only: final dual proposals classify constraint activity with128eps relative roundoff, without changing loads/solver/acceptance. Sample-times-evaluations is a work proxy, not measured per-environment latency or a speedup bound. Any direct shortcut would still need current wrench/cone/full-field validation.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "trajectory_metrics"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
