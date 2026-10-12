"""Device-side trajectory outcomes and per-environment recovery work, collected inside rollout timing."""

from dataclasses import dataclass

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.utils.array_class import ColliderState, V
from genesis.utils.misc import qd_to_numpy, qd_to_torch

COUNTERS = (
    "observed_steps",
    "lift_window_steps",
    "lift_grasp_pass_steps",
    "lift_grasp_fail_steps",
    "completed_lift_success_cycles",
    "completed_lift_failure_cycles",
    "weak_grip_steps",
    "weak_grip_descents",
    "release_steps",
    "release_descents",
    "loaded_steps",
    "fit_evaluation_sum",
    "fit_evaluation_max",
    "sample_count_max",
    "integration_retry_frames",
    "elastic_corrections",
    "elastic_fallbacks",
    "invalid_stress_steps",
    "two_finger_contact_steps",
    "egg_contact_count_max",
)


@dataclass(frozen=True)
class RolloutMetricState:
    counters: qd.Tensor
    previous_phase: qd.Tensor
    previous_height: qd.Tensor
    lift_seen: qd.Tensor
    lift_passed: qd.Tensor
    maximum_height: qd.Tensor


@qd.kernel
def kernel_trajectory_metrics(
    i_egg: int,
    i_left: int,
    i_right: int,
    initial_position: qd.types.ndarray(),
    phase: qd.types.ndarray(),
    position: qd.types.ndarray(),
    metrics_state: RolloutMetricState,
    collider_state: ColliderState,
):
    for i_b in range(metrics_state.previous_phase.shape[0]):
        value = phase[i_b]
        height = position[i_b, 2] - initial_position[i_b, 2]
        left, right, n_contacts = False, False, 0
        for i_col in range(collider_state.n_contacts[i_b]):
            i_contact = collider_state.contact_sort_idx[i_col, i_b]
            i_a = collider_state.contact_data.link_a[i_contact, i_b]
            i_other = collider_state.contact_data.link_b[i_contact, i_b]
            if i_other == i_egg:
                i_other = i_a
            if (
                i_a == i_egg or collider_state.contact_data.link_b[i_contact, i_b] == i_egg
            ) and collider_state.contact_data.force[i_contact, i_b].norm_sqr() > 1e-24:
                n_contacts += 1
                left = left or i_other == i_left
                right = right or i_other == i_right
        metrics_state.counters[18, i_b] += gs.qd_int(left and right)
        metrics_state.counters[19, i_b] = qd.max(metrics_state.counters[19, i_b], n_contacts)
        if value < metrics_state.previous_phase[i_b]:
            metrics_state.lift_seen[i_b] = False
            metrics_state.lift_passed[i_b] = True
        metrics_state.counters[0, i_b] += 1
        metrics_state.maximum_height[i_b] = qd.max(metrics_state.maximum_height[i_b], height)
        if 0.52 <= value < 0.60:
            metrics_state.counters[1, i_b] += 1
            passed = height >= 0.05 and left and right
            i_result = 2 if passed else 3
            metrics_state.counters[i_result, i_b] += 1
            metrics_state.lift_seen[i_b] = True
            metrics_state.lift_passed[i_b] = metrics_state.lift_passed[i_b] and passed
        if metrics_state.previous_phase[i_b] < 0.60 <= value and metrics_state.lift_seen[i_b]:
            i_result = 4 if metrics_state.lift_passed[i_b] else 5
            metrics_state.counters[i_result, i_b] += 1
            metrics_state.lift_seen[i_b] = False
        if 0.70 <= value < 0.76:
            metrics_state.counters[6, i_b] += 1
            if height < metrics_state.previous_height[i_b] - 0.001:
                metrics_state.counters[7, i_b] += 1
        if value >= 0.92:
            metrics_state.counters[8, i_b] += 1
            if height < metrics_state.previous_height[i_b] - 0.001:
                metrics_state.counters[9, i_b] += 1
        metrics_state.previous_phase[i_b] = value
        metrics_state.previous_height[i_b] = height


@qd.kernel
def kernel_recovery_work_metrics(
    metrics_state: RolloutMetricState,
    contact_state: StressContactState,
    stress_state: StressState,
):
    for i_b in range(metrics_state.previous_phase.shape[0]):
        loaded, refined = False, False
        evaluations, maximum_evaluations, maximum_samples = 0, 0, 0
        for i_c in range(contact_state.valid.shape[0]):
            if contact_state.valid[i_c, i_b]:
                loaded = True
                refined = refined or contact_state.is_refined[i_c, i_b]
                evaluations += contact_state.evaluations[i_c, i_b]
                maximum_evaluations = qd.max(maximum_evaluations, contact_state.evaluations[i_c, i_b])
                maximum_samples = qd.max(maximum_samples, contact_state.candidate_count[i_c, i_b])
        metrics_state.counters[10, i_b] += gs.qd_int(loaded)
        metrics_state.counters[11, i_b] += evaluations
        metrics_state.counters[12, i_b] = qd.max(metrics_state.counters[12, i_b], maximum_evaluations)
        metrics_state.counters[13, i_b] = qd.max(metrics_state.counters[13, i_b], maximum_samples)
        metrics_state.counters[14, i_b] += gs.qd_int(refined)
        metrics_state.counters[15, i_b] += stress_state.corrections[i_b]
        metrics_state.counters[16, i_b] += stress_state.fallbacks[i_b]
        metrics_state.counters[17, i_b] += gs.qd_int(not stress_state.step_valid[i_b])


class RolloutMetrics:
    def __init__(self, workload):
        self.workload = workload
        self.finger_links = tuple(workload.robot.get_link(name).idx for name in ("left_finger", "right_finger"))
        self.state = RolloutMetricState(
            counters=V(dtype=gs.qd_int, shape=(len(COUNTERS), workload.n_envs)),
            previous_phase=V(dtype=gs.qd_float, shape=(workload.n_envs,)),
            previous_height=V(dtype=gs.qd_float, shape=(workload.n_envs,)),
            lift_seen=V(dtype=gs.qd_bool, shape=(workload.n_envs,)),
            lift_passed=V(dtype=gs.qd_bool, shape=(workload.n_envs,)),
            maximum_height=V(dtype=gs.qd_float, shape=(workload.n_envs,)),
        )
        self.native_buffer_bytes = 0
        for tensor in (
            self.state.counters,
            self.state.previous_phase,
            self.state.previous_height,
            self.state.lift_seen,
            self.state.lift_passed,
            self.state.maximum_height,
        ):
            view = qd_to_torch(tensor, copy=False)
            self.native_buffer_bytes += view.numel() * view.element_size()
        self.reset()

    def reset(self):
        self.state.counters.fill(0)
        self.state.previous_phase.fill(0.0)
        self.state.previous_height.fill(0.0)
        self.state.lift_seen.fill(False)
        self.state.lift_passed.fill(True)
        self.state.maximum_height.fill(0.0)

    def record(self):
        kernel_trajectory_metrics(
            self.workload.link.idx,
            self.finger_links[0],
            self.finger_links[1],
            self.workload.initial_position,
            self.workload.phase,
            self.workload.egg.get_pos(),
            self.state,
            self.workload.scene.rigid_solver.collider.collider_state,
        )
        if self.workload.stress:
            entry = self.workload.scene.rigid_solver.stress_recovery.links[0]
            kernel_recovery_work_metrics(self.state, entry.contacts, entry.state)

    def summary(self):
        counts = qd_to_numpy(self.state.counters, transpose=True)
        height = qd_to_numpy(self.state.maximum_height, transpose=True)
        pending = qd_to_numpy(self.state.lift_seen, transpose=True)
        result = {
            "native_buffer_bytes": self.native_buffer_bytes,
            "contract": {
                "lift_success": "Every observed phase in [0.52,0.60) has egg height >= initial height + 0.05 m and nonzero (>1e-12 N) contacts with both finger links, counted at crossing 0.60.",
                "descent": "Height decreases by more than 0.001 m over one dt=0.01 step in the named controller window.",
                "tail": "Per-environment recorded fit evaluations, sample counts, integration retries and elastic work. These are parallel work proxies, not individual GPU wall times.",
                "timing": "Both device collection kernels and the public position getter are inside each rollout step. Boundary exports are outside repeat timing.",
            },
            "totals": {
                name: int(counts[:, i].sum(dtype=np.int64))
                for i, name in enumerate(COUNTERS)
                if not name.endswith("_max")
            },
            "per_environment": {name: counts[:, i].tolist() for i, name in enumerate(COUNTERS)},
            "maximum_relative_height_m": height.tolist(),
            "unfinished_lift_window_environments": np.flatnonzero(pending).tolist(),
            "work_percentiles": {},
        }
        for name in (
            "fit_evaluation_sum",
            "fit_evaluation_max",
            "sample_count_max",
            "integration_retry_frames",
            "elastic_corrections",
            "elastic_fallbacks",
            "invalid_stress_steps",
        ):
            i = COUNTERS.index(name)
            result["work_percentiles"][name] = dict(
                zip(("p0", "p50", "p90", "p99", "p100"), np.percentile(counts[:, i], (0, 50, 90, 99, 100)).tolist())
            )
        return result
