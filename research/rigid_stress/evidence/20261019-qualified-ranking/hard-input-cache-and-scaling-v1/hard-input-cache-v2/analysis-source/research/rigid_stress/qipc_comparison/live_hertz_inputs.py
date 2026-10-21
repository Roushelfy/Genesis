"""Same-substep Franka geometry/Hertz input pilot, with retained failures.

This research adapter replaces the requested stress callback with the native
input stage. It does not return an elastic field or qualify a pressure law.
"""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.base_solver import StateChange, Subscriber
from genesis.engine.solvers.rigid.stress.association import kernel_associate
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.curvature import StressCurvature, StressCurvatureInfo, func_surface_curvature
from genesis.engine.solvers.rigid.stress.hertz_regions import (
    StressHertzRegions,
    StressHertzRegionState,
    kernel_hertz_roots,
    kernel_hertz_planar,
    kernel_hertz_regions,
)
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_capture_rigid_sources
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.engine.solvers.rigid.stress.pair_geometry import func_opponent_boundary
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery, kernel_begin_step
from genesis.utils import array_class, geom
from genesis.utils.array_class import V, V_MAT, V_VEC
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_hertz_grouping import kernel_serial_grouping, kernel_compare_grouping
from .native_hertz import HertzState, NativeHertz


@dataclass(frozen=True)
class InputAuditState:
    status_counts: qd.Tensor
    invalid_steps: qd.Tensor
    loaded_steps: qd.Tensor
    force_error: qd.Tensor
    moment_error: qd.Tensor


@dataclass(frozen=True)
class BoundaryProbeState:
    geoms: qd.Tensor
    types: qd.Tensor
    normals: qd.Tensor
    points: qd.Tensor
    statuses: qd.Tensor
    curvature_statuses: qd.Tensor


@qd.kernel
def kernel_boundary_probe(
    i_l: int,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    contacts: StressContactState,
    sources: StressSourceState,
    probe: BoundaryProbeState,
    dyn_info: array_class.DynInfo,
    curvature_info: StressCurvatureInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0:
            i_contact = collider_state.contact_sort_idx[i_c, i_b]
            own = collider_state.contact_data.geom_a[i_contact, i_b]
            other = collider_state.contact_data.geom_b[i_contact, i_b]
            if collider_state.contact_data.link_b[i_contact, i_b] == i_l:
                own = collider_state.contact_data.geom_b[i_contact, i_b]
                other = collider_state.contact_data.geom_a[i_contact, i_b]
            position = geom.qd_transform_by_trans_quat(
                sources.witness[i_c, i_b], contacts.frame_position[i_b], contacts.frame_quaternion[i_b]
            )
            for i_side in range(2):
                i_g = own if i_side == 0 else other
                point, boundary_normal, gap, i_f, status = func_opponent_boundary(
                    i_g, i_b, position, dyn_state, dyn_info, epsilon
                )
                normal, operator, curvature_status = func_surface_curvature(
                    i_g, i_f, i_b, point, dyn_state, dyn_info, curvature_info, epsilon
                )
                probe.geoms[i_c, i_b][i_side] = i_g
                probe.types[i_c, i_b][i_side] = dyn_info.geoms.type[i_g]
                probe.normals[i_c, i_b][i_side, :] = normal
                probe.points[i_c, i_b][i_side, :] = point
                probe.statuses[i_c, i_b][i_side] = status
                probe.curvature_statuses[i_c, i_b][i_side] = curvature_status


@qd.kernel
def kernel_estimate_inputs(regions: StressHertzRegionState, hertz: HertzState):
    for i_c, i_b in qd.ndrange(regions.status.shape[0], regions.status.shape[1]):
        i = i_c * regions.status.shape[1] + i_b
        hertz.force[i] = 0.0
        hertz.modulus[i] = 1e10 / (1 - 0.3**2)
        hertz.curvature[i] = regions.curvature[i_c, i_b]
        if regions.is_valid[i_b] and regions.source_count[i_c, i_b] > 0:
            hertz.force[i] = -regions.force[i_c, i_b].dot(regions.normal[i_c, i_b])


@qd.kernel
def kernel_audit(
    contacts: StressContactState,
    regions: StressHertzRegionState,
    audit: InputAuditState,
):
    for i_b in range(regions.is_valid.shape[0]):
        loaded = False
        source_force, source_moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        region_force, region_moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        origin, rotation = contacts.frame_position[i_b], contacts.frame_quaternion[i_b]
        for i_c in range(regions.status.shape[0]):
            status = regions.status[i_c, i_b]
            if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0:
                loaded = True
                audit.status_counts[i_b][status] += 1
                force = geom.qd_transform_by_quat(contacts.force[i_c, i_b], rotation)
                offset = geom.qd_transform_by_quat(contacts.position[i_c, i_b], rotation)
                source_force += force
                source_moment += offset.cross(force)
            if regions.source_count[i_c, i_b] > 0:
                force = regions.force[i_c, i_b]
                region_force += force
                region_moment += regions.moment[i_c, i_b] + (regions.center[i_c, i_b] - origin).cross(force)
        if loaded:
            audit.loaded_steps[i_b] += 1
        if not regions.is_valid[i_b]:
            audit.invalid_steps[i_b] += 1
        else:
            audit.force_error[i_b] = qd.max(audit.force_error[i_b], (source_force - region_force).norm())
            audit.moment_error[i_b] = qd.max(audit.moment_error[i_b], (source_moment - region_moment).norm())


def array(field):
    return qd_to_numpy(field, transpose=True, copy=True)


class InputObserver:
    def __init__(self, solver, check_grouping=False):
        self.solver = solver
        self.entry = solver.stress_recovery.links[0]
        self.epsilon = float(np.finfo(gs.np_float).eps)
        qd.sync()
        begun = perf_counter()
        self.cache = StressCurvature(solver.dyn_info, self.epsilon)
        qd.sync()
        self.curvature_build_seconds = perf_counter() - begun
        self.profile = False
        self.audit_enabled = True
        self.phase_samples = []
        shape = self.entry.contacts.valid.shape
        self.sources = create_sources(*shape)
        self.regions = StressHertzRegions(self.entry.contacts)
        self.grouping_reference = StressHertzRegions(self.entry.contacts) if check_grouping else None
        self.grouping_errors = V(dtype=gs.qd_float, shape=(shape[1], 3))
        self.grouping_errors.fill(0)
        self.check_grouping = check_grouping
        self.hertz = NativeHertz(shape[0] * shape[1])
        self.cache_subscriber = Subscriber(
            to=frozenset((StateChange.GEOMETRY, StateChange.DYNAMICS)), callback=self.reset_geometry_cache
        )
        solver.subscribe(self.cache_subscriber)
        self.capture_probe = False
        self.probe = BoundaryProbeState(
            geoms=V_VEC(2, dtype=gs.qd_int, shape=shape),
            types=V_VEC(2, dtype=gs.qd_int, shape=shape),
            normals=V_MAT(2, 3, dtype=gs.qd_float, shape=shape),
            points=V_MAT(2, 3, dtype=gs.qd_float, shape=shape),
            statuses=V_VEC(2, dtype=gs.qd_int, shape=shape),
            curvature_statuses=V_VEC(2, dtype=gs.qd_int, shape=shape),
        )
        self.audit = InputAuditState(
            status_counts=V_VEC(6, dtype=gs.qd_int, shape=(shape[1],)),
            invalid_steps=V(dtype=gs.qd_int, shape=(shape[1],)),
            loaded_steps=V(dtype=gs.qd_int, shape=(shape[1],)),
            force_error=V(dtype=gs.qd_float, shape=(shape[1],)),
            moment_error=V(dtype=gs.qd_float, shape=(shape[1],)),
        )

    def reset_geometry_cache(self, change, envs_idx):
        self.regions.reset(self.solver.scene._sanitize_envs_idx(envs_idx))

    def phase(self, name, function, *args):
        if self.profile:
            qd.sync()
            begun = perf_counter()
            function(*args)
            qd.sync()
            self.phase_samples.append((name, perf_counter() - begun))
        else:
            function(*args)

    def reset_audit(self):
        self.audit.status_counts.fill(0)
        self.audit.invalid_steps.fill(0)
        self.audit.loaded_steps.fill(0)
        self.audit.force_error.fill(0)
        self.audit.moment_error.fill(0)

    def update(self, recovery, i_substep):
        assert recovery is self.solver.stress_recovery
        entry, solver = self.entry, self.solver
        if i_substep == 0:
            self.phase("begin_step", kernel_begin_step, entry.state)
        self.phase(
            "association",
            kernel_associate,
            entry.link.idx,
            solver._links_offset_pos,
            solver._links_offset_quat,
            entry.omega,
            solver.dyn_state,
            entry.contacts,
            solver.collider.collider_state,
            solver._links_offset_quat.ndim == 3,
            not solver._disable_constraint,
            entry.state.step_valid,
            solver._errno,
        )
        self.phase(
            "provenance",
            kernel_capture_rigid_sources,
            entry.link.idx,
            entry.contacts,
            self.sources,
            solver.collider.collider_state,
        )
        self.phase(
            "geometry_roots",
            kernel_hertz_roots,
            entry.link.idx,
            solver.dyn_state,
            solver.collider.collider_state,
            entry.contacts,
            self.sources,
            self.regions.state,
            self.regions.cache,
            solver.dyn_info,
            self.cache.info,
            self.epsilon,
        )
        self.phase(
            "geometry_planar",
            kernel_hertz_planar,
            solver.dyn_state,
            solver.collider.collider_state,
            entry.contacts,
            self.sources,
            self.regions.state,
            self.regions.cache,
            solver.dyn_info,
            self.cache.info,
            self.epsilon,
        )
        self.phase(
            "grouping_wrench",
            kernel_hertz_regions,
            entry.contacts,
            self.sources,
            self.regions.state,
            solver.collider.collider_state,
            self.epsilon,
        )
        self.phase("hertz_inputs", kernel_estimate_inputs, self.regions.state, self.hertz.state)
        self.phase("hertz_law", self.hertz.update)
        if self.audit_enabled:
            self.phase("research_audit", kernel_audit, entry.contacts, self.regions.state, self.audit)
        if self.check_grouping and self.audit_enabled:
            kernel_serial_grouping(
                entry.contacts, self.sources, self.regions.state, self.grouping_reference.state, self.epsilon
            )
            kernel_compare_grouping(self.regions.state, self.grouping_reference.state, self.grouping_errors)
        if self.capture_probe:
            self.phase(
                "research_probe",
                kernel_boundary_probe,
                entry.link.idx,
                solver.dyn_state,
                solver.collider.collider_state,
                entry.contacts,
                self.sources,
                self.probe,
                solver.dyn_info,
                self.cache.info,
                self.epsilon,
            )

    def report(self):
        return {
            "status_counts": array(self.audit.status_counts).tolist(),
            "invalid_environment_steps": array(self.audit.invalid_steps).tolist(),
            "loaded_environment_steps": array(self.audit.loaded_steps).tolist(),
            "maximum_force_discrepancy_n": array(self.audit.force_error).tolist(),
            "maximum_moment_discrepancy_nm": array(self.audit.moment_error).tolist(),
        }

    def snapshot(self):
        state = self.regions.state
        return {
            "source_valid": array(self.entry.contacts.valid),
            "source_force": array(self.entry.contacts.force),
            "source_position": array(self.entry.contacts.position),
            "source_normal": array(self.entry.contacts.normal),
            "source_geom": array(self.sources.geom),
            "source_witness": array(self.sources.witness),
            "frame_position": array(self.entry.contacts.frame_position),
            "frame_quaternion": array(self.entry.contacts.frame_quaternion),
            "center": array(state.center),
            "normal": array(state.normal),
            "curvature": array(state.curvature),
            "normal_residual": array(state.normal_residual),
            "gap": array(state.gap),
            "geometry_kind": array(state.geometry_kind),
            "status": array(state.status),
            "force": array(state.force),
            "moment": array(state.moment),
            "source_count": array(state.source_count),
            "is_valid": array(state.is_valid),
            "cache_face": array(self.regions.cache.face),
            "cache_hint_kind": array(self.regions.cache.hint_kind),
            "cache_hit": array(self.regions.cache.is_hit),
            "query_evaluations": array(self.regions.cache.evaluations),
            "hertz_axes": array(self.hertz.state.axes),
            "hertz_status": array(self.hertz.state.status),
            "probe_geoms": array(self.probe.geoms),
            "probe_types": array(self.probe.types),
            "probe_normals": array(self.probe.normals),
            "probe_points": array(self.probe.points),
            "probe_statuses": array(self.probe.statuses),
            "probe_curvature_statuses": array(self.probe.curvature_statuses),
        }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mode", choices=("inputs", "rigid"), required=True)
    cli.add_argument("--envs", type=int, default=4)
    cli.add_argument("--steps", type=int, default=720)
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--conditions", type=Path)
    cli.add_argument(
        "--check-grouping",
        action="store_true",
        help="Compare every audited input with the original serial grouping on the same device.",
    )
    cli.add_argument(
        "--profile-phases", action="store_true", help="Synchronize individual phases in the audited warmup only."
    )
    cli.add_argument(
        "--omit-timed-audit",
        action="store_true",
        help="Keep research audit counters in the separate audited rollout only.",
    )
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "arguments": vars(args) | {"conditions": str(args.conditions), "output": str(args.output)},
        },
    )
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    begun = perf_counter()
    workload = FrankaEgg(
        args.envs,
        level=1,
        stress=args.mode == "inputs",
        seed=623016,
        varied=True,
        conditions=args.conditions,
        output_mode="max",
        stress_update_mode="every_step",
        fused_rigid_post=False,
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
    )
    # The observer reads no prescribed radius; retain the identical rigid controller.
    workload.stress = False
    solver = workload.scene.rigid_solver
    observer = InputObserver(solver, args.check_grouping) if args.mode == "inputs" else None
    workload.save_conditions(args.output / "conditions.npz")
    qd.sync()
    setup_seconds = perf_counter() - begun
    positions, quaternions, joints, rollouts = [], [], [], []
    snapshots = []

    def observe(recovery, i_substep):
        observer.update(recovery, i_substep)

    context = patch.object(RigidStressRecovery, "recover", observe) if observer is not None else nullcontext()
    with context, torch.inference_mode(), gzip.open(args.output / "history.jsonl.gz", "wt") as history:
        for repeat in range(args.repeats + 1):
            if repeat:
                workload.restart()
            if observer is not None:
                observer.reset_audit()
                observer.capture_probe = repeat == 0
                observer.profile = args.profile_phases and repeat == 0
                observer.audit_enabled = repeat == 0 or not args.omit_timed_audit
            qd.sync()
            begun = perf_counter()
            for step in range(args.steps):
                workload.step()
                if repeat == 0:
                    positions.append(tensor_to_array(workload.egg.get_pos()).copy())
                    quaternions.append(tensor_to_array(workload.egg.get_quat()).copy())
                    joints.append(tensor_to_array(workload.robot.get_qpos()).copy())
                    if observer is not None:
                        snapshot = observer.snapshot()
                        active = snapshot["source_valid"] & (np.linalg.norm(snapshot["source_force"], axis=2) > 0)
                        row = {
                            "step": step,
                            "reset_count": workload.reset_count,
                            "is_valid": snapshot["is_valid"].tolist(),
                            "source_indices": np.argwhere(active).tolist(),
                            "source_status": snapshot["status"][active].tolist(),
                            "normal_residual": snapshot["normal_residual"][active].tolist(),
                            "curvature": snapshot["curvature"][active].tolist(),
                            "region_count": np.count_nonzero(snapshot["source_count"], axis=1).tolist(),
                            "query_evaluations": snapshot["query_evaluations"][active].tolist(),
                            "cache_hint_kind": snapshot["cache_hint_kind"][active].tolist(),
                            "cache_hit": snapshot["cache_hit"][active].tolist(),
                        }
                        history.write(json.dumps(row, allow_nan=False) + "\n")
                        if step % 60 == 0 or step == args.steps - 1:
                            path = args.output / f"step{step:04d}.npz"
                            np.savez_compressed(path, **snapshot)
                            snapshots.append(file_id(path))
            qd.sync()
            seconds = perf_counter() - begun
            solver.check_errno()
            row = {"repeat": repeat, "audited": repeat == 0, "seconds": seconds, "reset_count": workload.reset_count}
            if observer is not None and observer.audit_enabled:
                row["inputs"] = observer.report()
                assert max(row["inputs"]["maximum_force_discrepancy_n"]) < 1e-10
                assert max(row["inputs"]["maximum_moment_discrepancy_nm"]) < 1e-10
            row["research_audit_in_timing"] = observer is not None and observer.audit_enabled
            rollouts.append(row)
            write_json(args.output / "rollouts.json", rollouts)
    trajectory = args.output / "trajectory.npz"
    np.savez_compressed(trajectory, position=positions, quaternion=quaternions, joints=joints)
    assert np.isfinite(positions).all() and np.isfinite(quaternions).all() and np.isfinite(joints).all()
    if observer is not None:
        assert max(array(observer.audit.force_error)) < 1e-10
        assert max(array(observer.audit.moment_error)) < 1e-10
    grouping_errors = array(observer.grouping_errors).T if observer is not None and args.check_grouping else None
    if grouping_errors is not None:
        np.testing.assert_array_equal(grouping_errors, 0)
    phase_rows = []
    if observer is not None and observer.phase_samples:
        names = sorted({name for name, _ in observer.phase_samples})
        for name in names:
            values = np.array([seconds for phase, seconds in observer.phase_samples if phase == name])
            warmed = values[60:]
            phase_rows.append(
                {
                    "phase": name,
                    "samples": len(warmed),
                    "excluded_initial_samples": 60,
                    "median_seconds": float(np.median(warmed)),
                    "p95_seconds": float(np.quantile(warmed, 0.95)),
                    "sum_seconds": float(warmed.sum()),
                }
            )
        write_json(args.output / "phase-samples.json", observer.phase_samples)
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "mode": args.mode,
            "setup_seconds": setup_seconds,
            "curvature_build_seconds_including_first_compile": observer.curvature_build_seconds
            if observer is not None
            else None,
            "phase_profile": phase_rows,
            "grouping_errors": grouping_errors.tolist() if grouping_errors is not None else None,
            "grouping_check_scope": "Every audited step, all slots and environments: exact validity, leader, count, force and moment against the original serial kernel. Research controls are excluded from primary timing.",
            "phase_profile_scope": "Intrusive synchronized dispatch plus device execution in audited repeat zero; samples exclude first 60 steps. Primary inclusive scene timing uses uninstrumented repeats. No stress, traction transfer or history kernels run in this input-only pilot.",
            "omit_timed_audit": args.omit_timed_audit,
            "envs": args.envs,
            "contact_capacity_per_environment": observer.entry.contacts.valid.shape[0]
            if observer is not None
            else None,
            "shared_curvature_faces": observer.cache.info.status.shape[0] if observer is not None else None,
            "shared_curvature_vertices": observer.cache.info.vertices_weight.shape[0] if observer is not None else None,
            "steps": args.steps,
            "timestep": 0.01,
            "substeps": 1,
            "rollouts": rollouts,
            "trajectory": file_id(trajectory),
            "snapshots": snapshots,
            "scope": "Same preintegration substep native curvature/region/Hertz input pilot. Timings include the research callback; audit counters are included unless omit_timed_audit is true. Detailed host snapshots and phase synchronization occur only in repeat zero. No traction/stress output, pressure/finite-shell qualification, or tangential history law. Failed inputs remain reported and do not qualify usable estimator throughput.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
