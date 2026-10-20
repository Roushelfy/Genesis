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
from genesis.engine.solvers.rigid.stress.association import kernel_associate
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.curvature import StressCurvature
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegions, StressHertzRegionState
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_capture_rigid_sources
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery, kernel_begin_step
from genesis.utils import geom
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .native_hertz import HertzState, NativeHertz


@dataclass(frozen=True)
class InputAuditState:
    status_counts: qd.Tensor
    invalid_steps: qd.Tensor
    loaded_steps: qd.Tensor
    force_error: qd.Tensor
    moment_error: qd.Tensor


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
    def __init__(self, solver):
        self.solver = solver
        self.entry = solver.stress_recovery.links[0]
        self.epsilon = float(np.finfo(gs.np_float).eps)
        self.cache = StressCurvature(solver.dyn_info, self.epsilon)
        shape = self.entry.contacts.valid.shape
        self.sources = create_sources(*shape)
        self.regions = StressHertzRegions(self.entry.contacts)
        self.hertz = NativeHertz(shape[0] * shape[1])
        self.audit = InputAuditState(
            status_counts=V_VEC(6, dtype=gs.qd_int, shape=(shape[1],)),
            invalid_steps=V(dtype=gs.qd_int, shape=(shape[1],)),
            loaded_steps=V(dtype=gs.qd_int, shape=(shape[1],)),
            force_error=V(dtype=gs.qd_float, shape=(shape[1],)),
            moment_error=V(dtype=gs.qd_float, shape=(shape[1],)),
        )

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
            kernel_begin_step(entry.state)
        kernel_associate(
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
        kernel_capture_rigid_sources(entry.link.idx, entry.contacts, self.sources, solver.collider.collider_state)
        self.regions.update(
            entry.link.idx,
            entry.contacts,
            self.sources,
            solver.dyn_state,
            solver.collider.collider_state,
            solver.dyn_info,
            self.cache.info,
            self.epsilon,
        )
        kernel_estimate_inputs(self.regions.state, self.hertz.state)
        self.hertz.update()
        kernel_audit(entry.contacts, self.regions.state, self.audit)

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
            "hertz_axes": array(self.hertz.state.axes),
            "hertz_status": array(self.hertz.state.status),
        }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mode", choices=("inputs", "rigid"), required=True)
    cli.add_argument("--envs", type=int, default=4)
    cli.add_argument("--steps", type=int, default=720)
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--conditions", type=Path)
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
    observer = InputObserver(solver) if args.mode == "inputs" else None
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
            if observer is not None:
                row["inputs"] = observer.report()
            rollouts.append(row)
            write_json(args.output / "rollouts.json", rollouts)
    trajectory = args.output / "trajectory.npz"
    np.savez_compressed(trajectory, position=positions, quaternion=quaternions, joints=joints)
    assert np.isfinite(positions).all() and np.isfinite(quaternions).all() and np.isfinite(joints).all()
    if observer is not None:
        assert max(array(observer.audit.force_error)) < 1e-10
        assert max(array(observer.audit.moment_error)) < 1e-10
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "mode": args.mode,
            "setup_seconds": setup_seconds,
            "envs": args.envs,
            "steps": args.steps,
            "timestep": 0.01,
            "substeps": 1,
            "rollouts": rollouts,
            "trajectory": file_id(trajectory),
            "snapshots": snapshots,
            "scope": "Same preintegration substep native curvature/region/Hertz input pilot. Timings include aggregate audit counters and the research callback; detailed host snapshots occur only in repeat zero. No traction/stress output, pressure/finite-shell qualification, or tangential history law. Failed inputs remain reported and do not qualify usable estimator throughput.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
