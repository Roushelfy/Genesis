"""Measure face-parallel Q10 scatter against the native contact-warp ordering."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.contact import (
    StressContactState,
    func_contact_sample,
    func_force_frame,
    func_scatter_warp,
    kernel_scatter,
)
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy


@dataclass(frozen=True)
class FaceTasks:
    tasks: qd.Tensor
    count: qd.Tensor
    wrench: qd.Tensor


def create_workspace(contacts, surface, faces_per_env=32):
    capacity = contacts.valid.shape[1] * faces_per_env
    return FaceTasks(
        V_VEC(2, dtype=gs.qd_int, shape=(capacity,)),
        V(dtype=qd.i64, shape=()),
        V_VEC(6, dtype=gs.qd_float, shape=(capacity,)),
    )


def generate_fallback():
    source = Path("genesis/engine/solvers/rigid/stress/contact.py").read_text()
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == "func_scatter_warp"
    )
    node.name = "func_scatter_fallback"
    node.args.args.append(ast.arg(arg="workspace", annotation=ast.Name(id="FaceTasks", ctx=ast.Load())))
    overflow = "workspace.count[None] > workspace.tasks.shape[0]"
    loops = [item for item in node.body if isinstance(item, ast.For)]
    loops[0].iter.args[0] = ast.parse(f"qd.select({overflow}, stress_state.force.shape[0], 0)", mode="eval").body
    loops[1].iter.args[0] = ast.parse(
        f"qd.select({overflow}, contact_state.active_count[None] * 32, 0)", mode="eval"
    ).body
    generated = (
        "from genesis.engine.solvers.rigid.stress.contact import *\n"
        "from research.rigid_stress.probe_face_tasks import FaceTasks\n"
        + ast.unparse(ast.fix_missing_locations(node))
        + "\n"
    )
    directory = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "cache" / "face_tasks"
    directory.mkdir(parents=True, exist_ok=True)
    module_path = directory / "scatter_fallback_generated.py"
    module_path.write_text(generated)
    spec = importlib.util.spec_from_file_location("scatter_fallback_generated", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, generated


@qd.func
def func_scatter_faces(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    cached: qd.template(),
    workspace: FaceTasks,
):
    workspace.count[None] = qd.select(
        qd.i64(contacts.active_count[None]) * surface.face_origin.shape[0] <= 2147483647
        and contacts.active_count[None] <= workspace.wrench.shape[0],
        0,
        workspace.tasks.shape[0] + 1,
    )
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.active.shape[0]):
        state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for slot in range(
        qd.select(contacts.active_count[None] <= workspace.wrench.shape[0], contacts.active_count[None], 0)
    ):
        workspace.wrench[slot] = qd.Vector.zero(gs.qd_float, 6)
    for task in range(
        qd.select(
            qd.i64(contacts.active_count[None]) * surface.face_origin.shape[0] <= 2147483647
            and contacts.active_count[None] <= workspace.wrench.shape[0],
            contacts.active_count[None] * surface.face_origin.shape[0],
            0,
        )
    ):
        slot, face = task // surface.face_origin.shape[0], task % surface.face_origin.shape[0]
        pair = contacts.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        if contacts.status[i_c, i_b] == 0:
            low, high = surface.face_bounds_low[face], surface.face_bounds_high[face]
            center = contacts.center[i_c, i_b]
            distance = qd.max(low - center, 0.0) + qd.max(center - high, 0.0)
            if distance.dot(distance) < contacts.radius[i_c, i_b] ** 2:
                index = qd.atomic_add(workspace.count[None], 1)
                if index < workspace.tasks.shape[0]:
                    workspace.tasks[index] = qd.Vector([slot, face])
    qd.loop_config(block_dim=128)
    for i_thread in range(
        qd.i32(
            qd.select(
                workspace.count[None] <= workspace.tasks.shape[0],
                workspace.count[None] * 32,
                0,
            )
        )
    ):
        lane, task = i_thread % 32, i_thread // 32
        slot, face = workspace.tasks[task][0], workspace.tasks[task][1]
        pair = contacts.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        force = contacts.force[i_c, i_b]
        first, second = func_force_frame(force)
        n_q = surface.shape.shape[0]
        start, count = face * n_q, n_q
        if contacts.is_refined[i_c, i_b] and face == contacts.anchor_face[i_c, i_b]:
            start, count = surface.positions.shape[0], 7 * n_q
        total, moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        load = qd.Matrix.zero(gs.qd_float, 6, 3)
        for chunk in range((count + 31) // 32):
            sample = chunk * 32 + lane
            if sample < count:
                coordinates, weight, position, shape, _face = func_contact_sample(
                    start + sample, i_c, i_b, first, second, contacts, surface
                )
                sample_force = weight * qd.max(0.0, coordinates.dot(contacts.coefficient[i_c, i_b])) * force
                total += sample_force
                moment += (position - contacts.position[i_c, i_b]).cross(sample_force)
                load += shape.outer_product(sample_force)
        for a, b in qd.static(qd.ndrange(6, 3)):
            load[a, b] = qd.simt.subgroup.reduce_all_add(load[a, b])
        for a in qd.static(range(3)):
            total[a] = qd.simt.subgroup.reduce_all_add(total[a])
            moment[a] = qd.simt.subgroup.reduce_all_add(moment[a])
        if lane == 0:
            for i_local in range(6):
                i_n = info.surface_nodes[face, i_local]
                for a in qd.static(range(3)):
                    qd.atomic_add(state.force[i_n, i_b][a], load[i_local, a])
            for a in qd.static(range(3)):
                qd.atomic_add(workspace.wrench[slot][a], total[a])
                qd.atomic_add(workspace.wrench[slot][a + 3], moment[a])
    for slot in range(qd.select(workspace.count[None] <= workspace.tasks.shape[0], contacts.active_count[None], 0)):
        pair = contacts.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        if contacts.status[i_c, i_b] == 0:
            accumulated = workspace.wrench[slot]
            total = qd.Vector([accumulated[a] for a in qd.static(range(3))])
            moment = qd.Vector([accumulated[a + 3] for a in qd.static(range(3))])
            force = contacts.force[i_c, i_b]
            contacts.force_error[i_c, i_b] = (total - force).norm()
            contacts.moment_error[i_c, i_b] = moment.norm()
            if (total - force).norm() > 1e-8 * force.norm() or moment.norm() > 1e-8 * force.norm() * contacts.radius[
                i_c, i_b
            ]:
                contacts.status[i_c, i_b] = 4


@qd.kernel(graph=True)
def kernel_scatter_faces(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: FaceTasks,
):
    func_scatter_faces(contacts, state, info, surface, True, workspace)


def main():
    sys.modules.setdefault("research.rigid_stress.probe_face_tasks", sys.modules[__name__])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--faces-per-env", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=510000, logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True)
    for _ in range(900):
        workload.step()
    workload.scene.rigid_solver.check_errno()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    workspace = create_workspace(entry.contacts, entry.surface.info, args.faces_per_env)
    fallback, generated_source = generate_fallback()

    @qd.kernel(graph=True)
    def kernel_bounded(
        contacts: StressContactState,
        state: StressState,
        info: StressInfo,
        surface: StressSurfaceInfo,
        workspace: FaceTasks,
    ):
        func_scatter_faces(contacts, state, info, surface, True, workspace)
        fallback.func_scatter_fallback(contacts, state, info, surface, True, workspace)

    def native():
        kernel_scatter(entry.contacts, entry.state, entry.model.info, entry.surface.info)

    def faces():
        kernel_bounded(entry.contacts, entry.state, entry.model.info, entry.surface.info, workspace)

    native()
    expected = qd_to_numpy(entry.state.force, copy=True)
    faces()
    actual = qd_to_numpy(entry.state.force, copy=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    assert (qd_to_numpy(entry.contacts.status) == 0).all()
    entry.model.recover(entry.omega, entry.state, surface_load=True)
    assert qd_to_numpy(entry.state.valid).all()
    peak_faces = qd_to_numpy(entry.state.peak, copy=True)
    native()
    entry.model.recover(entry.omega, entry.state, surface_load=True)
    peak_native = qd_to_numpy(entry.state.peak, copy=True)
    np.testing.assert_allclose(peak_faces, peak_native, rtol=1e-10, atol=1e-6)
    timings = {}
    for name, operation in (("native", native), ("faces_including_pack_wrench", faces)):
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        timings[name] = 1e3 * (time.perf_counter() - started) / args.repetitions
    result = {
        "envs": args.envs,
        "milliseconds": timings,
        "face_tasks": int(qd_to_numpy(workspace.count)),
        "workspace_capacity": workspace.tasks.shape[0],
        "fallback_used": int(qd_to_numpy(workspace.count)) > workspace.tasks.shape[0],
        "generated_fallback_source": generated_source,
        "generated_fallback_sha256": hashlib.sha256(generated_source.encode()).hexdigest(),
        "nodal_force_max_difference_N": float(np.max(np.abs(actual - expected))),
        "global_peak_max_difference_Pa": float(np.max(np.abs(peak_faces - peak_native))),
        "full_residual_max_N": float(np.sqrt(qd_to_numpy(entry.state.residual_norm_squared)).max()),
        "force_error_max_N": float(qd_to_numpy(entry.contacts.force_error).max()),
        "moment_error_max_Nm": float(qd_to_numpy(entry.contacts.moment_error).max()),
        "note": "Snapshot scheduling trial; actual rollout selection still required. Same eligible faces and fixed/refined Q10 samples, no load/contact truncation.",
    }
    result["workspace_bytes"] = workspace.tasks.shape[0] * 8 + 8 + np.prod(workspace.wrench.shape).item() * 48
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
