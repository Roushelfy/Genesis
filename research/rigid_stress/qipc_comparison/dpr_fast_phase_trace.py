"""Locate stalled adaptive operations with synchronized production phase calls."""

import time

import quadrants as qd

from genesis.engine.solvers.rigid.stress.adaptive_pressure import func_adaptive_contact, func_adaptive_initialize
from genesis.engine.solvers.rigid.stress.contact import (
    func_pack_contacts,
    func_pressure_correct_warp,
    func_pressure_warp,
)
from genesis.engine.solvers.rigid.stress.scatter import func_prepare_scatter, func_pressure_moments

from .artifacts import write_json


@qd.kernel
def kernel_phase(
    contacts: qd.template(),
    state: qd.template(),
    info: qd.template(),
    surface: qd.template(),
    workspace: qd.template(),
    cached_bounds: qd.template(),
    reuse_moments: qd.template(),
    phase: qd.template(),
):
    if qd.static(phase == 0):
        func_adaptive_initialize(contacts)
        func_pack_contacts(contacts)
    elif qd.static(phase == 1):
        if qd.static(reuse_moments):
            func_pressure_moments(contacts, state, info, surface, workspace, cached_bounds)
        else:
            func_prepare_scatter(contacts, state, info, surface, workspace, cached_bounds)
    elif qd.static(phase == 2):
        if qd.static(reuse_moments):
            func_pressure_warp(contacts, surface, True, workspace.refined_pressure, True)
        else:
            func_pressure_warp(contacts, surface, True, contacts.valid, False)
    elif qd.static(phase == 3):
        func_pressure_correct_warp(contacts, surface)
    elif qd.static(phase == 4):
        cached = (
            workspace.count[None] <= workspace.tasks.shape[0]
            and contacts.active_count[None] <= workspace.source_heads.shape[0]
        )
        for i_pair in range(qd.select(cached, contacts.active_count[None], 0)):
            workspace.source_heads[i_pair] = qd.Vector([-1, 0])
        for i_task in range(qd.select(cached, workspace.count[None], 0)):
            i_pair = workspace.tasks[i_task][0]
            previous = qd.atomic_exchange(workspace.source_heads[i_pair][0], i_task)
            qd.atomic_add(workspace.source_heads[i_pair][1], 1)
            workspace.next_task[i_task] = qd.Vector([previous])
    elif qd.static(phase == 5):
        qd.loop_config(block_dim=128)
        for i_thread in range(contacts.active_count[None] * 32):
            i_pair, i_lane = i_thread // 32, i_thread % 32
            pair = contacts.active_pairs[i_pair]
            func_adaptive_contact(
                i_pair, pair[0], pair[1], i_lane, contacts, info, surface, workspace, True, reuse_moments, True
            )


def trace_adaptive(contacts, state, info, surface, workspace, cached_bounds, reuse_moments, directory):
    report = {
        "schema": "genesis.adaptive-phase-trace.v1",
        "scope": "Diagnostic synchronization between unchanged production functions; excluded from primary timing.",
        "complete": False,
        "phases": [],
    }
    for phase, name in enumerate(("initialize_pack", "moments", "initial_fit", "nonlinear_fit", "chain", "adaptive")):
        entry = {"phase": phase, "name": name, "complete": False}
        report["phases"].append(entry)
        write_json(directory / "phase-trace.json", report)
        start = time.perf_counter()
        kernel_phase(contacts, state, info, surface, workspace, cached_bounds, reuse_moments, phase)
        qd.sync()
        entry.update(complete=True, elapsed_seconds=time.perf_counter() - start)
        write_json(directory / "phase-trace.json", report)
    report["complete"] = True
    write_json(directory / "phase-trace.json", report)
