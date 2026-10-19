"""Feed independent live environments through one native batched recovery call."""

from time import perf_counter

import numpy as np
from scipy.sparse.csgraph import connected_components
import torch

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.utils.misc import qd_to_numpy, qd_to_torch

from .artifacts import file_id
from .common_metric import voigt_tensor
from .native_reference_replay import ContactViews


def sample_batch(observer, forces_world, rotations, angular_world, friction, loads, paths=None):
    assert observer.source_reduction == "none"
    entry = observer.entry
    contacts = ContactViews.allocate(entry.contacts)
    envs = contacts.valid.shape[0]
    assert forces_world.shape == (envs, len(observer.points), 3)
    assert rotations.shape == (envs, 3, 3) and angular_world.shape == (envs, 3)
    assert np.shape(friction) == np.shape(loads) == (envs,)
    paths = [None] * envs if paths is None else paths
    assert len(paths) == envs
    clock = perf_counter if observer.phase_timing else lambda: 0.0
    started = clock()
    kernel_begin_step(entry.state)
    qd_to_torch(observer.errno, transpose=True, copy=False).zero_()
    qd_to_torch(entry.state.force, transpose=True, copy=False).zero_()
    for view in (
        contacts.valid,
        contacts.position,
        contacts.force,
        contacts.normal,
        contacts.frame_position,
        contacts.frame_quaternion,
    ):
        view.zero_()
    contacts.frame_quaternion[..., 0] = 1
    material_force = forces_world @ rotations
    normals = np.zeros_like(observer.points)
    normals[:, 2] = np.where(observer.points[:, 2] > 0, -1.0, 1.0)
    packets = []
    for env in range(envs):
        selected = np.flatnonzero(np.any(material_force[env] != 0, axis=1))
        assert len(selected) <= contacts.valid.shape[1]
        _, components = connected_components(observer.adjacency[selected][:, selected], directed=False)
        count = len(selected)
        for destination, values in (
            (contacts.position, observer.points[selected]),
            (contacts.force, material_force[env, selected]),
            (contacts.normal, (normals @ rotations[env])[selected]),
        ):
            destination[env, :count] = torch.as_tensor(values, dtype=gs.tc_float, device=gs.device)
        contacts.friction[env].fill_(float(friction[env]))
        contacts.radius[env].fill_(observer.radius)
        contacts.valid[env, :count] = True
        packets.append((selected, components))
    if entry.intrinsic is not None:
        sources = entry.intrinsic.raw_sources
        sources.geom.fill(entry.link.geoms[0].idx)
        sources.method.fill(0)
        sources.vertex.fill(-1)
        sources.witness_face.fill(-1)
        sources.has_vertex_ambiguity.fill(False)
        sources.free_moment.fill(0)
        pair = qd_to_torch(sources.pair, transpose=True, copy=False)
        witness = qd_to_torch(sources.witness, transpose=True, copy=False)
        pair.fill_(-1)
        witness.fill_(float("nan"))
        for env, (selected, components) in enumerate(packets):
            pair[env, : len(selected)] = torch.as_tensor(components, dtype=gs.tc_int, device=gs.device)
            witness[env, : len(selected)] = contacts.position[env, : len(selected)]
        entry.intrinsic.scatter(
            entry.contacts,
            entry.state,
            entry.model.info,
            entry.surface.info,
            entry.options,
            np.finfo(gs.np_float).eps.item(),
            observer.errno,
        )
    else:
        kernel_anchor(np.finfo(gs.np_float).eps.item(), entry.contacts, entry.surface.info)
        kernel_pressure(entry.contacts, entry.surface.info, entry.options.cooperative_pressure)
        kernel_scatter(
            entry.contacts,
            entry.state,
            entry.model.info,
            entry.surface.info,
            entry.options.cooperative_scatter,
            entry.options.cached_face_bounds,
        )
    qd_to_torch(entry.omega, transpose=True, copy=False)[:] = torch.as_tensor(
        np.einsum("bi,bij->bj", angular_world, rotations), dtype=gs.tc_float, device=gs.device
    )
    entry.model.recover(entry.omega, entry.state, surface_load=False)
    kernel_accept(entry.state, entry.contacts, observer.errno)
    torch.cuda.synchronize()
    voigt = (
        qd_to_numpy(entry.state.stress_tensor, transpose=True)
        if entry.output_mode == "full"
        else np.empty((envs, 0, 6))
    )
    residual = qd_to_numpy(entry.state.residual, transpose=True)
    rhs = qd_to_numpy(entry.state.rhs, transpose=True)
    errno = qd_to_numpy(observer.errno, transpose=True)
    peaks = qd_to_numpy(entry.state.step_peak, transpose=True)
    valid = qd_to_numpy(entry.state.valid, transpose=True)
    status = qd_to_numpy(entry.contacts.status, transpose=True)
    elapsed = clock() - started
    nodal_force = qd_to_numpy(entry.state.force, transpose=True) if any(path is not None for path in paths) else None
    rows = []
    for env, (selected, components) in enumerate(packets):
        stress = voigt_tensor(voigt[env])
        path = paths[env]
        if path is not None:
            np.savez_compressed(
                path,
                vertices_m=observer.coordinates,
                tetrahedra=observer.tetrahedra,
                stress_pa=stress,
                forces_n=nodal_force[env],
                rhs_n=rhs[env],
                equation_residual_n=residual[env],
                load_n=loads[env],
                raw_vm_peak_pa=peaks[env],
                source_points_m=observer.points[selected],
                source_forces_n=material_force[env, selected],
                source_components=components,
                original_contact_points_m=observer.points,
                original_contact_forces_n=material_force[env],
            )
        rows.append(
            {
                "env": env,
                "native_valid": bool(
                    valid[env] and errno[env] == 0 and np.isfinite(stress).all() and np.isfinite(peaks[env])
                ),
                "errno": int(errno[env]),
                "source_count": len(selected),
                "source_capacity": contacts.valid.shape[1],
                "source_reduction": "none",
                "contact_status": status[env, : len(selected)].tolist(),
                "raw_vm_peak_pa": float(peaks[env]),
                "field_host_bytes": voigt[env].nbytes,
                "residual_host_bytes": residual[env].nbytes + rhs[env].nbytes,
                "elastic_equation_relative_residual": float(
                    np.linalg.norm(residual[env]) / max(np.linalg.norm(rhs[env]), 1e-30)
                ),
                "field": file_id(path) if path is not None else None,
            }
        )
    return rows, {
        "phase_timing_enabled": observer.phase_timing,
        "batch_recovery_observation_seconds_excluding_frame_IO": elapsed,
    }
