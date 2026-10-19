"""Read actual native integration samples and verify their complete signed P2 scatter."""

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import (
    StressContactState,
    func_contact_cell,
    func_contact_index,
    func_contact_sample,
    func_force_frame,
)
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.grid import func_patch_grid
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionInfo,
    StressTractionState,
    func_source_traction,
    func_traction_geometry,
    func_traction_shape,
    func_traction_support_range,
)
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy


@qd.func
def emit(
    position,
    force,
    shape,
    face,
    source,
    sample,
    info: StressInfo,
    counter: qd.Tensor,
    points: qd.Tensor,
    forces: qd.Tensor,
    nodes: qd.Tensor,
    shapes: qd.Tensor,
    ids: qd.Tensor,
    save: qd.template(),
):
    # Exact nonzero check; there is no force-magnitude filter in the observer.
    if force[0] != 0.0 or force[1] != 0.0 or force[2] != 0.0:
        index = qd.atomic_add(counter[None], 1)
        if qd.static(save):
            if index < points.shape[0]:
                points[index], forces[index], shapes[index] = position, force, shape
                ids[index] = qd.Vector([source, sample, face])
                for k in qd.static(range(6)):
                    nodes[index][k] = info.surface_nodes[face, k]


@qd.kernel
def dpr_samples(
    contacts: StressContactState,
    surface: StressSurfaceInfo,
    info: StressInfo,
    counter: qd.Tensor,
    points: qd.Tensor,
    forces: qd.Tensor,
    nodes: qd.Tensor,
    shapes: qd.Tensor,
    ids: qd.Tensor,
    save: qd.template(),
):
    for source in range(contacts.valid.shape[0]):
        if contacts.valid[source, 0] and contacts.status[source, 0] == 0:
            force = contacts.force[source, 0]
            # This is the production scatter's own condition, not a new filter.
            if force.norm() > 1e-30:
                first, second = func_force_frame(force)
                low, dimensions = func_patch_grid(contacts.center[source, 0], contacts.radius[source, 0], surface.grid)
                for cell in range(dimensions[0] * dimensions[1] * dimensions[2] + 1):
                    start, end = func_contact_cell(cell, source, 0, low, dimensions, contacts, surface)
                    for entry in range(start, end):
                        sample = func_contact_index(entry, cell, dimensions, surface)
                        coordinate, weight, position, shape, face = func_contact_sample(
                            sample, source, 0, first, second, contacts, surface
                        )
                        actual = weight * qd.max(0.0, coordinate.dot(contacts.coefficient[source, 0])) * force
                        emit(
                            position,
                            actual,
                            shape,
                            face,
                            source,
                            sample,
                            info,
                            counter,
                            points,
                            forces,
                            nodes,
                            shapes,
                            ids,
                            save,
                        )


@qd.kernel
def scr_samples(
    contacts: StressContactState,
    traction: StressTractionState,
    traction_info: StressTractionInfo,
    surface: StressSurfaceInfo,
    info: StressInfo,
    counter: qd.Tensor,
    points: qd.Tensor,
    forces: qd.Tensor,
    nodes: qd.Tensor,
    shapes: qd.Tensor,
    ids: qd.Tensor,
    save: qd.template(),
):
    for source in range(contacts.valid.shape[0]):
        if contacts.valid[source, 0]:
            start, end = func_traction_support_range(source, 0, traction)
            for sample in range(traction.sample_count[source, 0]):
                if traction.profile[sample, source, 0] > 0.0:
                    force = qd.Vector.zero(gs.qd_float, 3)
                    for constraint in range(start, end):
                        force += func_source_traction(sample, source, constraint, 0, contacts, traction, traction_info)
                    position, _normal = func_traction_geometry(sample, source, 0, traction, traction_info)
                    face, shape = func_traction_shape(sample, source, 0, traction, surface, traction_info)
                    emit(
                        position,
                        force,
                        shape,
                        face,
                        source,
                        sample,
                        info,
                        counter,
                        points,
                        forces,
                        nodes,
                        shapes,
                        ids,
                        save,
                    )


def snapshot(observer):
    """Export the frozen one-environment full-source state without mutating it."""
    entry = observer.entry
    assert observer.source_reduction == "none" and entry.contacts.valid.shape[1] == 1
    expected = qd_to_numpy(entry.state.force, transpose=True)[0].copy()
    original_stress = qd_to_numpy(entry.state.stress_tensor, transpose=True).copy()
    assert qd_to_numpy(entry.state.valid, transpose=True)[0]
    counter = V(dtype=gs.qd_int, shape=())

    def allocate(count):
        return (
            V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            V_VEC(6, dtype=gs.qd_int, shape=(count,)),
            V_VEC(6, dtype=gs.qd_float, shape=(count,)),
            V_VEC(3, dtype=gs.qd_int, shape=(count,)),
        )

    def collect(buffers, save):
        counter.fill(0)
        if entry.intrinsic is None:
            dpr_samples(entry.contacts, entry.surface.info, entry.model.info, counter, *buffers, save)
        else:
            intrinsic = entry.intrinsic
            assert entry.options.intrinsic_load_partition == "body_source_prior"
            count = int(qd_to_numpy(intrinsic.packing.count, transpose=True)[0])
            assert count <= intrinsic.contacts.valid.shape[0], "Complete body union must fit one native block"
            scr_samples(
                intrinsic.contacts,
                intrinsic.tractions,
                intrinsic.traction_info,
                entry.surface.info,
                entry.model.info,
                counter,
                *buffers,
                save,
            )
        return int(qd_to_numpy(counter, transpose=True))

    count = collect(allocate(1), False)
    assert count > 0
    buffers = allocate(count)
    assert collect(buffers, True) == count, "Frozen native quadrature changed during capture"
    points, forces, nodes, shapes, ids = [qd_to_numpy(buffer, transpose=True) for buffer in buffers]
    assert points.shape == forces.shape == (count, 3) and nodes.shape == shapes.shape == (count, 6)
    order = np.lexsort((ids[:, 1], ids[:, 0]))
    points, forces, nodes, shapes, ids = [a[order] for a in (points, forces, nodes, shapes, ids)]
    assert len(np.unique(ids[:, :2], axis=0)) == count
    assert all(np.isfinite(a).all() for a in (points, forces, shapes))
    np.testing.assert_allclose(shapes.sum(axis=1), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.einsum("pi,pij->pj", shapes, observer.coordinates[nodes]), points, rtol=0, atol=1e-14)
    assembled = np.zeros_like(expected)
    np.add.at(assembled, nodes.reshape(-1), (shapes[..., None] * forces[:, None, :]).reshape(-1, 3))
    force_scale = np.linalg.norm(expected)
    assert force_scale > 0
    scatter_error = float(np.linalg.norm(assembled - expected) / force_scale)
    assert scatter_error <= 1e-10, "Captured quadrature does not reproduce actual native nodal forces"
    load_scale = np.linalg.norm(forces, axis=1).sum()
    force_error = float(np.linalg.norm(forces.sum(axis=0) - expected.sum(axis=0)) / load_scale)
    moment_error = float(
        np.linalg.norm(np.cross(points, forces).sum(axis=0) - np.cross(observer.coordinates, expected).sum(axis=0))
        / (load_scale * 0.03)
    )
    assert max(force_error, moment_error) <= 1e-10
    np.testing.assert_array_equal(qd_to_numpy(entry.state.force, transpose=True)[0], expected)
    np.testing.assert_array_equal(qd_to_numpy(entry.state.stress_tensor, transpose=True), original_stress)
    arrays = {
        "points_material_m": points,
        "quadrature_forces_n": forces,
        "p2_nodes": nodes,
        "p2_shape": shapes,
        "source_sample_face": ids,
        "native_nodal_forces_n": expected,
    }
    if entry.intrinsic is not None:
        arrays["packed_raw_contact_indices"] = qd_to_numpy(entry.intrinsic.packing.indices, transpose=True)[0]
    return arrays, {
        "samples": count,
        "signed_p2_scatter_relative_error": scatter_error,
        "force_relative_error": force_error,
        "moment_relative_error": moment_error,
        "native_force_and_stress_unchanged": True,
        "passed": True,
    }
