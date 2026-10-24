"""Diagnostic microkernels subdividing native finite-footprint processing."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import (
    StressContactState,
    func_contact_section,
    func_force_frame,
    func_patch_sample,
)
from genesis.engine.solvers.rigid.stress.automatic_radius import func_estimate_radius
from genesis.engine.solvers.rigid.stress.grid import func_patch_cell, func_patch_grid
from genesis.engine.solvers.rigid.stress.scatter import StressScatterWorkspace
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.utils import array_class


@qd.kernel
def kernel_moment_geometry_counts(
    contacts: StressContactState, surface: StressSurfaceInfo, workspace: StressScatterWorkspace, counts: qd.Tensor
):
    for i in range(counts.shape[0]):
        counts[i] = 0
    for i_task in range(qd.i32(qd.select(workspace.count[None] <= workspace.tasks.shape[0], workspace.count[None], 0))):
        task = workspace.tasks[i_task]
        pair = contacts.active_pairs[task[0]]
        i_c, i_b, i_f = pair[0], pair[1], task[1]
        category = 0
        if contacts.is_refined[i_c, i_b]:
            closest, _remaining, contained = func_contact_section(
                i_f, contacts.center[i_c, i_b], contacts.radius[i_c, i_b], surface
            )
            category = 3
            if contained:
                category = qd.select(contacts.hertz_profile[None], 1, 2)
            else:
                vertices = qd.Matrix.zero(gs.qd_float, 4, 3)
                n_fans = 3
                if surface.quad_vertices.shape[0] > 0:
                    n_fans = 4
                    normal = surface.face_normal[i_f]
                    origin = surface.face_origin[i_f]
                    for i in qd.static(range(4)):
                        vertex = surface.quad_vertices[i_f][i, :]
                        vertices[i, :] = vertex - (vertex - origin).dot(normal) * normal
                else:
                    vertices[0, :] = surface.face_origin[i_f]
                    vertices[1, :] = surface.face_origin[i_f] + surface.face_edges[i_f][0, :]
                    vertices[2, :] = surface.face_origin[i_f] + surface.face_edges[i_f][1, :]
                for i in range(n_fans):
                    first, second = vertices[i, :] - closest, vertices[(i + 1) % n_fans, :] - closest
                    area = first.cross(second).norm()
                    qd.atomic_add(counts[qd.select(area <= 1e-14 * first.norm() * second.norm(), 4, 5)], 1)
        qd.atomic_add(counts[category], 1)


@qd.kernel
def kernel_estimate_radii(
    contacts: StressContactState,
    collider: array_class.ColliderState,
    dyn: array_class.DynState,
    info: qd.template(),
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[i_c, i_b]:
            source = collider.contact_sort_idx[i_c, i_b]
            func_estimate_radius(i_c, i_b, source, contacts, collider, dyn, info)


@qd.kernel
def kernel_restore_geometry(position: qd.types.ndarray(), quaternion: qd.types.ndarray(), dyn: array_class.DynState):
    for i_g, i_b in qd.ndrange(dyn.geoms.pos.shape[0], dyn.geoms.pos.shape[1]):
        for axis in qd.static(range(3)):
            dyn.geoms.pos[i_g, i_b][axis] = position[i_b, i_g, axis]
        for axis in qd.static(range(4)):
            dyn.geoms.quat[i_g, i_b][axis] = quaternion[i_b, i_g, axis]


@qd.kernel
def kernel_query(contacts: StressContactState, surface: StressSurfaceInfo, visited: qd.Tensor, inside: qd.Tensor):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        visited[i_c, i_b] = 0
        inside[i_c, i_b] = 0
        if contacts.valid[i_c, i_b] and contacts.status[i_c, i_b] == 0 and contacts.force[i_c, i_b].norm() > 1e-30:
            low, dimensions = func_patch_grid(contacts.center[i_c, i_b], contacts.radius[i_c, i_b], surface.grid)
            for i_cell in range(dimensions[0] * dimensions[1] * dimensions[2]):
                start, end = func_patch_cell(i_cell, low, dimensions, surface.grid)
                visited[i_c, i_b] += end - start
                for entry in range(start, end):
                    i_q = surface.grid.samples[entry]
                    relative = (surface.positions[i_q] - contacts.center[i_c, i_b]) / contacts.radius[i_c, i_b]
                    if relative.dot(relative) < 1.0:
                        inside[i_c, i_b] += 1


@qd.kernel
def kernel_weights_gram(contacts: StressContactState, surface: StressSurfaceInfo, grams: qd.Tensor):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[i_c, i_b] and contacts.status[i_c, i_b] == 0 and contacts.force[i_c, i_b].norm() > 1e-30:
            center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
            first, second = func_force_frame(contacts.force[i_c, i_b])
            low, dimensions = func_patch_grid(center, radius, surface.grid)
            gram = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i_cell in range(dimensions[0] * dimensions[1] * dimensions[2]):
                start, end = func_patch_cell(i_cell, low, dimensions, surface.grid)
                for entry in range(start, end):
                    i_q = surface.grid.samples[entry]
                    coordinates, weight = func_patch_sample(i_q, center, first, second, radius, surface)
                    gram += weight * coordinates.outer_product(coordinates)
            grams[i_c, i_b] = gram


@qd.kernel
def kernel_small_solve(contacts: StressContactState, grams: qd.Tensor, coefficients: qd.Tensor):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[i_c, i_b] and contacts.status[i_c, i_b] == 0 and contacts.force[i_c, i_b].norm() > 1e-30:
            gram = grams[i_c, i_b]
            if gram[0, 0] > 0.0 and gram.determinant() > 1e-14 * gram.trace() ** 3:
                gram /= gram[0, 0]
                coefficients[i_c, i_b] = gram.inverse() @ qd.Vector([1.0, 0.0, 0.0])
