"""Diagnostic microkernels subdividing native finite-footprint processing."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, func_force_frame, func_patch_sample
from genesis.engine.solvers.rigid.stress.grid import func_patch_cell, func_patch_grid
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo


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
