"""Capture native quadrature forces and physical area measures for offline load-model audits."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, func_contact_sample, func_force_frame
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionInfo,
    StressTractionSamples,
    StressTractionState,
    func_circle_panel,
    func_source_traction,
    func_traction_constraint_active,
    func_traction_sample_index,
    func_traction_support_range,
)


@qd.kernel
def kernel_continuous_boundary_load(
    tensor: qd.types.ndarray(),
    bending_scale: float,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
):
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.force.shape[1]):
        state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_q, i_b in qd.ndrange(surface.positions.shape[0], state.force.shape[1]):
        stress = qd.Matrix([[tensor[i, j] for j in qd.static(range(3))] for i in qd.static(range(3))])
        stress[0, 0] += bending_scale * surface.positions[i_q][2]
        n_quadrature = surface.shape.shape[0]
        i_face = i_q // n_quadrature
        force = stress @ surface.face_normal[i_face] * surface.weights[i_q]
        for i_local in qd.static(range(6)):
            i_n = info.surface_nodes[i_face, i_local]
            for i_a in qd.static(range(3)):
                qd.atomic_add(state.force[i_n, i_b][i_a], surface.shape[i_q % n_quadrature][i_local] * force[i_a])


@qd.kernel
def kernel_intrinsic_load_samples(
    forces: qd.Tensor,
    areas: qd.Tensor,
    contacts: StressContactState,
    tractions: StressTractionState,
    info: StressTractionInfo,
    surface: StressSurfaceInfo,
):
    for i_q, i_s, i_b in qd.ndrange(forces.shape[0], forces.shape[1], forces.shape[2]):
        forces[i_q, i_s, i_b] = qd.Vector.zero(gs.qd_float, 3)
        areas[i_q, i_s, i_b] = 0.0
        if i_q < tractions.sample_count[i_s, i_b]:
            for i_c in range(contacts.valid.shape[0]):
                start, end = func_traction_support_range(i_c, i_b, tractions)
                if func_traction_constraint_active(i_c, i_b, tractions) and (start <= i_s < end):
                    forces[i_q, i_s, i_b] += func_source_traction(i_q, i_s, i_c, i_b, contacts, tractions, info)
            i_sample = func_traction_sample_index(i_q, i_s, i_b, tractions)
            n = info.gauss.shape[0]
            i_local = i_sample % (n * n)
            u, v = info.gauss[i_local // n], info.gauss[i_local % n]
            power = tractions.fan_coordinate_power[None]
            circle_start = 3 * info.positions.shape[0]
            area = gs.qd_float(0.0)
            if i_sample >= circle_start:
                radius = tractions.circle.radius[i_s, i_b]
                if tractions.circle.refined[i_s, i_b]:
                    distance, _, jacobian = func_circle_panel(i_sample - circle_start, i_s, i_b, tractions, info)
                    area = radius * radius * distance * jacobian
                else:
                    area = 2.0 * qd.math.pi * radius * radius * power * u ** (2 * power - 1)
            else:
                i_face = i_sample // (3 * n * n)
                i_fan = (i_sample // (n * n)) % 3
                edges = info.face_edges[i_face]
                area = tractions.fan_barycentric[i_face, i_s, i_b][(i_fan + 2) % 3]
                area *= edges[0, :].cross(edges[1, :]).norm()
                area *= u**power * power * power * u ** (power - 1) * v ** (power - 1)
            areas[i_q, i_s, i_b] = area * surface.gauss_weights[i_local // n] * surface.gauss_weights[i_local % n]


@qd.kernel
def kernel_pressure_load_samples(
    samples: StressTractionSamples,
    forces: qd.Tensor,
    areas: qd.Tensor,
    contacts: StressContactState,
    surface: StressSurfaceInfo,
):
    for i_q, i_c, i_b in qd.ndrange(forces.shape[0], forces.shape[1], forces.shape[2]):
        forces[i_q, i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        areas[i_q, i_c, i_b] = 0.0
        samples.faces[i_q, i_c, i_b] = -1
        samples.positions[i_q, i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        samples.normals[i_q, i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        samples.shape[i_q, i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
        if contacts.valid[i_c, i_b] and contacts.status[i_c, i_b] == 0:
            is_sample = i_q < surface.positions.shape[0] or contacts.is_refined[i_c, i_b]
            if is_sample:
                first, second = func_force_frame(contacts.force[i_c, i_b])
                coordinates, weight, position, basis, i_face = func_contact_sample(
                    i_q, i_c, i_b, first, second, contacts, surface
                )
                if weight > 0.0:
                    relative = (position - contacts.center[i_c, i_b]) / contacts.radius[i_c, i_b]
                    profile = qd.exp(-0.5 * relative.dot(relative) / (0.45 * 0.45))
                    profile *= (1.0 - relative.dot(relative)) ** 2
                    fraction = weight * qd.max(0.0, coordinates.dot(contacts.coefficient[i_c, i_b]))
                    forces[i_q, i_c, i_b] = fraction * contacts.force[i_c, i_b]
                    areas[i_q, i_c, i_b] = weight / profile
                    samples.positions[i_q, i_c, i_b] = position
                    samples.normals[i_q, i_c, i_b] = -surface.face_normal[i_face]
                    samples.faces[i_q, i_c, i_b] = i_face
                    samples.shape[i_q, i_c, i_b] = basis
