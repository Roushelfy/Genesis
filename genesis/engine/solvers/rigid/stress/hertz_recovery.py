"""Hertz normal and radial friction fields with a finite-domain regular elastic response."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind, ErrorCode, V, V_MAT, V_VEC, maybe_shape

from .data import StressInfo, StressState
from .hertz import StressHertzState, func_boussinesq_tensor, func_hertz_particular, kernel_hertz_stress_quadrature
from .hertz_probes import (
    StressHertzProbeState,
    func_hertz_probe_location,
    func_hertz_curved_probe_location,
    kernel_hertz_probe_offsets,
)
from .hertz_tangential import StressHertzShear, func_cerruti_tensor, func_hertz_friction_particular
from .model import StressModel
from .operators import func_shape_gradient, func_material_gradients, func_material_position
from .provenance import func_closest_triangle
from .surface import StressSurfaceInfo, func_surface_geometry, func_surface_projection


@dataclass(frozen=True)
class StressHertzFrames:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    center: qd.Tensor
    basis: qd.Tensor
    anchor: qd.Tensor


@dataclass(frozen=True)
class StressHertzQueryInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    position: qd.Tensor
    element: qd.Tensor
    barycentric: qd.Tensor


@dataclass(frozen=True)
class StressHertzRecoveryState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    status: qd.Tensor
    wrench_error: qd.Tensor
    tensile_peak: qd.Tensor
    tensor: qd.Tensor


class StressHertzRecovery:
    """Recover contacts on supported planar or quadratic material boundaries.

    Regions use region-major flattened arrays, with environment as the inner
    dimension. The supplied frame's third column follows the inward source
    normal. Affine sources require four-major-radius clearance from noncoplanar
    boundaries. Quadratic boundaries use their actual normal and area measure;
    material probes follow adjacent surface and volume charts. Material outside
    a region's half-space is rejected. Optional shear data supplies radial
    friction history for each region.

    Complete output retains all element corners in StressState and additional
    material probes in this component. Both banks contribute to the sampled VM
    and maximum principal stresses. Probe metadata must locate actual material.
    """

    def __init__(
        self,
        n_envs: int,
        n_regions: int,
        n_queries: int = 0,
        output_mode: str = "max",
        far_field: bool = True,
        moving_probes: bool = False,
        probe_walk_steps: int = 64,
        shear: StressHertzShear | None = None,
    ):
        assert n_envs > 0 and n_regions > 0 and n_queries >= 0 and output_mode in ("max", "full")
        assert probe_walk_steps > 0
        n_patches = n_envs * n_regions
        self.full = output_mode == "full"
        self.far_field = far_field
        self.has_queries = n_queries > 0
        self.has_shear = shear is not None
        self.shear = shear
        if self.shear is None:
            self.shear = StressHertzShear(
                friction=V(dtype=gs.qd_float, shape=()),
                force=V_VEC(2, dtype=gs.qd_float, shape=()),
                coefficients=V_VEC(2, dtype=gs.qd_float, shape=()),
            )
        self.moving_probes, self.probe_walk_steps = moving_probes, probe_walk_steps
        n_probes = 89 if moving_probes else 0
        self.probe_offsets = V_VEC(3, dtype=gs.qd_float, shape=(n_probes,))
        self.probes = StressHertzProbeState(
            position=V_VEC(3, dtype=gs.qd_float, shape=(n_probes, n_patches)),
            element=V(dtype=gs.qd_int, shape=maybe_shape((n_probes, n_patches), moving_probes)),
            barycentric=V_VEC(4, dtype=gs.qd_float, shape=(n_probes, n_patches)),
            tensor=V_VEC(6, dtype=gs.qd_float, shape=(n_probes if self.full else 0, n_patches)),
        )
        self.frames = StressHertzFrames(
            center=V_VEC(3, dtype=gs.qd_float, shape=(n_patches,)),
            basis=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_patches,)),
            anchor=V(dtype=gs.qd_int, shape=(n_patches,)),
        )
        self.queries = StressHertzQueryInfo(
            position=V_VEC(3, dtype=gs.qd_float, shape=(n_queries,)),
            element=V(dtype=gs.qd_int, shape=maybe_shape((n_queries,), n_queries > 0)),
            barycentric=V_VEC(4, dtype=gs.qd_float, shape=(n_queries,)),
        )
        self.state = StressHertzRecoveryState(
            status=V(dtype=gs.qd_int, shape=(n_envs,)),
            wrench_error=V(dtype=gs.qd_float, shape=(n_envs,)),
            tensile_peak=V(dtype=gs.qd_float, shape=(n_envs,)),
            tensor=V_VEC(6, dtype=gs.qd_float, shape=(n_queries if self.full else 0, n_envs)),
        )
        self.quadrature = V_VEC(2, dtype=gs.qd_float, shape=(16,))
        self.frames.anchor.fill(-1)
        self.state.status.fill(2)
        kernel_hertz_stress_quadrature(self.quadrature)
        if moving_probes:
            kernel_hertz_probe_offsets(self.probe_offsets)

    def scatter(
        self,
        hertz: StressHertzState,
        state: StressState,
        info: StressInfo,
        surface: StressSurfaceInfo,
        poisson: float,
        epsilon: float,
        errno: qd.Tensor,
    ) -> None:
        assert hertz.force.shape[0] == self.frames.center.shape[0]
        assert state.active.shape[0] == self.state.status.shape[0]
        kernel_hertz_residual_begin(
            epsilon, self.frames, hertz, state, self.state, info, surface, info.volume_points.shape[0] > 0
        )
        kernel_hertz_halfspace(epsilon, self.frames, hertz, self.state, info)
        kernel_hertz_residual_traction(
            poisson,
            epsilon,
            self.frames,
            hertz,
            self.shear,
            state,
            self.state,
            info,
            surface,
            self.has_shear,
            info.volume_points.shape[0] > 0,
        )
        kernel_hertz_residual_accept(self.frames, hertz, self.shear, state, self.state, info, self.has_shear, errno)

    def recover(
        self,
        omega: qd.Tensor,
        hertz: StressHertzState,
        state: StressState,
        model: StressModel,
        surface: StressSurfaceInfo,
        epsilon: float,
        errno: qd.Tensor,
    ) -> None:
        self.scatter(hertz, state, model.info, surface, model.options.poisson, epsilon, errno)
        model.recover(omega, state, surface_load=True, compute_peak=False)
        self.evaluate(hertz, state, model, epsilon, errno)

    def evaluate(
        self, hertz: StressHertzState, state: StressState, model: StressModel, epsilon: float, errno: qd.Tensor
    ) -> None:
        assert self.full == (state.stress_tensor.shape[0] > 0)
        if self.moving_probes:
            kernel_hertz_moving_probes(
                self.probe_offsets,
                self.frames,
                hertz,
                self.state,
                self.probes,
                model.info,
                self.probe_walk_steps,
                epsilon,
                model.is_curved_geometry,
            )
        kernel_hertz_combined_field(
            model.options.young,
            model.options.poisson,
            epsilon,
            self.quadrature,
            self.probe_offsets,
            self.frames,
            hertz,
            self.shear,
            state,
            self.state,
            self.probes,
            model.info,
            self.queries,
            self.full,
            self.far_field,
            self.has_queries,
            self.moving_probes,
            self.has_shear,
            model.is_curved_geometry,
        )
        kernel_hertz_combined_accept(state, self.state, self.probes, self.full, errno)


@qd.kernel
def kernel_hertz_moving_probes(
    offsets: qd.Tensor,
    frames: StressHertzFrames,
    hertz: StressHertzState,
    state: StressHertzRecoveryState,
    probes: StressHertzProbeState,
    info: StressInfo,
    walk_steps: int,
    epsilon: float,
    is_curved: qd.template(),
):
    for i_q, i_p in qd.ndrange(offsets.shape[0], frames.center.shape[0]):
        i_b = i_p % state.status.shape[0]
        probes.element[i_q, i_p] = -1
        probes.position[i_q, i_p] = qd.Vector([float("nan")] * 3)
        probes.barycentric[i_q, i_p] = qd.Vector.zero(gs.qd_float, 4)
        if state.status[i_b] <= 1 and hertz.status[i_p] == 0:
            direction, axes = hertz.direction[i_p], hertz.axes[i_p]
            rotation = qd.Matrix(
                [[direction[0], -direction[1], 0.0], [direction[1], direction[0], 0.0], [0.0, 0.0, 1.0]]
            )
            local = offsets[i_q] * qd.Vector([axes[0], axes[1], axes[0]])
            center = frames.center[i_p]
            position = center + frames.basis[i_p] @ rotation @ local
            anchor = frames.anchor[i_p]
            i_e = info.surface_owners[anchor]
            bary = qd.Vector.zero(gs.qd_float, 4)
            if qd.static(is_curved):
                position, i_e, bary = func_hertz_curved_probe_location(
                    anchor, center, frames.basis[i_p] @ rotation, local, info, walk_steps, epsilon
                )
            else:
                i_e, bary = func_hertz_probe_location(i_e, center, position, info, walk_steps)
            probes.position[i_q, i_p], probes.element[i_q, i_p], probes.barycentric[i_q, i_p] = position, i_e, bary
            if i_e < 0:
                qd.atomic_max(state.status[i_b], 6)


@qd.kernel
def kernel_hertz_residual_begin(
    epsilon: float,
    frames: StressHertzFrames,
    hertz: StressHertzState,
    stress_state: StressState,
    state: StressHertzRecoveryState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    is_curved: qd.template(),
):
    for i_b in range(state.status.shape[0]):
        state.status[i_b], state.wrench_error[i_b] = 1, 0.0
        for i_r in range(frames.center.shape[0] // state.status.shape[0]):
            i_p = i_r * state.status.shape[0] + i_b
            status = hertz.status[i_p]
            if status != 1:
                if state.status[i_b] == 1:
                    state.status[i_b] = 0
                basis, center, anchor = frames.basis[i_p], frames.center[i_p], frames.anchor[i_p]
                valid = status == 0 and center.norm() < float("inf") and basis.norm() < float("inf")
                valid = valid and (basis.transpose() @ basis - qd.Matrix.identity(gs.qd_float, 3)).norm() < 1e-10
                valid = valid and qd.abs(basis.determinant() - 1) < 1e-10
                valid = valid and anchor >= 0 and anchor < surface.face_normal.shape[0]
                if valid:
                    closest, bary = func_closest_triangle(anchor, center, surface)
                    normal = surface.face_normal[anchor]
                    if qd.static(is_curved):
                        closest, bary, area, is_projected = func_surface_projection(
                            anchor, center, basis[:, :2].transpose(), info
                        )
                        normal = area.normalized()
                        valid = valid and is_projected and bary.min() >= -1e-10 and bary.max() <= 1 + 1e-10
                    valid = valid and (closest - center).norm() <= epsilon
                    valid = valid and normal.dot(basis[:, 2]) < -1 + 1e-10
                if not valid:
                    state.status[i_b] = 2
    for i_n, i_b in qd.ndrange(stress_state.force.shape[0], state.status.shape[0]):
        stress_state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_hertz_halfspace(
    epsilon: float,
    frames: StressHertzFrames,
    hertz: StressHertzState,
    state: StressHertzRecoveryState,
    info: StressInfo,
):
    for i_n, i_p in qd.ndrange(info.vertices.shape[0], frames.center.shape[0]):
        i_b = i_p % state.status.shape[0]
        if state.status[i_b] == 0 and hertz.status[i_p] == 0:
            depth = (info.vertices[i_n] - frames.center[i_p]).dot(frames.basis[i_p][:, 2])
            if not depth >= -epsilon:
                qd.atomic_max(state.status[i_b], 3)


@qd.kernel
def kernel_hertz_residual_traction(
    poisson: float,
    epsilon: float,
    frames: StressHertzFrames,
    hertz: StressHertzState,
    shear: StressHertzShear,
    stress_state: StressState,
    state: StressHertzRecoveryState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    has_shear: qd.template(),
    is_curved: qd.template(),
):
    for i_f, i_p, i_lane in qd.ndrange(info.surface_nodes.shape[0], frames.center.shape[0], surface.gauss.shape[0]):
        i_b = i_p % state.status.shape[0]
        if state.status[i_b] == 0 and hertz.status[i_p] == 0:
            basis, center = frames.basis[i_p], frames.center[i_p]
            if qd.static(has_shear):
                direction = hertz.direction[i_p]
                basis = basis @ qd.Matrix(
                    [[direction[0], -direction[1], 0.0], [direction[1], direction[0], 0.0], [0.0, 0.0, 1.0]]
                )
            normal = surface.face_normal[i_f]
            coplanar = True
            for i in qd.static(range(3)):
                i_n = info.surface_nodes[i_f, i]
                coplanar = coplanar and qd.abs((info.vertices[i_n] - center).dot(basis[:, 2])) <= epsilon
            if qd.static(is_curved):
                coplanar = False
            if not coplanar:
                closest, bary = func_closest_triangle(i_f, center, surface)
                is_source, clear = False, (closest - center).norm() >= 4 * hertz.axes[i_p][0]
                if qd.static(is_curved):
                    closest, tangent, area = func_surface_geometry(i_f, bary, info)
                    projected, projected_bary, projected_area, is_projected = func_surface_projection(
                        i_f, center, basis[:, :2].transpose(), info
                    )
                    is_source = (
                        is_projected
                        and projected_bary.min() >= -1e-10
                        and projected_bary.max() <= 1 + 1e-10
                        and (projected - center).norm() <= epsilon
                    )
                    clear = True
                    if is_source:
                        closest, bary = projected, projected_bary
                        clear = projected_area.normalized().dot(basis[:, 2]) < -1 + 1e-10
                if not clear:
                    qd.atomic_max(state.status[i_b], 3)
                loads = qd.Matrix.zero(gs.qd_float, 6, 3)
                if clear:
                    diameter = qd.max(
                        surface.face_edges[i_f][0, :].norm(),
                        surface.face_edges[i_f][1, :].norm(),
                        (surface.face_edges[i_f][1, :] - surface.face_edges[i_f][0, :]).norm(),
                    )
                    distance = (closest - center).norm()
                    is_near = distance < diameter / 4
                    radial_splits = 2
                    if qd.static(is_curved):
                        radial_splits = 3
                    n_rules = 6 * radial_splits if is_near else 1
                    # Each lane integrates one angular knot over every radial rule.
                    for i_sample_ in range(n_rules * surface.gauss.shape[0]):
                        i_sample_local = i_sample_ * surface.gauss.shape[0] + i_lane
                        i_q = i_sample_local % surface.shape.shape[0]
                        i_sample = i_f * surface.shape.shape[0] + i_q
                        position, weight = surface.positions[i_sample], surface.weights[i_sample]
                        shape = surface.shape[i_q]
                        if qd.static(is_curved):
                            normal = surface.normals[i_sample]
                        if is_near:
                            i_rule = i_sample_local // surface.shape.shape[0]
                            i_edge = i_rule // (2 * radial_splits)
                            first_bary, second_bary = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
                            first_bary[i_edge], second_bary[(i_edge + 1) % 3] = 1.0, 1.0
                            i_first, i_second = (
                                info.surface_nodes[i_f, i_edge],
                                info.surface_nodes[i_f, (i_edge + 1) % 3],
                            )
                            first, second = info.vertices[i_first] - closest, info.vertices[i_second] - closest
                            n_gauss = surface.gauss.shape[0]
                            u = (surface.gauss[i_q // n_gauss] + (i_rule % (2 * radial_splits)) // 2) / radial_splits
                            v = (surface.gauss[i_q % n_gauss] + i_rule % 2) / 2
                            radius, derivative = u, gs.qd_float(1.0)
                            if not is_source:
                                length = ((1 - v) * first + v * second).norm()
                                denominator = length * (1 - u) + distance * u
                                radius = distance * u / denominator
                                derivative = distance * length / (denominator * denominator)
                            sample_bary = bary + radius * ((1 - v) * first_bary + v * second_bary - bary)
                            position = closest + radius * ((1 - v) * first + v * second)
                            measure = first.cross(second).norm()
                            if qd.static(is_curved):
                                # Tangency cancels the point singularity against
                                # this source-centered fan's radial Jacobian.
                                position, tangent, area = func_surface_geometry(i_f, sample_bary, info)
                                normal = area.normalized()
                                measure = area.norm() * qd.abs(bary[(i_edge + 2) % 3])
                            weight = (
                                measure
                                * radius
                                * derivative
                                * surface.gauss_weights[i_q // n_gauss]
                                * surface.gauss_weights[i_q % n_gauss]
                                / (2 * radial_splits)
                            )
                            shape = qd.Vector(
                                [
                                    sample_bary[0] * (2 * sample_bary[0] - 1),
                                    sample_bary[1] * (2 * sample_bary[1] - 1),
                                    sample_bary[2] * (2 * sample_bary[2] - 1),
                                    4 * sample_bary[0] * sample_bary[1],
                                    4 * sample_bary[0] * sample_bary[2],
                                    4 * sample_bary[1] * sample_bary[2],
                                ]
                            )
                        local = basis.transpose() @ (position - center)
                        if qd.static(is_curved):
                            if not local[2] >= -epsilon:
                                qd.atomic_max(state.status[i_b], 3)
                        local[2] = qd.max(local[2], 0.0)
                        tensor = func_boussinesq_tensor(local, hertz.force[i_p], poisson)
                        if qd.static(has_shear):
                            if shear.force[i_p].norm() > 0:
                                tensor += func_cerruti_tensor(local, shear.force[i_p], poisson)
                        traction = -(basis @ tensor @ basis.transpose() @ normal) * weight
                        loads += shape.outer_product(traction)
                for i, j in qd.static(qd.ndrange(6, 3)):
                    i_n = info.surface_nodes[i_f, i]
                    qd.atomic_add(stress_state.force[i_n, i_b][j], loads[i, j])


@qd.kernel
def kernel_hertz_residual_accept(
    frames: StressHertzFrames,
    hertz: StressHertzState,
    shear: StressHertzShear,
    stress_state: StressState,
    state: StressHertzRecoveryState,
    info: StressInfo,
    has_shear: qd.template(),
    errno: qd.Tensor,
):
    for i_b in range(state.status.shape[0]):
        expected_force, expected_moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        load_scale, moment_scale = gs.qd_float(0.0), gs.qd_float(0.0)
        for i_r in range(frames.center.shape[0] // state.status.shape[0]):
            i_p = i_r * state.status.shape[0] + i_b
            if hertz.status[i_p] == 0:
                force = hertz.force[i_p] * frames.basis[i_p][:, 2]
                if qd.static(has_shear):
                    direction = hertz.direction[i_p]
                    rotation = qd.Matrix(
                        [[direction[0], -direction[1], 0.0], [direction[1], direction[0], 0.0], [0.0, 0.0, 1.0]]
                    )
                    force += frames.basis[i_p] @ rotation @ qd.Vector([shear.force[i_p][0], shear.force[i_p][1], 0.0])
                expected_force += force
                expected_moment += frames.center[i_p].cross(force)
                load_scale += hertz.force[i_p]
                moment_scale += hertz.force[i_p] * hertz.axes[i_p][0]
        if state.status[i_b] == 0:
            actual_force, actual_moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
            for i_n in range(stress_state.force.shape[0]):
                force = stress_state.force[i_n, i_b]
                actual_force += force
                actual_moment += info.vertices[i_n].cross(force)
            error = qd.max(
                (actual_force - expected_force).norm() / load_scale,
                (actual_moment - expected_moment).norm() / moment_scale,
            )
            state.wrench_error[i_b] = error
            if not error < 1e-4:
                state.status[i_b] = 4
        if state.status[i_b] > 1:
            qd.atomic_or(errno[i_b], ErrorCode.INVALID_STRESS_LOAD)
            for i_n in range(stress_state.force.shape[0]):
                stress_state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_hertz_combined_field(
    young: float,
    poisson: float,
    epsilon: float,
    quadrature: qd.Tensor,
    probe_offsets: qd.Tensor,
    frames: StressHertzFrames,
    hertz: StressHertzState,
    shear: StressHertzShear,
    stress_state: StressState,
    state: StressHertzRecoveryState,
    probes: StressHertzProbeState,
    info: StressInfo,
    queries: StressHertzQueryInfo,
    full: qd.template(),
    far_field: qd.template(),
    has_queries: qd.template(),
    moving_probes: qd.template(),
    has_shear: qd.template(),
    is_curved: qd.template(),
):
    for i_b in range(state.status.shape[0]):
        stress_state.peak[i_b], state.tensile_peak[i_b] = 0.0, -float("inf")
    for i_q, i_b in qd.ndrange(
        4 * info.elements.shape[0]
        + queries.position.shape[0]
        + probes.position.shape[0] * (frames.center.shape[0] // state.status.shape[0]),
        state.status.shape[0],
    ):
        i_e, bary, position = i_q // 4, qd.Vector.zero(gs.qd_float, 4), qd.Vector.zero(gs.qd_float, 3)
        active_query, is_auto_query, i_probe, i_source = True, False, 0, 0
        if i_q < 4 * info.elements.shape[0]:
            i_n = info.elements[i_e, i_q % 4]
            bary[i_q % 4] = 1.0
            position = info.vertices[i_n]
        elif i_q < 4 * info.elements.shape[0] + queries.position.shape[0]:
            if qd.static(has_queries):
                i_extra = i_q - 4 * info.elements.shape[0]
                i_e, bary, position = queries.element[i_extra], queries.barycentric[i_extra], queries.position[i_extra]
        else:
            if qd.static(moving_probes):
                i_auto = i_q - 4 * info.elements.shape[0] - queries.position.shape[0]
                is_auto_query = True
                i_probe = i_auto % probes.position.shape[0]
                i_source = (i_auto // probes.position.shape[0]) * state.status.shape[0] + i_b
                i_e = probes.element[i_probe, i_source]
                bary, position = probes.barycentric[i_probe, i_source], probes.position[i_probe, i_source]
                active_query = hertz.status[i_source] == 0
        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        if active_query and state.status[i_b] <= 1 and stress_state.valid[i_b]:
            valid = i_e >= 0 and i_e < info.elements.shape[0] and bary.min() >= -1e-10 and bary.max() <= 1 + 1e-10
            valid = valid and qd.abs(bary.sum() - 1) < 1e-10 and position.norm() < float("inf")
            if valid:
                i_origin = info.elements[i_e, 0]
                origin = stress_state.displacement[i_origin, i_b]
                derivative = qd.Matrix.zero(gs.qd_float, 3, 3)
                gradients = func_material_gradients(i_e, bary, info, is_curved)
                located = func_material_position(i_e, bary, info, is_curved)
                for i in range(10):
                    i_n = info.elements[i_e, i]
                    gradient = func_shape_gradient(i, bary, gradients, info.edges)
                    derivative += (stress_state.displacement[i_n, i_b] - origin).outer_product(gradient)
                valid = (located - position).norm() <= epsilon
                tensor = young / (2 * (1 + poisson)) * (derivative + derivative.transpose())
                for i in qd.static(range(3)):
                    tensor[i, i] += young * poisson / ((1 + poisson) * (1 - 2 * poisson)) * derivative.trace()
                for i_r in range(frames.center.shape[0] // state.status.shape[0]):
                    i_p = i_r * state.status.shape[0] + i_b
                    if hertz.status[i_p] == 0:
                        direction = hertz.direction[i_p]
                        rotation = qd.Matrix(
                            [[direction[0], -direction[1], 0.0], [direction[1], direction[0], 0.0], [0.0, 0.0, 1.0]]
                        )
                        basis = frames.basis[i_p] @ rotation
                        local = basis.transpose() @ (position - frames.center[i_p])
                        valid = valid and local[2] >= -epsilon
                        local[2] = qd.max(local[2], 0.0)
                        axes = hertz.axes[i_p]
                        if qd.static(moving_probes):
                            if is_auto_query and i_p == i_source:
                                # The probe is defined in this source frame. Retain exact
                                # surface depth instead of round-tripping world coordinates.
                                source_local = probe_offsets[i_probe] * qd.Vector([axes[0], axes[1], axes[0]])
                                if qd.static(is_curved):
                                    source_local[2] = local[2]
                                local = source_local
                        particular, resolved = qd.Matrix.zero(gs.qd_float, 3, 3), True
                        if qd.static(has_shear):
                            particular, resolved = func_hertz_friction_particular(
                                i_p,
                                local,
                                axes,
                                hertz.peak_pressure[i_p],
                                hertz.force[i_p],
                                poisson,
                                quadrature,
                                shear,
                                far_field,
                            )
                        else:
                            particular, resolved = func_hertz_particular(
                                local, axes, hertz.peak_pressure[i_p], hertz.force[i_p], poisson, quadrature, far_field
                            )
                        tensor += basis @ particular @ basis.transpose()
                        valid = valid and resolved
                valid = valid and tensor.norm() < float("inf")
            if not valid:
                qd.atomic_max(state.status[i_b], 6)
        deviator = tensor - qd.Matrix.identity(gs.qd_float, 3) * (tensor.trace() / 3)
        vm = qd.sqrt(1.5 * (deviator * deviator).sum())
        if active_query:
            qd.atomic_max(stress_state.peak[i_b], vm)
        scale = qd.sqrt((deviator * deviator).sum() / 6)
        maximum = tensor.trace() / 3
        if scale > 0:
            determinant = (deviator / scale).determinant() / 2
            maximum += 2 * scale * qd.cos(qd.acos(qd.min(1.0, qd.max(-1.0, determinant))) / 3)
        if active_query:
            qd.atomic_max(state.tensile_peak[i_b], maximum)
        if qd.static(full):
            components = qd.Vector([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])
            if i_q < 4 * info.elements.shape[0]:
                stress_state.stress_tensor[i_e, i_q % 4, i_b] = components
                stress_state.von_mises[i_e, i_q % 4, i_b] = qd.Vector([vm])
            elif i_q < 4 * info.elements.shape[0] + queries.position.shape[0]:
                i_extra = i_q - 4 * info.elements.shape[0]
                state.tensor[i_extra, i_b] = components
            else:
                if qd.static(moving_probes):
                    probes.tensor[i_probe, i_source] = components


@qd.kernel
def kernel_hertz_combined_accept(
    stress_state: StressState,
    state: StressHertzRecoveryState,
    probes: StressHertzProbeState,
    full: qd.template(),
    errno: qd.Tensor,
):
    for i_b in range(state.status.shape[0]):
        if not stress_state.valid[i_b]:
            qd.atomic_max(state.status[i_b], 5)
            qd.atomic_or(errno[i_b], ErrorCode.INVALID_STRESS_SOLVE)
        stress_state.step_valid[i_b] = stress_state.step_valid[i_b] and state.status[i_b] <= 1
        if not stress_state.step_valid[i_b]:
            qd.atomic_or(errno[i_b], ErrorCode.INVALID_STRESS_LOAD)
            if not stress_state.invalid_reported[i_b]:
                stress_state.invalid_steps[i_b] += 1
                stress_state.invalid_reported[i_b] = True
            stress_state.peak[i_b], state.tensile_peak[i_b] = float("nan"), float("nan")
            stress_state.step_peak[i_b] = float("nan")
        else:
            stress_state.step_peak[i_b] = qd.max(stress_state.step_peak[i_b], stress_state.peak[i_b])
    if qd.static(full):
        for i_e, i_corner, i_b in qd.ndrange(stress_state.stress_tensor.shape[0], 4, state.status.shape[0]):
            if not stress_state.step_valid[i_b]:
                stress_state.stress_tensor[i_e, i_corner, i_b] = qd.Vector([float("nan")] * 6)
                stress_state.von_mises[i_e, i_corner, i_b] = qd.Vector([float("nan")])
        for i_q, i_b in qd.ndrange(state.tensor.shape[0], state.status.shape[0]):
            if not stress_state.step_valid[i_b]:
                state.tensor[i_q, i_b] = qd.Vector([float("nan")] * 6)
        for i_q, i_p in qd.ndrange(probes.tensor.shape[0], probes.tensor.shape[1]):
            i_b = i_p % state.status.shape[0]
            if not stress_state.step_valid[i_b]:
                probes.tensor[i_q, i_p] = qd.Vector([float("nan")] * 6)
