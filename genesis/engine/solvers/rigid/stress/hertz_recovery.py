"""Normal Hertz fields with a finite-domain regular elastic response."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind, ErrorCode, V, V_MAT, V_VEC, maybe_shape

from .data import StressInfo, StressState
from .hertz import StressHertzState, func_boussinesq_tensor, func_hertz_particular, kernel_hertz_stress_quadrature
from .model import StressModel
from .operators import func_shape_gradient
from .provenance import func_closest_triangle
from .surface import StressSurfaceInfo


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
    """Recover normal contacts whose footprints lie on supported flat patches.

    Regions use region-major flattened arrays, with environment as the inner
    dimension. The supplied frame's third column points into material. The
    actual anchor and a four-major-radius clearance from noncoplanar boundaries
    are checked. Material outside a region's half-space is rejected explicitly.
    This normal-only component does not infer contact association or friction.

    Complete output retains all element corners in StressState and additional
    material probes in this component. Both banks contribute to the sampled VM
    and maximum principal stresses. Probe metadata must locate actual material.
    """

    def __init__(
        self, n_envs: int, n_regions: int, n_queries: int = 0, output_mode: str = "max", far_field: bool = True
    ):
        assert n_envs > 0 and n_regions > 0 and n_queries >= 0 and output_mode in ("max", "full")
        n_patches = n_envs * n_regions
        self.full = output_mode == "full"
        self.far_field = far_field
        self.has_queries = n_queries > 0
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
        kernel_hertz_residual_begin(epsilon, self.frames, hertz, state, self.state, surface)
        kernel_hertz_halfspace(epsilon, self.frames, hertz, self.state, info)
        kernel_hertz_residual_traction(poisson, epsilon, self.frames, hertz, state, self.state, info, surface)
        kernel_hertz_residual_accept(self.frames, hertz, state, self.state, info, errno)

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
        kernel_hertz_combined_field(
            model.options.young,
            model.options.poisson,
            epsilon,
            self.quadrature,
            self.frames,
            hertz,
            state,
            self.state,
            model.info,
            self.queries,
            self.full,
            self.far_field,
            self.has_queries,
        )
        kernel_hertz_combined_accept(state, self.state, self.full, errno)


@qd.kernel
def kernel_hertz_residual_begin(
    epsilon: float,
    frames: StressHertzFrames,
    hertz: StressHertzState,
    stress_state: StressState,
    state: StressHertzRecoveryState,
    surface: StressSurfaceInfo,
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
                    valid = (closest - center).norm() <= epsilon
                    valid = valid and surface.face_normal[anchor].dot(basis[:, 2]) < -1 + 1e-10
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
    stress_state: StressState,
    state: StressHertzRecoveryState,
    info: StressInfo,
    surface: StressSurfaceInfo,
):
    for i_f, i_p in qd.ndrange(info.surface_nodes.shape[0], frames.center.shape[0]):
        i_b = i_p % state.status.shape[0]
        if state.status[i_b] == 0 and hertz.status[i_p] == 0:
            basis, center = frames.basis[i_p], frames.center[i_p]
            normal = surface.face_normal[i_f]
            coplanar = True
            for i in qd.static(range(3)):
                i_n = info.surface_nodes[i_f, i]
                coplanar = coplanar and qd.abs((info.vertices[i_n] - center).dot(basis[:, 2])) <= epsilon
            if not coplanar:
                closest, bary = func_closest_triangle(i_f, center, surface)
                clear = (closest - center).norm() >= 4 * hertz.axes[i_p][0]
                if not clear:
                    qd.atomic_max(state.status[i_b], 3)
                loads = qd.Matrix.zero(gs.qd_float, 6, 3)
                if clear:
                    for i_q in range(surface.shape.shape[0]):
                        i_sample = i_f * surface.shape.shape[0] + i_q
                        local = basis.transpose() @ (surface.positions[i_sample] - center)
                        local[2] = qd.max(local[2], 0.0)
                        tensor = func_boussinesq_tensor(local, hertz.force[i_p], poisson)
                        traction = -(basis @ tensor @ basis.transpose() @ normal) * surface.weights[i_sample]
                        loads += surface.shape[i_q].outer_product(traction)
                for i, j in qd.static(qd.ndrange(6, 3)):
                    i_n = info.surface_nodes[i_f, i]
                    qd.atomic_add(stress_state.force[i_n, i_b][j], loads[i, j])


@qd.kernel
def kernel_hertz_residual_accept(
    frames: StressHertzFrames,
    hertz: StressHertzState,
    stress_state: StressState,
    state: StressHertzRecoveryState,
    info: StressInfo,
    errno: qd.Tensor,
):
    for i_b in range(state.status.shape[0]):
        expected_force, expected_moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        load_scale, moment_scale = gs.qd_float(0.0), gs.qd_float(0.0)
        for i_r in range(frames.center.shape[0] // state.status.shape[0]):
            i_p = i_r * state.status.shape[0] + i_b
            if hertz.status[i_p] == 0:
                force = hertz.force[i_p] * frames.basis[i_p][:, 2]
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
    frames: StressHertzFrames,
    hertz: StressHertzState,
    stress_state: StressState,
    state: StressHertzRecoveryState,
    info: StressInfo,
    queries: StressHertzQueryInfo,
    full: qd.template(),
    far_field: qd.template(),
    has_queries: qd.template(),
):
    for i_b in range(state.status.shape[0]):
        stress_state.peak[i_b], state.tensile_peak[i_b] = 0.0, -float("inf")
    for i_q, i_b in qd.ndrange(4 * info.elements.shape[0] + queries.position.shape[0], state.status.shape[0]):
        i_e, bary, position = i_q // 4, qd.Vector.zero(gs.qd_float, 4), qd.Vector.zero(gs.qd_float, 3)
        if i_q < 4 * info.elements.shape[0]:
            i_n = info.elements[i_e, i_q % 4]
            bary[i_q % 4] = 1.0
            position = info.vertices[i_n]
        else:
            if qd.static(has_queries):
                i_extra = i_q - 4 * info.elements.shape[0]
                i_e, bary, position = queries.element[i_extra], queries.barycentric[i_extra], queries.position[i_extra]
        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        if state.status[i_b] <= 1 and stress_state.valid[i_b]:
            valid = i_e >= 0 and i_e < info.elements.shape[0] and bary.min() >= -1e-10 and bary.max() <= 1 + 1e-10
            valid = valid and qd.abs(bary.sum() - 1) < 1e-10 and position.norm() < float("inf")
            if valid:
                i_origin = info.elements[i_e, 0]
                origin = stress_state.displacement[i_origin, i_b]
                derivative, located = qd.Matrix.zero(gs.qd_float, 3, 3), qd.Vector.zero(gs.qd_float, 3)
                for i in range(10):
                    i_n = info.elements[i_e, i]
                    gradient = func_shape_gradient(i, bary, info.gradients[i_e], info.edges)
                    derivative += (stress_state.displacement[i_n, i_b] - origin).outer_product(gradient)
                    if i < 4:
                        located += bary[i] * info.vertices[i_n]
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
        qd.atomic_max(stress_state.peak[i_b], vm)
        scale = qd.sqrt((deviator * deviator).sum() / 6)
        maximum = tensor.trace() / 3
        if scale > 0:
            determinant = (deviator / scale).determinant() / 2
            maximum += 2 * scale * qd.cos(qd.acos(qd.min(1.0, qd.max(-1.0, determinant))) / 3)
        qd.atomic_max(state.tensile_peak[i_b], maximum)
        if qd.static(full):
            components = qd.Vector([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])
            if i_q < 4 * info.elements.shape[0]:
                stress_state.stress_tensor[i_e, i_q % 4, i_b] = components
                stress_state.von_mises[i_e, i_q % 4, i_b] = qd.Vector([vm])
            else:
                i_extra = i_q - 4 * info.elements.shape[0]
                state.tensor[i_extra, i_b] = components


@qd.kernel
def kernel_hertz_combined_accept(
    stress_state: StressState, state: StressHertzRecoveryState, full: qd.template(), errno: qd.Tensor
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
