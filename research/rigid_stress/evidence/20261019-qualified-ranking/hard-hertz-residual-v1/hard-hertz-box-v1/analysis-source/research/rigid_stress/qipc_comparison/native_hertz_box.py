"""Bounded native experiment: normal Hertz plus a residual on a flat-faced box.

The prescribed contact plane is z=0, material is at z>=0 and each environment
has one supplied Hertz ellipse. This is a throughput/field-combination control,
not an arbitrary curved-body contact model or a public scene integration.
"""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.utils.array_class import V, V_MAT, V_VEC

from .native_hertz import HertzState
from .native_hertz_stress import func_hertz_tensor, kernel_hertz_stress_quadrature


@dataclass(frozen=True)
class BoxQueries:
    position: qd.Tensor
    element: qd.Tensor
    barycentric: qd.Tensor
    center: qd.Tensor
    status: qd.Tensor
    wrench_error: qd.Tensor
    peak: qd.Tensor
    tensor: qd.Tensor


class NativeHertzBox:
    def __init__(self, count, positions, elements, barycentric, full=True):
        self.quadrature = V_VEC(2, dtype=gs.qd_float, shape=(16,))
        self.query = BoxQueries(
            position=V_VEC(3, dtype=gs.qd_float, shape=(len(positions),)),
            element=V(dtype=gs.qd_int, shape=(len(positions),)),
            barycentric=V_VEC(4, dtype=gs.qd_float, shape=(len(positions),)),
            center=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            status=V(dtype=gs.qd_int, shape=(count,)),
            wrench_error=V(dtype=gs.qd_float, shape=(count,)),
            peak=V(dtype=gs.qd_float, shape=(count,)),
            tensor=V_MAT(3, 3, dtype=gs.qd_float, shape=(len(positions) if full else 0, count)),
        )
        self.query.position.from_numpy(positions.astype(gs.np_float))
        self.query.element.from_numpy(elements.astype(gs.np_int))
        self.query.barycentric.from_numpy(barycentric.astype(gs.np_float))
        self.query.center.fill(0)
        kernel_hertz_stress_quadrature(self.quadrature)

    def scatter(self, hertz, state, info, surface):
        kernel_box_clear(self.query, hertz, state)
        kernel_box_traction(self.query, hertz, state, info, surface, self.quadrature)
        kernel_box_accept(self.query, hertz, state, info)

    def evaluate(self, hertz, state, info):
        kernel_box_field(self.query, hertz, state, info, self.quadrature)


@qd.func
def func_box_tensor(position, center, hertz: HertzState, env, quadrature: qd.template()):
    direction = hertz.direction[env]
    frame = qd.Matrix([[direction[0], -direction[1], 0.0], [direction[1], direction[0], 0.0], [0.0, 0.0, 1.0]])
    local = frame.transpose() @ (position - center)
    axes = hertz.axes[env]
    tensor, resolved = func_hertz_tensor(local / axes[0], axes[1] / axes[0], 0.3, quadrature)
    return hertz.peak_pressure[env] * (frame @ tensor @ frame.transpose()), resolved


@qd.kernel
def kernel_box_clear(query: BoxQueries, hertz: HertzState, state: StressState):
    for env in range(query.status.shape[0]):
        status = hertz.status[env]
        center = query.center[env]
        if not (center.norm() < float("inf") and center[2] == 0):
            status = 2
        query.status[env], query.wrench_error[env], query.peak[env] = status, 0.0, 0.0
    for node, env in qd.ndrange(state.force.shape[0], state.force.shape[1]):
        state.force[node, env] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_box_traction(
    query: BoxQueries,
    hertz: HertzState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    quadrature: qd.types.ndarray(),
):
    for face, env in qd.ndrange(info.surface_nodes.shape[0], query.status.shape[0]):
        if query.status[env] == 0:
            normal = surface.face_normal[face]
            # On the complete contact plane prescribed and particular Hertz
            # tractions cancel analytically, including a sub-face footprint.
            if normal[2] > -1 + 1e-12:
                loads = qd.Matrix.zero(gs.qd_float, 6, 3)
                for q in range(surface.shape.shape[0]):
                    sample = face * surface.shape.shape[0] + q
                    tensor, resolved = func_box_tensor(
                        surface.positions[sample], query.center[env], hertz, env, quadrature
                    )
                    if not resolved or not tensor.norm() < float("inf"):
                        qd.atomic_max(query.status[env], 3)
                    traction = -(tensor @ normal) * surface.weights[sample]
                    loads += surface.shape[q].outer_product(traction)
                for i, j in qd.static(qd.ndrange(6, 3)):
                    qd.atomic_add(state.force[info.surface_nodes[face, i], env][j], loads[i, j])


@qd.kernel
def kernel_box_accept(query: BoxQueries, hertz: HertzState, state: StressState, info: StressInfo):
    for env in range(query.status.shape[0]):
        force, moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        for node in range(state.force.shape[0]):
            current = state.force[node, env]
            force += current
            moment += (info.vertices[node] - query.center[env]).cross(current)
        if query.status[env] == 0:
            fn, a = hertz.force[env], hertz.axes[env][0]
            error = qd.max((force - qd.Vector([0.0, 0.0, fn])).norm() / fn, moment.norm() / (fn * a))
            query.wrench_error[env] = error
            if not error < 1e-4:
                query.status[env] = 4
        if query.status[env] != 0:
            for node in range(state.force.shape[0]):
                state.force[node, env] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_box_field(
    query: BoxQueries,
    hertz: HertzState,
    state: StressState,
    info: StressInfo,
    quadrature: qd.types.ndarray(),
):
    for env in range(query.peak.shape[0]):
        query.peak[env] = 0.0
    for sample, env in qd.ndrange(query.position.shape[0], query.status.shape[0]):
        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        if query.status[env] == 0 and state.valid[env]:
            element, bary = query.element[sample], query.barycentric[sample]
            derivative = qd.Matrix.zero(gs.qd_float, 3, 3)
            origin = state.displacement[info.elements[element, 0], env]
            for i in qd.static(range(4)):
                displacement = state.displacement[info.elements[element, i], env] - origin
                derivative += displacement.outer_product((4 * bary[i] - 1) * info.gradients[element][i, :])
            for edge in qd.static(range(6)):
                i, j = info.edges[edge, 0], info.edges[edge, 1]
                gradient = 4 * (bary[i] * info.gradients[element][j, :] + bary[j] * info.gradients[element][i, :])
                displacement = state.displacement[info.elements[element, 4 + edge], env] - origin
                derivative += displacement.outer_product(gradient)
            tensor = (1e10 / 2.6) * (derivative + derivative.transpose())
            for i in qd.static(range(3)):
                tensor[i, i] += (1e10 * 0.3 / (1.3 * 0.4)) * derivative.trace()
            particular, resolved = func_box_tensor(query.position[sample], query.center[env], hertz, env, quadrature)
            tensor += particular
            if not resolved or not tensor.norm() < float("inf"):
                qd.atomic_max(query.status[env], 3)
        elif query.status[env] != 1:
            tensor.fill(float("nan"))
        deviator = tensor - qd.Matrix.identity(gs.qd_float, 3) * (tensor.trace() / 3)
        vm = qd.sqrt(1.5 * (deviator * deviator).sum())
        qd.atomic_max(query.peak[env], vm)
        if qd.static(query.tensor.shape[0] > 0):
            query.tensor[sample, env] = tensor
    for env in range(query.peak.shape[0]):
        if (query.status[env] != 0 and query.status[env] != 1) or not state.valid[env]:
            query.peak[env] = float("nan")
    for sample, env in qd.ndrange(query.tensor.shape[0], query.status.shape[0]):
        if (query.status[env] != 0 and query.status[env] != 1) or not state.valid[env]:
            query.tensor[sample, env] = qd.Matrix([[float("nan") for _ in range(3)] for _ in range(3)])
