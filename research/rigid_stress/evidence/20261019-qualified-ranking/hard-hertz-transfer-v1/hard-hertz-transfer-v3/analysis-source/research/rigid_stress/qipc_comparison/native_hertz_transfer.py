"""Transfer finite Hertz regions through their witnessed boundary triangles.

Quadrature follows the physical ellipse, including footprints smaller than a
mesh face. Projected triangle walks preserve boundary connectivity. A constant
traction direction is an explicitly provisional tangential approximation;
its actual force/moment discrepancy remains visible and can reject transfer.
"""

from dataclasses import dataclass
from math import pi

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.utils.array_class import V, V_VEC

from .native_hertz import HertzState


@dataclass(frozen=True)
class HertzPatchState:
    environment: qd.Tensor
    anchor: qd.Tensor
    center: qd.Tensor
    normal: qd.Tensor
    tangent: qd.Tensor
    force: qd.Tensor
    moment: qd.Tensor
    status: qd.Tensor
    force_error: qd.Tensor
    moment_error: qd.Tensor


@dataclass(frozen=True)
class HertzSampleState:
    face: qd.Tensor
    barycentric: qd.Tensor
    position: qd.Tensor
    weight: qd.Tensor
    admitted: qd.Tensor


class HertzSurfaceTransfer:
    """Retain exact-size patch/sample buffers and scatter only admitted environments.

    Patch moments are about the supplied center. Status 0 is admitted, 1 is
    unloaded, 2 invalid Hertz input, 3 an unresolved surface walk, 4 unmatched
    wrench and 5 invalid patch input. An unresolved region never produces a
    partial environment load. The caller still owns production error reporting.
    """

    def __init__(self, count: int, n_envs: int, angles: int = 32):
        assert count > 0 and n_envs > 0 and angles >= 8 and angles % 4 == 0
        self.patches = HertzPatchState(
            environment=V(dtype=gs.qd_int, shape=(count,)),
            anchor=V(dtype=gs.qd_int, shape=(count,)),
            center=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            normal=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            tangent=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            force=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            moment=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            status=V(dtype=gs.qd_int, shape=(count,)),
            force_error=V(dtype=gs.qd_float, shape=(count,)),
            moment_error=V(dtype=gs.qd_float, shape=(count,)),
        )
        self.samples = HertzSampleState(
            face=V(dtype=gs.qd_int, shape=(3 * angles, count)),
            barycentric=V_VEC(3, dtype=gs.qd_float, shape=(3 * angles, count)),
            position=V_VEC(3, dtype=gs.qd_float, shape=(3 * angles, count)),
            weight=V(dtype=gs.qd_float, shape=(3 * angles,)),
            admitted=V(dtype=gs.qd_bool, shape=(n_envs,)),
        )
        kernel_hertz_weights(self.samples)

    def scatter(
        self, hertz: HertzState, state: StressState, info: StressInfo, surface: StressSurfaceInfo, epsilon: float
    ):
        kernel_hertz_samples(self.patches, self.samples, hertz, info, surface, epsilon)
        kernel_hertz_accept(self.patches, self.samples, hertz, epsilon)
        kernel_hertz_scatter(self.patches, self.samples, state, info)


@qd.func
def func_projected_surface_walk(
    anchor: int,
    center: qd.types.vector(3),
    offset: qd.types.vector(3),
    normal: qd.types.vector(3),
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    epsilon: float,
):
    """Lift a straight tangent-plane segment through adjacent facing triangles.

    Edge/vertex crossings may have zero parameter advance. Their shared vertex
    identities and facing orientation determine the next triangle; no Euclidean
    nearest-face search can jump to an unrelated sheet or across a cavity.
    """
    current, result = anchor, -1
    parameter = gs.qd_float(0.0)
    point, barycentric = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
    active = anchor >= 0 and anchor < stress_info.surface_nodes.shape[0]
    for _ in range(stress_info.surface_nodes.shape[0]):
        if active:
            origin = surface_info.face_origin[current]
            face_normal = surface_info.face_normal[current]
            denominator = normal.dot(face_normal)
            if denominator <= 64 * epsilon:
                active = False
            else:
                start = center + normal * face_normal.dot(origin - center) / denominator
                direction = offset - normal * face_normal.dot(offset) / denominator
                dual = surface_info.face_dual[current]
                local = dual @ (start - origin)
                change = dual @ direction
                bary = qd.Vector([1 - local.sum(), local[0], local[1]])
                derivative = qd.Vector([-change.sum(), change[0], change[1]])
                tolerance = 64 * epsilon * (1 + (center.norm() + origin.norm() + offset.norm()) * dual.norm())
                contained, reached = True, True
                for a in qd.static(range(3)):
                    contained = contained and bary[a] + parameter * derivative[a] >= -tolerance
                    reached = reached and bary[a] + derivative[a] >= -tolerance
                if not contained:
                    active = False
                elif reached:
                    point, barycentric = start + direction, bary + derivative
                    result, active = current, False
                else:
                    crossing, edge = gs.qd_float(float("inf")), -1
                    for a in qd.static(range(3)):
                        if derivative[a] < -tolerance:
                            # Barycentric roundoff is not a segment-parameter
                            # tolerance: short footprints have tiny derivatives.
                            # Containment already bounds the current barycentric
                            # error, so clamp its zero-distance crossing here.
                            current_bary = bary[a] + parameter * derivative[a]
                            candidate = qd.max(parameter, parameter - current_bary / derivative[a])
                            if candidate < crossing:
                                crossing, edge = candidate, (a + 1) % 3
                    if edge < 0 or crossing > 1 + tolerance:
                        active = False
                    else:
                        neighbor = stress_info.surface_neighbors[current, edge]
                        if neighbor < 0:
                            active = False
                        else:
                            current, parameter = neighbor, qd.max(parameter, crossing)
    return result, barycentric, point


@qd.kernel
def kernel_hertz_weights(sample_state: HertzSampleState):
    for i in range(sample_state.weight.shape[0]):
        radial = i // (sample_state.weight.shape[0] // 3)
        t, weight = 0.5, 4 / 9
        if radial != 1:
            t = 0.5 + (radial - 1) * qd.sqrt(3 / 5) / 2
            weight = 5 / 18
        # r=sqrt(1-t²): normalized Hertz measure becomes 3*t² dt dtheta/(2*pi).
        sample_state.weight[i] = 3 * t * t * weight / (sample_state.weight.shape[0] // 3)


@qd.kernel
def kernel_hertz_samples(
    patch_state: HertzPatchState,
    sample_state: HertzSampleState,
    hertz_state: HertzState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    epsilon: float,
):
    for i_s, i_p in qd.ndrange(sample_state.face.shape[0], sample_state.face.shape[1]):
        sample_state.face[i_s, i_p] = -1
        sample_state.position[i_s, i_p] = qd.Vector.zero(gs.qd_float, 3)
        sample_state.barycentric[i_s, i_p] = qd.Vector.zero(gs.qd_float, 3)
        if hertz_state.status[i_p] == 0:
            angles = sample_state.face.shape[0] // 3
            radial = i_s // angles
            t = 0.5
            if radial != 1:
                t = 0.5 + (radial - 1) * qd.sqrt(3 / 5) / 2
            radius = qd.sqrt(1 - t * t)
            theta = 2 * pi * ((i_s % angles) + 0.5) / angles
            normal, tangent = patch_state.normal[i_p], patch_state.tangent[i_p]
            second = normal.cross(tangent)
            orientation = hertz_state.direction[i_p]
            major = orientation[0] * tangent + orientation[1] * second
            minor = -orientation[1] * tangent + orientation[0] * second
            axes = hertz_state.axes[i_p]
            offset = radius * (axes[0] * qd.cos(theta) * major + axes[1] * qd.sin(theta) * minor)
            anchor = patch_state.anchor[i_p]
            face, bary, position = func_projected_surface_walk(
                anchor, patch_state.center[i_p], offset, normal, stress_info, surface_info, epsilon
            )
            sample_state.face[i_s, i_p] = face
            sample_state.barycentric[i_s, i_p] = bary
            sample_state.position[i_s, i_p] = position


@qd.kernel
def kernel_hertz_accept(
    patch_state: HertzPatchState, sample_state: HertzSampleState, hertz_state: HertzState, epsilon: float
):
    for i_b in range(sample_state.admitted.shape[0]):
        sample_state.admitted[i_b] = True
    for i_p in range(patch_state.status.shape[0]):
        status = 2
        force_error, moment_error = 0.0, 0.0
        i_b = patch_state.environment[i_p]
        if hertz_state.status[i_p] == 1:
            status = 5
            if patch_state.force[i_p].norm() <= 0 and patch_state.moment[i_p].norm() <= 0:
                status = 1
        elif hertz_state.status[i_p] == 0:
            center, normal, tangent = patch_state.center[i_p], patch_state.normal[i_p], patch_state.tangent[i_p]
            force, target_moment = patch_state.force[i_p], patch_state.moment[i_p]
            finite = (
                center.norm() < float("inf") and force.norm() < float("inf") and target_moment.norm() < float("inf")
            )
            frame_ok = qd.abs(normal.dot(normal) - 1) < 256 * epsilon
            frame_ok = frame_ok and qd.abs(tangent.dot(tangent) - 1) < 256 * epsilon
            frame_ok = frame_ok and qd.abs(normal.dot(tangent)) < 256 * epsilon
            normal_error = qd.abs(-force.dot(normal) - hertz_state.force[i_p])
            if not finite or not frame_ok or normal_error > 1e-8 * force.norm():
                status = 5
            else:
                status, total = 0, gs.qd_float(0.0)
                moment = qd.Vector.zero(gs.qd_float, 3)
                for i_s in range(sample_state.face.shape[0]):
                    if sample_state.face[i_s, i_p] < 0:
                        status = 3
                    total += sample_state.weight[i_s]
                    moment += sample_state.weight[i_s] * (sample_state.position[i_s, i_p] - center).cross(force)
                force_error = qd.abs(total - 1) * force.norm()
                moment_error = (moment - target_moment).norm()
                axes = hertz_state.axes[i_p]
                lever = axes[0]
                roundoff = 256 * epsilon * (center.norm() + lever) * force.norm()
                if status == 0 and (
                    force_error > 1e-8 * force.norm() or moment_error > 1e-8 * force.norm() * lever + roundoff
                ):
                    status = 4
        if i_b < 0 or i_b >= sample_state.admitted.shape[0]:
            status = 5
        elif status > 1:
            sample_state.admitted[i_b] = False
        patch_state.status[i_p] = status
        patch_state.force_error[i_p], patch_state.moment_error[i_p] = force_error, moment_error


@qd.kernel
def kernel_hertz_scatter(
    patch_state: HertzPatchState, sample_state: HertzSampleState, stress_state: StressState, stress_info: StressInfo
):
    for i_n, i_b in qd.ndrange(stress_state.force.shape[0], stress_state.force.shape[1]):
        stress_state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_s, i_p in qd.ndrange(sample_state.face.shape[0], sample_state.face.shape[1]):
        i_b = patch_state.environment[i_p]
        if patch_state.status[i_p] == 0 and sample_state.admitted[i_b]:
            i_f = sample_state.face[i_s, i_p]
            bary = sample_state.barycentric[i_s, i_p]
            shape = qd.Vector(
                [
                    bary[0] * (2 * bary[0] - 1),
                    bary[1] * (2 * bary[1] - 1),
                    bary[2] * (2 * bary[2] - 1),
                    4 * bary[0] * bary[1],
                    4 * bary[0] * bary[2],
                    4 * bary[1] * bary[2],
                ]
            )
            force = sample_state.weight[i_s] * patch_state.force[i_p]
            for a in qd.static(range(6)):
                node = stress_info.surface_nodes[i_f, a]
                stress_state.force[node, i_b] += shape[a] * force
