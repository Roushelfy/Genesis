"""Physical sharp-edge regions and coupled load/support updates from rigid contacts."""

from dataclasses import dataclass
from math import pi
from typing import ClassVar

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.utils import geom
from genesis.utils.array_class import DataKind, V, V_MAT, V_VEC, iter_data
from genesis.utils.misc import qd_to_numpy

from .contact import StressContactState
from .data import StressInfo
from .edge import StressEdge, StressEdgeState
from .edge_geometry import func_box_edge_minimum
from .force_allocation import StressForceAllocation, StressForceAllocationState
from .hertz import StressHertz, StressHertzState
from .hertz_recovery import StressHertzFrames
from .hertz_regions import StressHertzRegionState
from .provenance import StressSourceState, func_same_surface_feature
from .surface import StressSurfaceInfo, func_surface_geometry, func_surface_height_derivatives
from .surface_walk import func_projected_surface_walk


@dataclass(frozen=True)
class StressHardBindingState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    is_active: qd.Tensor
    status: qd.Tensor
    profile: qd.Tensor
    geometry_status: qd.Tensor
    minimum: qd.Tensor
    basis: qd.Tensor
    anchor: qd.Tensor
    leader: qd.Tensor
    source_count: qd.Tensor
    force: qd.Tensor
    original_moment: qd.Tensor
    force_error: qd.Tensor
    moment_error: qd.Tensor
    cone_margin: qd.Tensor


class StressHardBinding:
    """Prepare physical pressure regions in environments containing sharp box edges.

    Profile1 is Hertz and2 is a slender box edge. Status0 accepts the numerical
    coupled input,1 is inactive,2 is invalid provenance,3 is unresolved geometry,
    4 is incompatible grouping,5 is an unsupported pressure law,6 is failed
    force allocation,7 is failed wrench/cone feedback and8 exceeds the5% region
    force-change budget. Complete stress sensitivity and physical validation
    remain separate from this input certificate.
    """

    def __init__(self, contacts: StressContactState, info: StressInfo):
        shape = contacts.valid.shape
        n_patches = shape[0] * shape[1]
        self.state = StressHardBindingState(
            is_active=V(dtype=gs.qd_int, shape=(shape[1],)),
            status=V(dtype=gs.qd_int, shape=(shape[1],)),
            profile=V(dtype=gs.qd_int, shape=shape),
            geometry_status=V(dtype=gs.qd_int, shape=shape),
            minimum=V_VEC(3, dtype=gs.qd_float, shape=shape),
            basis=V_MAT(3, 3, dtype=gs.qd_float, shape=shape),
            anchor=V(dtype=gs.qd_int, shape=shape),
            leader=V(dtype=gs.qd_int, shape=shape),
            source_count=V(dtype=gs.qd_int, shape=shape),
            force=V_VEC(3, dtype=gs.qd_float, shape=shape),
            original_moment=V_VEC(3, dtype=gs.qd_float, shape=shape),
            force_error=V(dtype=gs.qd_float, shape=(shape[1],)),
            moment_error=V(dtype=gs.qd_float, shape=(shape[1],)),
            cone_margin=V(dtype=gs.qd_float, shape=(shape[1],)),
        )
        self.frames = StressHertzFrames(
            center=V_VEC(3, dtype=gs.qd_float, shape=(n_patches,)),
            basis=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_patches,)),
            anchor=V(dtype=gs.qd_int, shape=(n_patches,)),
        )
        self.hertz, self.edge = StressHertz(n_patches), StressEdge(n_patches)
        self.allocation = StressForceAllocation(*shape)
        vertices = qd_to_numpy(info.vertices, transpose=True)
        self.allocation.state.length.fill(float(np.linalg.norm(np.ptp(vertices, axis=0))))

    def update(self, contacts, sources, regions, info, surface, modulus, poisson, epsilon):
        kernel_hard_begin(contacts, regions, self.state, self.allocation.state)
        kernel_hard_geometry(
            contacts, sources, regions, self.state, self.hertz.state, self.edge.state, info, surface, epsilon
        )
        kernel_hard_group(contacts, sources, regions, self.state, self.allocation.state, info, epsilon)
        for iteration in range(4):
            kernel_hard_law_inputs(modulus, poisson, self.state, self.hertz.state, self.edge.state)
            self.hertz.update()
            self.edge.update()
            kernel_hard_frames(
                contacts,
                regions,
                self.state,
                self.hertz.state,
                self.edge.state,
                self.frames,
                self.allocation.state,
                info,
                surface,
                epsilon,
            )
            kernel_hard_support(contacts, regions, self.state, self.hertz.state, self.edge.state, self.frames, epsilon)
            kernel_hard_target(self.state, self.allocation.state)
            if iteration < 3:
                self.allocation.update()
                kernel_hard_force_commit(self.state, self.allocation.state)
        kernel_hard_accept(contacts, self.state, self.hertz.state, self.edge.state, self.frames, self.allocation.state)

    def data(self, prefix: str):
        yield from iter_data(self.state, prefix + ".state")
        yield from iter_data(self.frames, prefix + ".frames")
        yield from iter_data(self.hertz.state, prefix + ".hertz")
        yield from iter_data(self.hertz.info, prefix + ".hertz.info")
        yield from iter_data(self.edge.state, prefix + ".edge")
        yield from iter_data(self.allocation.state, prefix + ".allocation")

    def reset(self, envs_idx: torch.Tensor):
        kernel_hard_reset(envs_idx, self.state, self.hertz.state, self.edge.state, self.frames, self.allocation.state)


@qd.kernel
def kernel_hard_reset(
    envs_idx: qd.types.ndarray(),
    state: StressHardBindingState,
    hertz: StressHertzState,
    edge: StressEdgeState,
    frames: StressHertzFrames,
    allocation: StressForceAllocationState,
):
    for i_selected in range(envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        state.is_active[i_b], state.status[i_b] = 0, 1
        state.force_error[i_b], state.moment_error[i_b], state.cone_margin[i_b] = 0, 0, 0
        allocation.status[i_b], allocation.residual[i_b], allocation.maximum_change[i_b], allocation.iterations[i_b] = (
            1,
            0,
            0,
            0,
        )
        allocation.target[i_b] = qd.Vector.zero(gs.qd_float, 6)
    for i_c, i_selected in qd.ndrange(state.profile.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        i_p = i_c * state.profile.shape[1] + i_b
        state.profile[i_c, i_b], state.geometry_status[i_c, i_b], state.source_count[i_c, i_b] = 0, 1, 0
        state.leader[i_c, i_b], state.anchor[i_c, i_b] = -1, -1
        state.force[i_c, i_b], state.original_moment[i_c, i_b] = (
            qd.Vector.zero(gs.qd_float, 3),
            qd.Vector.zero(gs.qd_float, 3),
        )
        state.minimum[i_c, i_b], state.basis[i_c, i_b] = (
            qd.Vector.zero(gs.qd_float, 3),
            qd.Matrix.identity(gs.qd_float, 3),
        )
        allocation.original[i_c, i_b], allocation.force[i_c, i_b] = (
            qd.Vector.zero(gs.qd_float, 3),
            qd.Vector.zero(gs.qd_float, 3),
        )
        hertz.force[i_p], hertz.status[i_p], hertz.peak_pressure[i_p] = 0, 1, 0
        hertz.axes[i_p] = qd.Vector.zero(gs.qd_float, 2)
        edge.force[i_p], edge.status[i_p], edge.amplitude[i_p], edge.width[i_p], edge.half_length[i_p] = 0, 1, 0, 0, 0
        frames.anchor[i_p], frames.center[i_p] = -1, qd.Vector([float("nan")] * 3)
        frames.basis[i_p] = qd.Matrix.identity(gs.qd_float, 3)


@qd.kernel
def kernel_hard_begin(
    contacts: StressContactState,
    regions: StressHertzRegionState,
    state: StressHardBindingState,
    allocation: StressForceAllocationState,
):
    for i_b in range(state.status.shape[0]):
        state.is_active[i_b], state.status[i_b] = 0, 1
        state.force_error[i_b], state.moment_error[i_b], state.cone_margin[i_b] = 0, 0, 0
        allocation.target[i_b] = qd.Vector.zero(gs.qd_float, 6)
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        state.profile[i_c, i_b], state.geometry_status[i_c, i_b] = 0, 1
        state.leader[i_c, i_b], state.source_count[i_c, i_b], state.anchor[i_c, i_b] = -1, 0, -1
        state.force[i_c, i_b], state.original_moment[i_c, i_b] = (
            qd.Vector.zero(gs.qd_float, 3),
            qd.Vector.zero(gs.qd_float, 3),
        )
        state.minimum[i_c, i_b], state.basis[i_c, i_b] = (
            qd.Vector.zero(gs.qd_float, 3),
            qd.Matrix.identity(gs.qd_float, 3),
        )
        allocation.original[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0 and regions.n_boundaries[i_c, i_b] == 4:
            planes = regions.boundary_planes[i_c, i_b]
            for i in range(4):
                if planes[i, 2] < 0:
                    qd.atomic_or(state.is_active[i_b], 1)


@qd.kernel
def kernel_hard_geometry(
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressHertzRegionState,
    state: StressHardBindingState,
    hertz: StressHertzState,
    edge: StressEdgeState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        i_p = i_c * contacts.valid.shape[1] + i_b
        hertz.curvature[i_p] = qd.Matrix.zero(gs.qd_float, 2, 2)
        edge.slope[i_p], edge.curvature[i_p] = 1.0, 1.0
        if state.is_active[i_b] and contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0:
            state.geometry_status[i_c, i_b] = 2
            if sources.status[i_c, i_b] == 0 and contacts.anchor_face[i_c, i_b] >= 0:
                state.geometry_status[i_c, i_b] = 3
                rotation = contacts.frame_quaternion[i_b]
                root = geom.qd_inv_transform_by_quat(regions.center[i_c, i_b] - contacts.frame_position[i_b], rotation)
                normal = geom.qd_inv_transform_by_quat(regions.normal[i_c, i_b], rotation)
                tangent = geom.qd_inv_transform_by_quat(regions.tangent[i_c, i_b], rotation)
                is_edge = False
                if regions.n_boundaries[i_c, i_b] == 4:
                    for i in range(4):
                        is_edge = is_edge or regions.boundary_planes[i_c, i_b][i, 2] < 0
                if is_edge:
                    anchor, minimum, bary, basis, slope, curvature, residual, code = func_box_edge_minimum(
                        contacts.anchor_face[i_c, i_b],
                        contacts.center[i_c, i_b],
                        root,
                        normal,
                        tangent,
                        regions.boundary_planes[i_c, i_b],
                        info,
                        surface,
                        epsilon,
                    )
                    if code == 0:
                        state.profile[i_c, i_b], state.geometry_status[i_c, i_b] = 2, 0
                        state.anchor[i_c, i_b], state.minimum[i_c, i_b], state.basis[i_c, i_b] = anchor, minimum, basis
                        edge.slope[i_p], edge.curvature[i_p] = slope, curvature
                elif regions.status[i_c, i_b] == 0 and (
                    regions.opponent_type[i_c, i_b] == gs.GEOM_TYPE.PLANE
                    or regions.opponent_type[i_c, i_b] == gs.GEOM_TYPE.BOX
                ):
                    tangent = (tangent - normal * tangent.dot(normal)).normalized()
                    basis = qd.Matrix.cols([tangent, -normal.cross(tangent), -normal])
                    anchor, point = contacts.anchor_face[i_c, i_b], contacts.center[i_c, i_b]
                    offset = root - point
                    curvature = qd.Matrix.zero(gs.qd_float, 2, 2)
                    is_active, is_minimum = True, False
                    # The collision normal root seeds a minimum of the actual material surface.
                    for i_iteration in range(9):
                        if is_active:
                            anchor, bary, point = func_projected_surface_walk(
                                anchor, point, offset, normal, info, surface, epsilon
                            )
                            is_active = False
                            if anchor >= 0:
                                gradient, curvature, is_regular = func_surface_height_derivatives(
                                    anchor, bary, basis[:, :2].transpose(), normal, info, epsilon
                                )
                                if is_regular and curvature.trace() > 0 and curvature.determinant() > 0:
                                    is_minimum = gradient.norm() <= qd.max(1e-10, 128 * epsilon * (1 + gradient.norm()))
                                    if not is_minimum and i_iteration < 8:
                                        offset = -basis[:, :2] @ (curvature.inverse() @ gradient)
                                        is_active = True
                    if is_minimum:
                        state.profile[i_c, i_b], state.geometry_status[i_c, i_b] = 1, 0
                        state.anchor[i_c, i_b], state.minimum[i_c, i_b], state.basis[i_c, i_b] = anchor, point, basis
                        hertz.curvature[i_p] = curvature


@qd.kernel
def kernel_hard_group(
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressHertzRegionState,
    state: StressHardBindingState,
    allocation: StressForceAllocationState,
    info: StressInfo,
    epsilon: float,
):
    for i_b in range(state.status.shape[0]):
        if state.is_active[i_b]:
            state.status[i_b] = 0
            for i_c in range(contacts.valid.shape[0]):
                if contacts.valid[i_c, i_b] and not contacts.force[i_c, i_b].norm() < float("inf"):
                    state.status[i_b] = 2
                if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0:
                    if state.geometry_status[i_c, i_b] != 0:
                        state.status[i_b] = qd.max(state.status[i_b], state.geometry_status[i_c, i_b])
                    else:
                        leader = i_c
                        curvature = regions.curvature[i_c, i_b]
                        resolution = 1e-6 * curvature.trace() / curvature.determinant()
                        resolution += 128 * epsilon * state.minimum[i_c, i_b].norm()
                        for j_c in range(i_c):
                            if (
                                state.geometry_status[j_c, i_b] == 0
                                and sources.geom[j_c, i_b] == sources.geom[i_c, i_b]
                            ):
                                if (
                                    sources.pair[j_c, i_b] == sources.pair[i_c, i_b]
                                    and state.profile[j_c, i_b] == state.profile[i_c, i_b]
                                ):
                                    if (state.minimum[j_c, i_b] - state.minimum[i_c, i_b]).norm() <= resolution:
                                        if func_same_surface_feature(
                                            state.anchor[j_c, i_b],
                                            state.anchor[i_c, i_b],
                                            state.minimum[i_c, i_b],
                                            resolution,
                                            info,
                                        ):
                                            leader = qd.min(leader, state.leader[j_c, i_b])
                        state.leader[i_c, i_b] = leader
                        if qd.abs(contacts.friction[i_c, i_b] - contacts.friction[leader, i_b]) > 128 * epsilon:
                            state.status[i_b] = 4
                        force = contacts.force[i_c, i_b]
                        state.force[leader, i_b] += force
                        state.original_moment[leader, i_b] += (
                            contacts.position[i_c, i_b].cross(force) + sources.free_moment[i_c, i_b]
                        )
                        state.source_count[leader, i_b] += 1
            for i_c in range(contacts.valid.shape[0]):
                if state.source_count[i_c, i_b] > 0:
                    allocation.original[i_c, i_b] = state.force[i_c, i_b]
                    allocation.friction[i_c, i_b] = contacts.friction[i_c, i_b]


@qd.kernel
def kernel_hard_law_inputs(
    modulus: float, poisson: float, state: StressHardBindingState, hertz: StressHertzState, edge: StressEdgeState
):
    for i_c, i_b in qd.ndrange(state.profile.shape[0], state.profile.shape[1]):
        i_p = i_c * state.profile.shape[1] + i_b
        hertz.force[i_p], edge.force[i_p] = 0, 0
        hertz.modulus[i_p], edge.modulus[i_p], edge.poisson[i_p] = modulus, modulus, poisson
        edge.shear_ratio[i_p] = 0
        if state.status[i_b] == 0 and state.source_count[i_c, i_b] > 0:
            force, basis = state.force[i_c, i_b], state.basis[i_c, i_b]
            if state.profile[i_c, i_b] == 1:
                hertz.force[i_p] = force.dot(basis[:, 2])
            elif state.profile[i_c, i_b] == 2:
                compression = -force.dot(basis[:, 2])
                edge.force[i_p] = compression
                if compression > 0:
                    edge.shear_ratio[i_p] = force.dot(basis[:, 0]) / compression


@qd.kernel
def kernel_hard_frames(
    contacts: StressContactState,
    regions: StressHertzRegionState,
    state: StressHardBindingState,
    hertz: StressHertzState,
    edge: StressEdgeState,
    frames: StressHertzFrames,
    allocation: StressForceAllocationState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(state.profile.shape[0], state.profile.shape[1]):
        i_p = i_c * state.profile.shape[1] + i_b
        frames.anchor[i_p] = -1
        if state.status[i_b] == 0 and state.source_count[i_c, i_b] > 0:
            profile, force = state.profile[i_c, i_b], state.force[i_c, i_b]
            minimum, basis, anchor = state.minimum[i_c, i_b], state.basis[i_c, i_b], state.anchor[i_c, i_b]
            center = minimum
            is_valid = hertz.status[i_p] == 0 if profile == 1 else edge.status[i_p] == 0
            if profile == 2 and is_valid:
                offset = basis[:, 0] * (2 * (1 - edge.exponent[i_p]) * edge.width[i_p] / 5)
                anchor, bary, center = func_projected_surface_walk(
                    anchor, minimum, offset, basis[:, 2], info, surface, epsilon
                )
                is_valid = anchor >= 0
                if is_valid:
                    point, tangent, area = func_surface_geometry(anchor, bary, info)
                    normal = area.normalized()
                    axis = (basis[:, 0] - normal * normal.dot(basis[:, 0])).normalized()
                    edge.width[i_p] /= axis.dot(basis[:, 0])
                    basis = qd.Matrix.cols([axis, axis.cross(normal), -normal])
                    compression = force.dot(basis[:, 2])
                    edge.amplitude[i_p] = (
                        3
                        * compression
                        * qd.sin(pi * edge.exponent[i_p])
                        / (4 * pi * edge.exponent[i_p] * edge.width[i_p] * edge.half_length[i_p])
                    )
                    hertz.force[i_p], hertz.status[i_p] = compression, 0
                    hertz.axes[i_p] = qd.Vector([edge.half_length[i_p], edge.width[i_p]])
                    hertz.direction[i_p] = qd.Vector([1.0, 0.0])
                    hertz.peak_pressure[i_p] = edge.amplitude[i_p]
                    is_valid = compression > 0
            if is_valid:
                frames.anchor[i_p], frames.center[i_p], frames.basis[i_p] = anchor, center, basis
                allocation.center[i_c, i_b], allocation.normal[i_c, i_b] = center, basis[:, 2]
            else:
                qd.atomic_max(state.status[i_b], 5)


@qd.kernel
def kernel_hard_support(
    contacts: StressContactState,
    regions: StressHertzRegionState,
    state: StressHardBindingState,
    hertz: StressHertzState,
    edge: StressEdgeState,
    frames: StressHertzFrames,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(state.profile.shape[0], state.profile.shape[1]):
        if state.status[i_b] == 0 and state.source_count[i_c, i_b] > 0 and regions.n_boundaries[i_c, i_b] == 4:
            i_p = i_c * state.profile.shape[1] + i_b
            rotation = contacts.frame_quaternion[i_b]
            root = geom.qd_inv_transform_by_quat(regions.center[i_c, i_b] - contacts.frame_position[i_b], rotation)
            normal = geom.qd_inv_transform_by_quat(regions.normal[i_c, i_b], rotation)
            tangent = geom.qd_inv_transform_by_quat(regions.tangent[i_c, i_b], rotation)
            basis = frames.basis[i_p]
            direction = hertz.direction[i_p]
            for i in range(4):
                coefficients = regions.boundary_planes[i_c, i_b][i, :]
                axis = tangent * coefficients[0] + normal.cross(tangent) * coefficients[1]
                x, y = axis.dot(basis[:, 0]), axis.dot(basis[:, 1])
                extent = 0.0
                if state.profile[i_c, i_b] == 2:
                    mean = 2 * (1 - edge.exponent[i_p]) * edge.width[i_p] / 5
                    extent = qd.max(x, 0) * (edge.width[i_p] - mean) + qd.max(-x, 0) * mean
                    extent += qd.abs(y) * edge.half_length[i_p]
                else:
                    major = x * direction[0] + y * direction[1]
                    minor = -x * direction[1] + y * direction[0]
                    extent = qd.sqrt((major * hertz.axes[i_p][0]) ** 2 + (minor * hertz.axes[i_p][1]) ** 2)
                tolerance = 256 * epsilon * (root.norm() + qd.abs(coefficients[2]) + hertz.axes[i_p][0])
                if axis.dot(frames.center[i_p] - root) + extent > coefficients[2] + tolerance:
                    qd.atomic_max(state.status[i_b], 5)


@qd.kernel
def kernel_hard_target(state: StressHardBindingState, allocation: StressForceAllocationState):
    for i_b in range(state.status.shape[0]):
        if state.status[i_b] == 0:
            origin, force, moment = (
                qd.Vector.zero(gs.qd_float, 3),
                qd.Vector.zero(gs.qd_float, 3),
                qd.Vector.zero(gs.qd_float, 3),
            )
            count = 0
            for i_c in range(state.profile.shape[0]):
                if state.source_count[i_c, i_b] > 0:
                    origin += allocation.center[i_c, i_b]
                    force += allocation.original[i_c, i_b]
                    moment += state.original_moment[i_c, i_b]
                    count += 1
            if count > 0:
                moment = (moment - (origin / count).cross(force)) / allocation.length[i_b]
                allocation.target[i_b] = qd.Vector([force[0], force[1], force[2], moment[0], moment[1], moment[2]])


@qd.kernel
def kernel_hard_force_commit(state: StressHardBindingState, allocation: StressForceAllocationState):
    for i_b in range(state.status.shape[0]):
        if state.status[i_b] == 0 and allocation.status[i_b] != 0:
            state.status[i_b] = 6
    for i_c, i_b in qd.ndrange(state.profile.shape[0], state.profile.shape[1]):
        if state.status[i_b] == 0 and state.source_count[i_c, i_b] > 0:
            state.force[i_c, i_b] = allocation.force[i_c, i_b]


@qd.kernel
def kernel_hard_accept(
    contacts: StressContactState,
    state: StressHardBindingState,
    hertz: StressHertzState,
    edge: StressEdgeState,
    frames: StressHertzFrames,
    allocation: StressForceAllocationState,
):
    for i_b in range(state.status.shape[0]):
        if state.status[i_b] == 0:
            force, moment, expected, expected_moment = [qd.Vector.zero(gs.qd_float, 3) for _ in range(4)]
            load_scale, moment_scale, margin = 0.0, 0.0, gs.qd_float(float("inf"))
            for i_c in range(state.profile.shape[0]):
                if state.source_count[i_c, i_b] > 0:
                    i_p = i_c * state.profile.shape[1] + i_b
                    current = state.force[i_c, i_b]
                    compression = current.dot(frames.basis[i_p][:, 2])
                    radial = (current - compression * frames.basis[i_p][:, 2]).norm()
                    margin = qd.min(margin, (contacts.friction[i_c, i_b] * compression - radial) / current.norm())
                    force += current
                    moment += frames.center[i_p].cross(current)
                    expected += allocation.original[i_c, i_b]
                    expected_moment += state.original_moment[i_c, i_b]
                    load_scale += allocation.original[i_c, i_b].norm()
                    moment_scale += compression * hertz.axes[i_p][0]
            state.force_error[i_b] = (force - expected).norm() / load_scale
            state.moment_error[i_b] = (moment - expected_moment).norm() / moment_scale
            state.cone_margin[i_b] = margin
            if state.force_error[i_b] > 1e-8 or state.moment_error[i_b] > 1e-8 or margin < -1e-10:
                state.status[i_b] = 7
            elif allocation.maximum_change[i_b] > 0.05:
                state.status[i_b] = 8
