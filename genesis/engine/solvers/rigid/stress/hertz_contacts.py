"""Connect current rigid contact regions to native Hertz recovery and radial friction memory."""

from dataclasses import dataclass
from typing import ClassVar, NamedTuple, TYPE_CHECKING

import torch

import quadrants as qd

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils import array_class, geom
from genesis.utils.array_class import DataItem, DataKind, V, V_MAT, V_VEC, iter_data, of_kind

from .contact import StressContactState
from .curvature import StressCurvature
from .data import StressInfo, StressState
from .edge import StressEdgeState
from .hard_binding import StressHardBinding, StressHardBindingState
from .hertz import StressHertz, StressHertzState
from .hertz_binding import (
    StressFrictionBindingState,
    StressFrictionRegions,
    kernel_friction_bind,
    kernel_friction_commit,
)
from .hertz_friction import StressFrictionHistory, StressFrictionInput, StressFrictionState, func_friction_reset
from .hertz_probes import StressHertzProbeState
from .hertz_recovery import StressHertzFrames, StressHertzRecovery, StressHertzRecoveryState
from .hertz_regions import StressHertzRegions, StressHertzRegionState
from .hertz_tangential import StressHertzShear
from .history import StressHistory
from .lifecycle import func_begin_environment
from .model import StressModel
from .provenance import StressSourceState, StressSourceStatus, create_sources, kernel_resolve_sources
from .source_vertices import StressSourceVertices, kernel_bind_source_vertices
from .surface import StressSurfaceInfo
from .surface_walk import func_projected_surface_walk

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


@dataclass(frozen=True)
class StressHertzContactState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    status: qd.Tensor
    force_error: qd.Tensor
    moment_error: qd.Tensor
    step_principal: qd.Tensor
    step_sampled_vm: qd.Tensor
    observed_step: qd.Tensor
    sampled_active: qd.Tensor
    input_is_valid: qd.Tensor = of_kind(DataKind.SCRATCH)


class StressHertzProbeField(NamedTuple):
    positions: torch.Tensor
    stress_tensor: torch.Tensor
    elements: torch.Tensor
    active: torch.Tensor


class HertzContacts:
    """Recover smooth Hertz and slender sharp-edge stresses from current rigid contacts.

    Smooth contacts use continuous radial vector friction memory. Sharp edges use
    grouped, wrench-compatible pressure estimates with proportional shear and
    finite-depth probes. Collision provenance and connected material surface
    walks bind each physical region. Failed geometry, load allocation or history
    correspondence invalidates the observation.
    """

    def __init__(self, link: "RigidLink", contacts: StressContactState, info: StressInfo, options: RigidStressOptions):
        n_regions, n_envs = contacts.valid.shape
        self.sources = create_sources(n_regions, n_envs)
        self.sources.uncertainty.fill(options.source_uncertainty)
        self.vertices = StressSourceVertices(link, info, options.source_geometry_tolerance)
        self.regions = StressHertzRegions(contacts)
        self.law = StressHertz(n_regions * n_envs)
        self.hard = StressHardBinding(contacts, info)
        self.friction_regions = StressFrictionRegions(
            force=V_VEC(3, dtype=gs.qd_float, shape=(n_regions, n_envs)),
            source_count=V(dtype=gs.qd_int, shape=(n_regions, n_envs)),
            leader=V(dtype=gs.qd_int, shape=(n_regions, n_envs)),
            profile=V(dtype=gs.qd_int, shape=(n_regions, n_envs)),
        )
        self.friction = StressFrictionHistory(n_regions * n_envs)
        self.shear = StressHertzShear(
            friction=V(dtype=gs.qd_float, shape=(n_regions * n_envs,)),
            force=V_VEC(2, dtype=gs.qd_float, shape=(n_regions * n_envs,)),
            coefficients=self.friction.state.coefficients,
        )
        self.binding = StressFrictionBindingState(
            geom=V(dtype=gs.qd_int, shape=(n_regions * n_envs,)),
            pair=V(dtype=gs.qd_int, shape=(n_regions * n_envs,)),
            center=V_VEC(3, dtype=gs.qd_float, shape=(n_regions * n_envs,)),
            basis=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_regions * n_envs,)),
            anchor=V(dtype=gs.qd_int, shape=(n_regions * n_envs,)),
            radius=V(dtype=gs.qd_float, shape=(n_regions * n_envs,)),
            is_lost=V(dtype=gs.qd_bool, shape=(n_envs,)),
        )
        self.recovery = StressHertzRecovery(
            n_envs,
            n_regions,
            output_mode=options.output_mode,
            moving_probes=True,
            shear=self.shear,
            edge=self.hard.edge.state,
        )
        self.state = StressHertzContactState(
            status=V(dtype=gs.qd_int, shape=(n_envs,)),
            force_error=V(dtype=gs.qd_float, shape=(n_regions, n_envs)),
            moment_error=V(dtype=gs.qd_float, shape=(n_regions, n_envs)),
            step_principal=V(dtype=gs.qd_float, shape=(n_envs,)),
            step_sampled_vm=V(dtype=gs.qd_float, shape=(n_envs,)),
            observed_step=V(dtype=gs.qd_int, shape=(n_envs,)),
            sampled_active=V(dtype=gs.qd_bool, shape=(n_regions, n_envs)),
            input_is_valid=V(dtype=gs.qd_bool, shape=(n_envs,)),
        )
        self.state.status.fill(0)
        self.state.step_principal.fill(float("nan"))
        self.reset(torch.arange(n_envs, device=gs.device, dtype=gs.tc_int), contacts)

    def recover(
        self,
        link: "RigidLink",
        solver: "RigidSolver",
        contacts: StressContactState,
        stress: StressState,
        omega: qd.Tensor,
        model: StressModel,
        surface: StressSurfaceInfo,
        curvature: StressCurvature,
        options: RigidStressOptions,
        history: StressHistory | None,
        epsilon: float,
        is_observed: bool,
    ) -> None:
        self.regions.update(
            link.idx,
            contacts,
            self.sources,
            solver.dyn_state,
            solver.collider.collider_state,
            solver.dyn_info,
            curvature.info,
            epsilon,
        )
        if self.vertices.info is not None:
            kernel_bind_source_vertices(contacts, self.sources, self.vertices.info, epsilon)
        kernel_resolve_sources(options.source_projection_limit, epsilon, contacts, self.sources, model.info, surface)
        modulus = options.hertz_reduced_modulus
        if modulus is None:
            modulus = options.young / (1 - options.poisson**2)
        self.hard.update(
            contacts, self.sources, self.regions.state, model.info, surface, modulus, options.poisson, epsilon
        )
        kernel_hertz_contact_inputs(
            contacts,
            self.sources,
            self.regions.state,
            self.law.state,
            self.recovery.frames,
            self.state,
            model.info,
            surface,
            modulus,
            options.source_projection_limit,
            epsilon,
        )
        self.law.update()
        kernel_hertz_contact_accept(
            contacts,
            self.regions.state,
            self.hard.state,
            self.hard.allocation.state.original,
            self.hard.hertz.state,
            self.hard.frames,
            self.law.state,
            self.recovery.frames,
            self.friction_regions,
            self.state,
            solver._errno,
        )
        kernel_friction_bind(
            contacts,
            self.sources,
            self.friction_regions,
            self.law.state,
            self.recovery.frames,
            self.binding,
            self.friction.input,
            self.shear,
            self.state.status,
            self.state.input_is_valid,
            model.info,
            surface,
            epsilon,
        )
        self.friction.update()
        kernel_friction_commit(
            contacts,
            self.sources,
            self.friction_regions,
            self.law.state,
            self.recovery.frames,
            self.binding,
            self.friction.input,
            self.friction.state,
            self.state.status,
            self.state.input_is_valid,
            solver._errno,
        )
        if not is_observed:
            return
        kernel_hertz_contact_begin(solver.sim.cur_step_global, self.state, stress)
        self.recovery.scatter(self.law.state, stress, model.info, surface, options.poisson, epsilon, solver._errno)
        model.recover(omega, stress, options, history, surface_load=True, compute_peak=False)
        self.recovery.evaluate(self.law.state, stress, model, epsilon, solver._errno)
        kernel_hertz_contact_peak(
            self.recovery.state.tensile_peak, self.law.state, self.hard.edge.state, stress, self.state
        )

    def reset(self, envs_idx: torch.Tensor, contacts: StressContactState, is_restore: bool = False) -> None:
        self.hard.reset(envs_idx)
        kernel_hertz_contact_reset(
            envs_idx,
            contacts,
            self.sources,
            self.regions.state,
            self.law.state,
            self.recovery.frames,
            self.state,
            self.recovery.state,
            self.recovery.probes,
            self.friction_regions,
            self.binding,
            self.friction.input,
            self.friction.state,
            self.shear,
            self.recovery.full,
            is_restore,
        )

    def data(self, prefix: str):
        yield from iter_data(self.sources, prefix + ".sources")
        if self.vertices.info is not None:
            yield from iter_data(self.vertices.info, prefix + ".vertices")
        yield from iter_data(self.regions.state, prefix + ".regions")
        yield from iter_data(self.law.info, prefix + ".law.info")
        yield from iter_data(self.law.state, prefix + ".law.state")
        yield from self.hard.data(prefix + ".hard")
        yield from iter_data(self.state, prefix + ".state")
        yield from iter_data(self.friction_regions, prefix + ".friction_regions")
        yield from iter_data(self.binding, prefix + ".binding")
        yield from self.friction.data(prefix + ".friction")
        yield from iter_data(self.shear, prefix + ".shear")
        yield from iter_data(self.recovery.frames, prefix + ".frames")
        yield from iter_data(self.recovery.probes, prefix + ".probes")
        yield from iter_data(self.recovery.state, prefix + ".recovery")
        yield DataItem(prefix + ".probe_offsets", self.recovery.probe_offsets, DataKind.CONSTANT)
        yield DataItem(prefix + ".quadrature", self.recovery.quadrature, DataKind.CONSTANT)
        yield DataItem(prefix + ".boundary_quadrature", self.recovery.boundary_quadrature, DataKind.CONSTANT)
        yield DataItem(prefix + ".edge_quadrature", self.recovery.edge_quadrature, DataKind.CONSTANT)


@qd.kernel
def kernel_hertz_contact_inputs(
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressHertzRegionState,
    law: StressHertzState,
    frames: StressHertzFrames,
    state: StressHertzContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    modulus: float,
    projection_limit: float,
    epsilon: float,
):
    for i_b in range(state.status.shape[0]):
        state.status[i_b] = 0 if regions.is_valid[i_b] else 1
    for i_c, i_b in qd.ndrange(regions.status.shape[0], regions.status.shape[1]):
        i_p = i_c * regions.status.shape[1] + i_b
        law.force[i_p], law.modulus[i_p] = 0.0, modulus
        law.curvature[i_p] = qd.Matrix.zero(gs.qd_float, 2, 2)
        frames.anchor[i_p] = -1
        frames.center[i_p] = qd.Vector.zero(gs.qd_float, 3)
        frames.basis[i_p] = qd.Matrix.identity(gs.qd_float, 3)
        contacts.status[i_c, i_b] = 0
        state.force_error[i_c, i_b], state.moment_error[i_c, i_b] = 0.0, 0.0
        if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0:
            if sources.status[i_c, i_b] != StressSourceStatus.SUCCESS:
                contacts.status[i_c, i_b] = 10 + sources.status[i_c, i_b]
                qd.atomic_or(state.status[i_b], 2)
            leader = regions.leader[i_c, i_b]
            if leader >= 0:
                if (
                    sources.geom[i_c, i_b] != sources.geom[leader, i_b]
                    or sources.pair[i_c, i_b] != sources.pair[leader, i_b]
                ):
                    contacts.status[i_c, i_b] = 54
                    qd.atomic_or(state.status[i_b], 8)
        if regions.is_valid[i_b] and regions.source_count[i_c, i_b] > 0:
            rotation, origin = contacts.frame_quaternion[i_b], contacts.frame_position[i_b]
            center = geom.qd_inv_transform_by_quat(regions.center[i_c, i_b] - origin, rotation)
            outward = geom.qd_inv_transform_by_quat(regions.normal[i_c, i_b], rotation)
            tangent = geom.qd_inv_transform_by_quat(regions.tangent[i_c, i_b], rotation)
            frames.basis[i_p] = qd.Matrix.cols([tangent, -outward.cross(tangent), -outward])
            frames.center[i_p] = center
            force = regions.force[i_c, i_b]
            compression = -force.dot(regions.normal[i_c, i_b])
            residual = force + compression * regions.normal[i_c, i_b]
            state.force_error[i_c, i_b] = (force - (compression * (-regions.normal[i_c, i_b]) + residual)).norm()
            curvature = regions.curvature[i_c, i_b]
            law.curvature[i_p] = qd.Matrix([[curvature[0, 0], -curvature[0, 1]], [-curvature[1, 0], curvature[1, 1]]])
            if sources.status[i_c, i_b] == StressSourceStatus.SUCCESS:
                material_force = geom.qd_inv_transform_by_quat(force, rotation)
                material_moment = geom.qd_inv_transform_by_quat(regions.moment[i_c, i_b], rotation)
                normal = -outward
                projection_normal = outward
                offset = qd.Vector.zero(gs.qd_float, 3)
                if compression > 0:
                    offset = material_force.cross(material_moment) / material_force.norm_sqr()
                    height = (contacts.center[i_c, i_b] - center).dot(normal)
                    offset += material_force * (height - offset.dot(normal)) / compression
                    if info.volume_points.shape[0] > 0:
                        # Translation along the resultant line preserves its moment on a curved boundary.
                        projection_normal = -material_force.normalized()
                anchor, bary, projected = func_projected_surface_walk(
                    contacts.anchor_face[i_c, i_b],
                    contacts.center[i_c, i_b],
                    center + offset - contacts.center[i_c, i_b],
                    projection_normal,
                    info,
                    surface,
                    epsilon,
                )
                if anchor >= 0 and (projected - center - offset).norm() <= projection_limit:
                    frames.anchor[i_p], frames.center[i_p] = anchor, projected
            if compression > 0:
                law.force[i_p] = compression


@qd.kernel
def kernel_hertz_contact_accept(
    contacts: StressContactState,
    regions: StressHertzRegionState,
    hard: StressHardBindingState,
    original: qd.Tensor,
    hard_law: StressHertzState,
    hard_frames: StressHertzFrames,
    law: StressHertzState,
    frames: StressHertzFrames,
    friction_regions: StressFrictionRegions,
    state: StressHertzContactState,
    errno: qd.Tensor,
):
    for i_b in range(state.status.shape[0]):
        if hard.is_active[i_b]:
            state.status[i_b] = 0 if hard.status[i_b] == 0 else 32
    for i_c, i_b in qd.ndrange(regions.status.shape[0], regions.status.shape[1]):
        i_p = i_c * regions.status.shape[1] + i_b
        friction_regions.force[i_c, i_b] = geom.qd_inv_transform_by_quat(
            regions.force[i_c, i_b], contacts.frame_quaternion[i_b]
        )
        friction_regions.source_count[i_c, i_b] = regions.source_count[i_c, i_b]
        friction_regions.leader[i_c, i_b] = regions.leader[i_c, i_b]
        friction_regions.profile[i_c, i_b] = 1 if regions.source_count[i_c, i_b] > 0 else 0
        if hard.is_active[i_b]:
            friction_regions.source_count[i_c, i_b], friction_regions.profile[i_c, i_b] = 0, 0
            friction_regions.leader[i_c, i_b] = hard.leader[i_c, i_b]
            friction_regions.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            law.force[i_p], law.status[i_p], law.peak_pressure[i_p], law.indentation[i_p] = 0, 1, 0, 0
            law.axes[i_p], law.direction[i_p] = qd.Vector.zero(gs.qd_float, 2), qd.Vector([1.0, 0.0])
            frames.anchor[i_p], frames.center[i_p] = -1, qd.Vector([float("nan")] * 3)
            state.force_error[i_c, i_b], state.moment_error[i_c, i_b] = 0, 0
            contacts.status[i_c, i_b] = 0
            if hard.status[i_b] == 0:
                friction_regions.source_count[i_c, i_b] = hard.source_count[i_c, i_b]
                if hard.source_count[i_c, i_b] > 0:
                    force = hard.force[i_c, i_b]
                    friction_regions.force[i_c, i_b], friction_regions.profile[i_c, i_b] = force, hard.profile[i_c, i_b]
                    law.force[i_p], law.status[i_p] = hard_law.force[i_p], hard_law.status[i_p]
                    law.axes[i_p], law.direction[i_p] = hard_law.axes[i_p], hard_law.direction[i_p]
                    law.peak_pressure[i_p], law.indentation[i_p] = (
                        hard_law.peak_pressure[i_p],
                        hard_law.indentation[i_p],
                    )
                    law.curvature[i_p] = hard_law.curvature[i_p]
                    frames.anchor[i_p], frames.center[i_p], frames.basis[i_p] = (
                        hard_frames.anchor[i_p],
                        hard_frames.center[i_p],
                        hard_frames.basis[i_p],
                    )
                    state.force_error[i_c, i_b] = (force - original[i_c, i_b]).norm()
                    state.moment_error[i_c, i_b] = (
                        frames.center[i_p].cross(force) - hard.original_moment[i_c, i_b]
                    ).norm()
            elif contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0:
                contacts.status[i_c, i_b] = 60 + hard.status[i_b]
        elif regions.source_count[i_c, i_b] > 0:
            force = regions.force[i_c, i_b]
            projected = geom.qd_transform_by_trans_quat(
                frames.center[i_p], contacts.frame_position[i_b], contacts.frame_quaternion[i_b]
            )
            moment = regions.moment[i_c, i_b] + (regions.center[i_c, i_b] - projected).cross(force)
            state.moment_error[i_c, i_b] = moment.norm()
            status = 0
            if law.status[i_p] != 0 or frames.anchor[i_p] < 0:
                status = 51
            elif state.force_error[i_c, i_b] > 1e-8 * force.norm():
                status = 52
            elif moment.norm() > 1e-8 * law.force[i_p] * law.axes[i_p][0]:
                status = 53
            if status:
                contacts.status[i_c, i_b] = status
                qd.atomic_or(state.status[i_b], 4)
    for i_b in range(state.status.shape[0]):
        if state.status[i_b] != 0:
            state.input_is_valid[i_b] = False
            qd.atomic_or(errno[i_b], array_class.ErrorCode.INVALID_STRESS_LOAD)


@qd.kernel
def kernel_hertz_contact_input_begin(state: StressHertzContactState):
    for i_b in range(state.status.shape[0]):
        state.input_is_valid[i_b] = True


@qd.kernel
def kernel_hertz_contact_begin(i_step: int, state: StressHertzContactState, stress: StressState):
    for i_b in range(state.status.shape[0]):
        if state.observed_step[i_b] != i_step:
            func_begin_environment(i_b, stress)
            state.step_principal[i_b], state.observed_step[i_b] = -float("inf"), i_step
            state.step_sampled_vm[i_b] = 0.0
        stress.step_valid[i_b] = stress.step_valid[i_b] and state.input_is_valid[i_b]


@qd.kernel
def kernel_hertz_contact_peak(
    tensile: qd.Tensor,
    law: StressHertzState,
    edge: StressEdgeState,
    stress: StressState,
    state: StressHertzContactState,
):
    for i_b in range(state.status.shape[0]):
        if stress.step_valid[i_b]:
            state.step_principal[i_b] = qd.max(state.step_principal[i_b], tensile[i_b])
            state.step_sampled_vm[i_b] = qd.max(state.step_sampled_vm[i_b], stress.peak[i_b])
        else:
            state.step_principal[i_b] = float("nan")
            state.step_sampled_vm[i_b] = float("nan")
    for i_c, i_b in qd.ndrange(state.sampled_active.shape[0], state.sampled_active.shape[1]):
        i_p = i_c * state.sampled_active.shape[1] + i_b
        state.sampled_active[i_c, i_b] = law.force[i_p] > 0 and stress.step_valid[i_b]
        if state.sampled_active[i_c, i_b] and edge.status[i_p] == 0:
            # The positive-depth probe bank cannot bound an ideal edge singularity.
            qd.atomic_max(stress.step_peak[i_b], float("inf"))


@qd.kernel
def kernel_hertz_contact_reset(
    envs_idx: qd.types.ndarray(),
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressHertzRegionState,
    law: StressHertzState,
    frames: StressHertzFrames,
    state: StressHertzContactState,
    recovery: StressHertzRecoveryState,
    probes: StressHertzProbeState,
    friction_regions: StressFrictionRegions,
    binding: StressFrictionBindingState,
    friction_input: StressFrictionInput,
    friction_state: StressFrictionState,
    shear: StressHertzShear,
    full: qd.template(),
    is_restore: bool,
):
    for i_selected in range(envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        state.status[i_b], state.step_principal[i_b] = 0, float("nan")
        state.step_sampled_vm[i_b] = float("nan")
        recovery.status[i_b], recovery.wrench_error[i_b], recovery.tensile_peak[i_b] = 1, 0.0, float("nan")
        regions.is_valid[i_b], state.input_is_valid[i_b], state.observed_step[i_b] = True, True, -1
        if not is_restore:
            binding.is_lost[i_b] = False
    for i_c, i_selected in qd.ndrange(regions.status.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        i_p = i_c * regions.status.shape[1] + i_b
        contacts.valid[i_c, i_b], contacts.status[i_c, i_b], contacts.anchor_face[i_c, i_b] = False, 0, -1
        contacts.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        sources.geom[i_c, i_b], sources.pair[i_c, i_b], sources.vertex[i_c, i_b] = -1, -1, -1
        sources.witness_face[i_c, i_b], sources.status[i_c, i_b] = -1, StressSourceStatus.UNRESOLVED
        sources.free_moment[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        regions.force[i_c, i_b], regions.moment[i_c, i_b] = (
            qd.Vector.zero(gs.qd_float, 3),
            qd.Vector.zero(gs.qd_float, 3),
        )
        regions.leader[i_c, i_b], regions.source_count[i_c, i_b], regions.status[i_c, i_b] = -1, 0, 1
        regions.boundary_planes[i_c, i_b] = qd.Matrix.zero(gs.qd_float, 4, 3)
        regions.n_boundaries[i_c, i_b] = 0
        regions.opponent_type[i_c, i_b] = -1
        law.force[i_p], law.status[i_p], law.peak_pressure[i_p], law.indentation[i_p] = 0.0, 1, 0.0, 0.0
        law.axes[i_p] = qd.Vector.zero(gs.qd_float, 2)
        frames.anchor[i_p], frames.center[i_p] = -1, qd.Vector([float("nan")] * 3)
        frames.basis[i_p] = qd.Matrix.identity(gs.qd_float, 3)
        state.force_error[i_c, i_b], state.moment_error[i_c, i_b] = 0.0, 0.0
        state.sampled_active[i_c, i_b] = False
        friction_regions.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        (
            friction_regions.source_count[i_c, i_b],
            friction_regions.leader[i_c, i_b],
            friction_regions.profile[i_c, i_b],
        ) = (0, -1, 0)
        shear.friction[i_p], shear.force[i_p] = 0.0, qd.Vector.zero(gs.qd_float, 2)
        if not is_restore:
            func_friction_reset(i_p, friction_input, friction_state)
            binding.geom[i_p], binding.pair[i_p], binding.anchor[i_p], binding.radius[i_p] = -1, -1, -1, 0.0
            binding.center[i_p] = qd.Vector.zero(gs.qd_float, 3)
            binding.basis[i_p] = qd.Matrix.identity(gs.qd_float, 3)
    for i_q, i_c, i_selected in qd.ndrange(probes.position.shape[0], regions.status.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        i_p = i_c * regions.status.shape[1] + i_b
        probes.position[i_q, i_p], probes.element[i_q, i_p] = qd.Vector([float("nan")] * 3), -1
        probes.barycentric[i_q, i_p] = qd.Vector.zero(gs.qd_float, 4)
        if qd.static(full):
            probes.tensor[i_q, i_p] = qd.Vector([float("nan")] * 6)
