"""Measure strict Hertz transfer and coarse elastic recovery in the moving Franka scene.

Rejected observations remain NaN while the independent rigid rollout continues for diagnostics. The pressure
uses proportional shear; near-field enrichment and friction memory remain separate physical-model work.
"""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, func_contact_force_admissible
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegionState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.source_vertices import StressSourceVertices, kernel_bind_source_vertices
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.utils import geom
from genesis.utils.array_class import V, V_VEC

from .live_hertz_inputs import InputObserver, array, main
from .native_hertz_transfer import HertzPatchState, HertzSampleState, HertzSurfaceTransfer, func_projected_surface_walk


@dataclass(frozen=True)
class SceneObservationState:
    status: qd.Tensor
    is_loaded: qd.Tensor
    friction: qd.Tensor
    required_friction: qd.Tensor
    counts: qd.Tensor


@qd.kernel
def kernel_prepare_patches(
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressHertzRegionState,
    patches: HertzPatchState,
    observation: SceneObservationState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    for i_b in range(regions.is_valid.shape[0]):
        observation.status[i_b] = 0 if regions.is_valid[i_b] else 1
        observation.is_loaded[i_b] = False
    for i_c, i_b in qd.ndrange(regions.status.shape[0], regions.status.shape[1]):
        i_p = i_c * regions.status.shape[1] + i_b
        patches.environment[i_p] = i_b
        patches.anchor[i_p] = -1
        patches.center[i_p] = qd.Vector.zero(gs.qd_float, 3)
        patches.normal[i_p] = qd.Vector([0.0, 0.0, 1.0])
        patches.tangent[i_p] = qd.Vector([1.0, 0.0, 0.0])
        patches.force[i_p] = qd.Vector.zero(gs.qd_float, 3)
        patches.moment[i_p] = qd.Vector.zero(gs.qd_float, 3)
        observation.friction[i_p] = float("inf")
        observation.required_friction[i_p] = 0.0
        if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0:
            observation.is_loaded[i_b] = True
            if sources.status[i_c, i_b] != 0:
                qd.atomic_or(observation.status[i_b], 2)
        if regions.source_count[i_c, i_b] > 0:
            rotation, origin = contacts.frame_quaternion[i_b], contacts.frame_position[i_b]
            center = geom.qd_inv_transform_by_quat(regions.center[i_c, i_b] - origin, rotation)
            normal = geom.qd_inv_transform_by_quat(regions.normal[i_c, i_b], rotation)
            patches.center[i_p] = center
            patches.normal[i_p] = normal
            patches.tangent[i_p] = geom.qd_inv_transform_by_quat(regions.tangent[i_c, i_b], rotation)
            patches.force[i_p] = geom.qd_inv_transform_by_quat(regions.force[i_c, i_b], rotation)
            patches.moment[i_p] = geom.qd_inv_transform_by_quat(regions.moment[i_c, i_b], rotation)
            if sources.status[i_c, i_b] == 0:
                anchor, barycentric, position = func_projected_surface_walk(
                    contacts.anchor_face[i_c, i_b],
                    contacts.center[i_c, i_b],
                    center - contacts.center[i_c, i_b],
                    normal,
                    info,
                    surface,
                    epsilon,
                )
                patches.anchor[i_p] = anchor
    for i_c, i_b in qd.ndrange(regions.status.shape[0], regions.status.shape[1]):
        i_l = regions.leader[i_c, i_b]
        if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0 and i_l >= 0:
            i_p = i_l * regions.status.shape[1] + i_b
            qd.atomic_min(observation.friction[i_p], contacts.friction[i_c, i_b])


@qd.kernel
def kernel_accept_surface_load(
    patches: HertzPatchState,
    samples: HertzSampleState,
    observation: SceneObservationState,
    state: StressState,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    for i_b in range(observation.status.shape[0]):
        if not samples.admitted[i_b]:
            qd.atomic_or(observation.status[i_b], 4)
    for i_s, i_p in qd.ndrange(samples.face.shape[0], samples.face.shape[1]):
        i_f = samples.face[i_s, i_p]
        if i_f >= 0 and patches.force[i_p].norm() > 0:
            force, normal = patches.force[i_p], -surface.face_normal[i_f]
            compression = force.dot(normal)
            tangent = force - compression * normal
            required = tangent.norm() / qd.max(compression, epsilon * force.norm())
            qd.atomic_max(observation.required_friction[i_p], required)
            if not func_contact_force_admissible(
                samples.position[i_s, i_p], force, normal, observation.friction[i_p], epsilon
            ):
                qd.atomic_or(observation.status[patches.environment[i_p]], 8)
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.force.shape[1]):
        if observation.status[i_b] != 0:
            state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_finish_observation(observation: SceneObservationState, state: StressState, full: qd.template()):
    for i_b in range(observation.status.shape[0]):
        if not state.valid[i_b]:
            qd.atomic_or(observation.status[i_b], 16)
        state.step_valid[i_b] = state.step_valid[i_b] and observation.status[i_b] == 0
        if not state.step_valid[i_b]:
            state.peak[i_b] = float("nan")
            state.step_peak[i_b] = float("nan")
    if qd.static(full):
        for i_e, i_q, i_b in qd.ndrange(state.stress_tensor.shape[0], 4, state.active.shape[0]):
            if not state.step_valid[i_b]:
                state.stress_tensor[i_e, i_q, i_b] = qd.Vector([float("nan")] * 6)
                state.von_mises[i_e, i_q, i_b] = qd.Vector([float("nan")])


@qd.kernel
def kernel_audit_observation(observation: SceneObservationState):
    for i_b in range(observation.status.shape[0]):
        status = observation.status[i_b]
        observation.counts[i_b][0] += 1
        if observation.is_loaded[i_b]:
            observation.counts[i_b][1] += 1
        if status == 0:
            observation.counts[i_b][2] += 1
            if observation.is_loaded[i_b]:
                observation.counts[i_b][3] += 1
        for i in qd.static(range(5)):
            if status & (1 << i):
                observation.counts[i_b][4 + i] += 1


class SceneObserver(InputObserver):
    def __init__(self, solver, check_grouping=False):
        super().__init__(solver, check_grouping)
        shape = self.entry.contacts.valid.shape
        self.transfer = HertzSurfaceTransfer(shape[0] * shape[1], shape[1], clear_inactive=False)
        self.vertices = StressSourceVertices(
            self.entry.link, self.entry.model.info, self.entry.options.source_geometry_tolerance
        )
        self.sources.uncertainty.fill(self.entry.options.source_uncertainty)
        self.observation = SceneObservationState(
            status=V(dtype=gs.qd_int, shape=(shape[1],)),
            is_loaded=V(dtype=gs.qd_bool, shape=(shape[1],)),
            friction=V(dtype=gs.qd_float, shape=(shape[0] * shape[1],)),
            required_friction=V(dtype=gs.qd_float, shape=(shape[0] * shape[1],)),
            counts=V_VEC(9, dtype=gs.qd_int, shape=(shape[1],)),
        )

    def update(self, recovery, i_substep):
        super().update(recovery, i_substep)
        entry = self.entry
        if self.vertices.info is not None:
            self.phase(
                "boundary_vertices",
                kernel_bind_source_vertices,
                entry.contacts,
                self.sources,
                self.vertices.info,
                self.epsilon,
            )
        self.phase(
            "boundary_sources",
            kernel_resolve_sources,
            entry.options.source_projection_limit,
            self.epsilon,
            entry.contacts,
            self.sources,
            entry.model.info,
            entry.surface.info,
        )
        self.phase(
            "boundary_patches",
            kernel_prepare_patches,
            entry.contacts,
            self.sources,
            self.regions.state,
            self.transfer.patches,
            self.observation,
            entry.model.info,
            entry.surface.info,
            self.epsilon,
        )
        self.phase(
            "traction_transfer",
            self.transfer.scatter,
            self.hertz.state,
            entry.state,
            entry.model.info,
            entry.surface.info,
            self.epsilon,
        )
        self.phase(
            "load_acceptance",
            kernel_accept_surface_load,
            self.transfer.patches,
            self.transfer.samples,
            self.observation,
            entry.state,
            entry.surface.info,
            self.epsilon,
        )
        self.phase(
            "elastic_recovery_and_peak", entry.model.recover, entry.omega, entry.state, entry.options, None, True
        )
        self.phase(
            "output_acceptance", kernel_finish_observation, self.observation, entry.state, entry.output_mode == "full"
        )
        if self.audit_enabled:
            self.phase("research_scene_audit", kernel_audit_observation, self.observation)

    def reset_audit(self):
        super().reset_audit()
        self.observation.counts.fill(0)

    def report(self):
        return super().report() | {
            "cleared_inactive_sample_geometry": self.transfer.clear_inactive,
            "scene_count_columns": [
                "steps",
                "loaded",
                "admitted",
                "admitted_loaded",
                "geometry",
                "source",
                "transfer",
                "cone",
                "solve",
            ],
            "scene_counts": array(self.observation.counts).tolist(),
        }

    def snapshot(self):
        shape = self.entry.contacts.valid.shape
        result = super().snapshot() | {
            "scene_status": array(self.observation.status),
            "scene_peak_pa": array(self.entry.state.step_peak),
            "source_projection_status": array(self.sources.status),
            "patch_status": array(self.transfer.patches.status).reshape(shape).T,
            "patch_moment_error_nm": array(self.transfer.patches.moment_error).reshape(shape).T,
            "patch_required_friction": array(self.observation.required_friction).reshape(shape).T,
            "nodal_force": array(self.entry.state.force),
            "omega": array(self.entry.omega),
            "balanced_rhs": array(self.entry.state.rhs),
            "displacement": array(self.entry.state.displacement),
            "elastic_valid": array(self.entry.state.valid),
        }
        if self.entry.output_mode == "full":
            result["stress_tensor"] = array(self.entry.state.stress_tensor)
            result["von_mises"] = array(self.entry.state.von_mises)
        return result


if __name__ == "__main__":
    main(
        SceneObserver,
        ("max", "full", "rigid"),
        "Same-substep native geometry, Hertz pressure, witnessed finite-footprint transfer, original wrench and actual-facet friction acceptance, coarse P2 elastic recovery and max/full output. Rejected observations are NaN and excluded from admitted throughput while the rigid diagnostic rollout continues. Proportional shear has no friction memory; coarse fields lack Hertz near-field enrichment. These timings do not qualify physical shell stress accuracy or a usable public estimator.",
    )
