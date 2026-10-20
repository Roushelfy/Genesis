"""Conservative P2 work transfer for already integrated material-boundary samples."""

from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions

from .contact import StressContactState, func_contact_force_admissible
from .data import StressInfo, StressState
from .provenance import StressSourceState, StressSourceStatus, create_sources, kernel_resolve_sources
from .source_vertices import StressSourceVertices, kernel_bind_source_vertices
from .surface import StressSurfaceInfo

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink


class ResolvedContacts:
    def __init__(self, link: "RigidLink", contacts: StressContactState, info: StressInfo, options: RigidStressOptions):
        self.raw_sources = create_sources(*contacts.valid.shape)
        self.raw_sources.uncertainty.fill(options.source_uncertainty)
        self.vertices = StressSourceVertices(link, info, options.source_geometry_tolerance)

    def scatter(
        self,
        contacts: StressContactState,
        state: StressState,
        info: StressInfo,
        surface: StressSurfaceInfo,
        options: RigidStressOptions,
        epsilon: float,
    ) -> None:
        if self.vertices.info is not None:
            kernel_bind_source_vertices(contacts, self.raw_sources, self.vertices.info, epsilon)
        kernel_resolve_sources(options.source_projection_limit, epsilon, contacts, self.raw_sources, info, surface)
        kernel_scatter_resolved(epsilon, contacts, self.raw_sources, state, info, surface)


@qd.kernel
def kernel_scatter_resolved(
    source_epsilon: float,
    contact_state: StressContactState,
    source_state: StressSourceState,
    stress_state: StressState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
):
    for i_n, i_b in qd.ndrange(stress_state.force.shape[0], stress_state.force.shape[1]):
        stress_state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        contact_state.status[i_c, i_b] = 0
        contact_state.force_error[i_c, i_b] = 0.0
        contact_state.moment_error[i_c, i_b] = 0.0
        contact_state.candidate_count[i_c, i_b] = 0
        contact_state.evaluations[i_c, i_b] = 0
        contact_state.is_refined[i_c, i_b] = False
        if contact_state.valid[i_c, i_b]:
            force = contact_state.force[i_c, i_b]
            point = contact_state.position[i_c, i_b]
            admitted = func_contact_force_admissible(
                point, force, contact_state.normal[i_c, i_b], contact_state.friction[i_c, i_b], source_epsilon
            )
            status = 0
            if not admitted:
                status = 1
            elif source_state.status[i_c, i_b] != StressSourceStatus.SUCCESS:
                status = 10 + source_state.status[i_c, i_b]
            else:
                i_face = contact_state.anchor_face[i_c, i_b]
                bary = source_state.barycentric[i_c, i_b]
                shape = qd.Vector(
                    [
                        bary[0] * (2.0 * bary[0] - 1.0),
                        bary[1] * (2.0 * bary[1] - 1.0),
                        bary[2] * (2.0 * bary[2] - 1.0),
                        4.0 * bary[0] * bary[1],
                        4.0 * bary[0] * bary[2],
                        4.0 * bary[1] * bary[2],
                    ]
                )
                # Signed corner weights are required for virtual work; clipping
                # them would change the physical sample's force and moment.
                position = qd.Vector.zero(gs.qd_float, 3)
                for i_corner in qd.static(range(6)):
                    i_node = stress_info.surface_nodes[i_face, i_corner]
                    position += shape[i_corner] * stress_info.vertices[i_node]
                scale = qd.max(
                    surface_info.face_edges[i_face][0, :].norm(), surface_info.face_edges[i_face][1, :].norm()
                )
                force_error = qd.abs(shape.sum() - 1.0) * force.norm()
                moment_error = ((position - point).cross(force) - source_state.free_moment[i_c, i_b]).norm()
                contact_state.force_error[i_c, i_b] = force_error
                contact_state.moment_error[i_c, i_b] = moment_error
                contact_state.candidate_count[i_c, i_b] = 1
                contact_state.evaluations[i_c, i_b] = 1
                if force_error > 1e-8 * force.norm() or moment_error > 1e-8 * force.norm() * scale:
                    status = 4
                if status == 0:
                    for i_corner in qd.static(range(6)):
                        i_node = stress_info.surface_nodes[i_face, i_corner]
                        stress_state.force[i_node, i_b] += shape[i_corner] * force
            contact_state.status[i_c, i_b] = status
