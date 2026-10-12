"""Stream all witnessed rigid contacts through intrinsic finite-area traction recovery."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import quadrants as qd

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, DataKind, V, iter_data
from genesis.utils.misc import qd_to_numpy

from .compatibility import (
    create_wrench_search,
    kernel_certify_continuous_force,
    kernel_diagnose_wrenches,
)
from .contact import StressContactState, create_contacts
from .data import StressInfo, StressState
from .failure import create_contact_failure, kernel_capture_contact_failure
from .fan_refinement import kernel_refine_failed_fans, kernel_select_failed_fans
from .footprint import (
    StressFootprintState,
    create_anchor_fan_tractions,
    create_footprints,
    create_surface_fan_tractions,
    kernel_anchor_fan_profiles,
    kernel_footprint_ports,
    kernel_refine_failed_disks,
    kernel_surface_fan_profiles,
)
from .paths import StressSurfacePaths
from .provenance import StressSourceState, create_sources, kernel_resolve_sources
from .source_priors import (
    kernel_partition_source_priors,
    kernel_refresh_refined_source_priors,
    kernel_snapshot_source_priors,
)
from .source_vertices import StressSourceVertices, kernel_bind_source_vertices
from .surface import StressSurfaceInfo
from .traction import (
    StressTractionInfo,
    StressTractionState,
    create_traction_info,
    func_source_traction,
    func_traction_constraint_active,
    func_traction_geometry,
    func_traction_source_friction,
    func_traction_support_range,
    kernel_fit_tractions,
    kernel_prepare_cached_tractions,
    kernel_scatter_tractions,
)

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink


# Final original-point wrench and local-cone acceptance retains its independent 1e-8 budget
_TRACTION_FIT_TOLERANCE = 9e-9
_WRENCH_CERTIFICATE_DEPTH = 12


@dataclass(frozen=True)
class StressContactPacking:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    indices: qd.Tensor
    count: qd.Tensor
    maximum: qd.Tensor
    block_start: qd.Tensor = field(default_factory=lambda: V(dtype=gs.qd_int, shape=()))
    block_end: qd.Tensor = field(default_factory=lambda: V(dtype=gs.qd_int, shape=()))


@dataclass(frozen=True)
class StressWrenchCertificates:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    ray: qd.Tensor
    margin: qd.Tensor


class IntrinsicContacts:
    def __init__(
        self,
        link: "RigidLink",
        raw_contacts: StressContactState,
        stress_info: StressInfo,
        surface_info: StressSurfaceInfo,
        options: RigidStressOptions,
        paths: StressSurfacePaths | None,
    ):
        n_contacts, n_envs = raw_contacts.valid.shape
        n_chunk = min(n_contacts, options.contact_block_size)
        body_partition = options.intrinsic_load_partition == "body_source_prior"
        shared_support = options.intrinsic_support == "pair_union" or body_partition
        if shared_support and not body_partition:
            collider = link.solver.collider
            geom_indices = {geom.idx for geom in link.geoms}
            has_large_pair = any(
                is_large and (i_a in geom_indices or i_b in geom_indices)
                for (i_a, i_b), is_large in zip(collider._valid_collision_pairs, collider._large_contact_pair_mask)
            )
            pair_capacity = (
                collider.collider_config.n_contacts_per_nonconvex_pair
                if has_large_pair
                else collider.collider_config.n_contacts_per_convex_pair
            )
            n_chunk = min(n_contacts, max(n_chunk, pair_capacity))
        self.paths = paths if paths is not None else StressSurfacePaths(stress_info, options.surface_edge_subdivisions)
        self.vertices = StressSourceVertices(link, stress_info, options.source_geometry_tolerance)
        self.traction_info = create_traction_info(surface_info)
        self.raw_sources = create_sources(n_contacts, n_envs)
        self.failure = create_contact_failure(n_envs)
        self.raw_sources.uncertainty.fill(options.source_uncertainty)
        self._allocate_workspace(surface_info, options, n_chunk, n_envs)
        self.certificates = StressWrenchCertificates(
            ray=V_VEC(6, dtype=gs.qd_float, shape=(n_contacts, n_envs)),
            margin=V(dtype=gs.qd_float, shape=(n_contacts, n_envs)),
        )
        self.packing = StressContactPacking(
            indices=V(dtype=gs.qd_int, shape=(n_contacts, n_envs)),
            count=V(dtype=gs.qd_int, shape=(n_envs,)),
            maximum=V(dtype=gs.qd_int, shape=()),
            block_start=V(dtype=gs.qd_int, shape=(n_contacts, n_envs) if shared_support and not body_partition else ()),
            block_end=V(dtype=gs.qd_int, shape=(n_contacts, n_envs) if shared_support and not body_partition else ()),
        )

    def _allocate_workspace(
        self, surface_info: StressSurfaceInfo, options: RigidStressOptions, n_chunk: int, n_envs: int
    ) -> None:
        body_partition = options.intrinsic_load_partition == "body_source_prior"
        shared_support = options.intrinsic_support == "pair_union" or body_partition
        self.contacts = create_contacts(
            n_chunk, n_envs, options.contact_radius, cooperative=False, cooperative_scatter=False
        )
        self.sources = create_sources(n_chunk, n_envs)
        self.footprints = create_footprints(
            self.paths.info.positions.shape[0], n_chunk, n_envs, self.paths.info.face_ports.shape[1]
        )
        self.tractions = (
            create_surface_fan_tractions(
                surface_info,
                n_chunk,
                n_envs,
                shared_support=shared_support,
                surface_fan_frontier=options.surface_fan_frontier,
                source_priors=body_partition,
                cached_acceptance=options.cooperative_tractions and gs.backend == gs.cuda,
            )
            if options.intrinsic_integration == "surface_fan"
            else create_anchor_fan_tractions(
                surface_info,
                n_chunk,
                n_envs,
                compact=True,
                packed=True,
                shared_support=shared_support,
                source_priors=body_partition,
                cached_acceptance=options.cooperative_tractions and gs.backend == gs.cuda,
            )
        )
        self.wrench_search = create_wrench_search(n_chunk, n_envs, _WRENCH_CERTIFICATE_DEPTH)

    def scatter(
        self,
        raw_contacts: StressContactState,
        state: StressState,
        stress_info: StressInfo,
        surface_info: StressSurfaceInfo,
        options: RigidStressOptions,
        epsilon: float,
        errno: qd.Tensor,
    ) -> None:
        kernel_pack_intrinsic_contacts(raw_contacts, self.packing, self.certificates, state)
        body_partition = options.intrinsic_load_partition == "body_source_prior"
        n_chunk = self.contacts.valid.shape[0]
        if options.intrinsic_support == "pair_union" and not body_partition:
            kernel_pack_contact_groups(raw_contacts, self.raw_sources, self.packing, n_chunk)
        n_work = int(qd_to_numpy(self.packing.maximum, transpose=True))
        if body_partition and n_work > n_chunk:
            # Contact count is an input, not a fixed property of a grasp. Grow
            # the complete workspace before loading; never truncate a body union.
            n_chunk = min(raw_contacts.valid.shape[0], max(n_work, 2 * n_chunk))
            self._allocate_workspace(surface_info, options, n_chunk, raw_contacts.valid.shape[1])
        stride = 1 if options.intrinsic_support == "pair_union" and not body_partition else n_chunk
        for i_start in range(0, n_work, stride):
            kernel_load_intrinsic_contacts(
                i_start, raw_contacts, self.raw_sources, self.contacts, self.sources, self.packing
            )
            if self.vertices.info is not None:
                kernel_bind_source_vertices(self.contacts, self.sources, self.vertices.info, epsilon)
            kernel_resolve_sources(
                options.source_projection_limit, epsilon, self.contacts, self.sources, stress_info, surface_info
            )
            kernel_footprint_ports(self.contacts, self.sources, self.footprints, surface_info, self.paths.info)
            if options.intrinsic_integration == "surface_fan":
                kernel_surface_fan_profiles(
                    self.contacts,
                    self.footprints,
                    self.tractions,
                    stress_info,
                    surface_info,
                    self.paths.info,
                    self.traction_info,
                    epsilon,
                )
            else:
                kernel_anchor_fan_profiles(
                    self.contacts, self.footprints, self.tractions, stress_info, surface_info, self.paths.info
                )
            if options.intrinsic_support == "pair_union" or body_partition:
                kernel_bind_traction_support(
                    self.contacts, self.sources, self.footprints, self.tractions, body_union=body_partition
                )
            if body_partition:
                kernel_snapshot_source_priors(self.contacts, self.sources, self.tractions)
                kernel_partition_source_priors(self.contacts, self.sources, self.tractions)
            kernel_fit_tractions(
                self.contacts,
                self.sources,
                self.tractions,
                self.traction_info,
                epsilon,
                _TRACTION_FIT_TOLERANCE,
                128,
                cooperative=options.cooperative_tractions,
                warm_start=options.warm_start_tractions,
            )
            if options.intrinsic_integration == "surface_fan":
                kernel_refine_failed_disks(
                    self.contacts,
                    self.sources,
                    self.footprints,
                    self.tractions,
                    surface_info,
                    self.traction_info,
                    epsilon,
                )
                if body_partition:
                    kernel_refresh_refined_source_priors(self.contacts, self.tractions)
                kernel_fit_tractions(
                    self.contacts,
                    self.sources,
                    self.tractions,
                    self.traction_info,
                    epsilon,
                    _TRACTION_FIT_TOLERANCE,
                    128,
                    retry_mode=2,
                    cooperative=options.cooperative_tractions,
                    warm_start=options.warm_start_tractions,
                )
                kernel_select_failed_fans(self.contacts, self.footprints, self.tractions)
                kernel_refine_failed_fans(
                    self.contacts,
                    self.sources,
                    self.footprints,
                    self.tractions,
                    stress_info,
                    surface_info,
                    self.paths.info,
                    self.traction_info,
                    epsilon,
                )
                if body_partition:
                    kernel_refresh_refined_source_priors(self.contacts, self.tractions)
                kernel_fit_tractions(
                    self.contacts,
                    self.sources,
                    self.tractions,
                    self.traction_info,
                    epsilon,
                    _TRACTION_FIT_TOLERANCE,
                    128,
                    retry_mode=3,
                    cooperative=options.cooperative_tractions,
                )
                kernel_fit_tractions(
                    self.contacts,
                    self.sources,
                    self.tractions,
                    self.traction_info,
                    epsilon,
                    _TRACTION_FIT_TOLERANCE,
                    128,
                    retry_mode=4,
                    cooperative=options.cooperative_tractions,
                )
            kernel_certify_continuous_force(self.contacts, self.tractions, surface_info, epsilon, 1e-8)
            kernel_diagnose_wrenches(
                self.contacts,
                self.sources,
                self.tractions,
                self.wrench_search,
                self.traction_info,
                surface_info,
                epsilon,
                1e-8,
            )
            fused_acceptance = len(self.tractions.acceptance.force.shape) == 2
            if fused_acceptance:
                kernel_prepare_cached_tractions(
                    epsilon,
                    self.contacts,
                    self.tractions,
                    state,
                    stress_info,
                    surface_info,
                    self.traction_info,
                    errno,
                    reset_force=False,
                )
            kernel_commit_intrinsic_contacts(
                i_start,
                epsilon,
                raw_contacts,
                self.raw_sources,
                self.contacts,
                self.sources,
                self.footprints,
                self.tractions,
                self.traction_info,
                self.packing,
                self.certificates,
                cooperative=options.cooperative_tractions,
            )
            if body_partition:
                kernel_commit_body_partition(raw_contacts, self.packing)
            if not fused_acceptance:
                kernel_scatter_tractions(
                    self.contacts,
                    self.tractions,
                    state,
                    stress_info,
                    surface_info,
                    self.traction_info,
                    errno,
                    reset_force=False,
                    cooperative=options.cooperative_tractions,
                )
        kernel_capture_contact_failure(raw_contacts, self.raw_sources, self.failure, errno, grouped=body_partition)

    def data(self, prefix: str):
        for name, value in (
            ("vertices", self.vertices.info),
            ("traction_info", self.traction_info),
            ("raw_sources", self.raw_sources),
            ("failure", self.failure),
            ("contacts", self.contacts),
            ("sources", self.sources),
            ("footprints", self.footprints),
            ("tractions", self.tractions),
            ("packing", self.packing),
            ("certificates", self.certificates),
            ("wrench_search", self.wrench_search),
        ):
            if value is not None:
                yield from iter_data(value, f"{prefix}.{name}")


@qd.kernel
def kernel_pack_intrinsic_contacts(
    raw_contacts: StressContactState,
    packing: StressContactPacking,
    certificates: StressWrenchCertificates,
    state: StressState,
):
    for _ in range(1):
        packing.maximum[None] = 0
    if qd.static(gs.backend == gs.cuda):
        for i_b in range(packing.count.shape[0]):
            packing.count[i_b] = 0
    for i_c, i_b in qd.ndrange(raw_contacts.valid.shape[0], raw_contacts.valid.shape[1]):
        certificates.ray[i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
        certificates.margin[i_c, i_b] = 0.0
        raw_contacts.status[i_c, i_b] = 0
        raw_contacts.force_error[i_c, i_b] = 0.0
        raw_contacts.moment_error[i_c, i_b] = 0.0
        raw_contacts.evaluations[i_c, i_b] = 0
        raw_contacts.candidate_count[i_c, i_b] = 0
        if qd.static(gs.backend == gs.cuda):  # noqa: SIM102 — omit atomic collection in the CPU specialization.
            if raw_contacts.valid[i_c, i_b]:
                destination = qd.atomic_add(packing.count[i_b], 1)
                packing.indices[destination, i_b] = i_c
    for i_b in range(packing.count.shape[0]):
        n_contacts = 0
        if qd.static(gs.backend == gs.cuda):
            n_contacts = packing.count[i_b]
            if n_contacts <= 32:
                # Sort the short atomic list back into its exact stable order.
                for i_c in range(1, n_contacts):
                    value = packing.indices[i_c, i_b]
                    destination = i_c
                    for i_j in range(i_c):
                        if packing.indices[i_j, i_b] > value:
                            destination = qd.min(destination, i_j)
                    for offset in range(i_c - destination):
                        source = i_c - 1 - offset
                        packing.indices[source + 1, i_b] = packing.indices[source, i_b]
                    packing.indices[destination, i_b] = value
            else:
                # Dense lists retain linear work rather than quadratic tails.
                destination = 0
                for i_c in range(raw_contacts.valid.shape[0]):
                    if raw_contacts.valid[i_c, i_b]:
                        packing.indices[destination, i_b] = i_c
                        destination += 1
        else:
            for i_c in range(raw_contacts.valid.shape[0]):
                if raw_contacts.valid[i_c, i_b]:
                    packing.indices[n_contacts, i_b] = i_c
                    n_contacts += 1
            packing.count[i_b] = n_contacts
        qd.atomic_max(packing.maximum[None], n_contacts)
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.force.shape[1]):
        state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_pack_contact_groups(
    raw_contacts: StressContactState,
    raw_sources: StressSourceState,
    packing: StressContactPacking,
    n_chunk: int,
):
    """Pack complete geom/pair groups without padding the raw contact index array."""
    for _ in range(1):
        packing.maximum[None] = 0
    for i_c, i_b in qd.ndrange(packing.indices.shape[0], packing.indices.shape[1]):
        packing.block_start[i_c, i_b] = 0
        packing.block_end[i_c, i_b] = 0
    for i_b in range(packing.count.shape[0]):
        n_contacts = packing.count[i_b]
        # Sort only actual contacts. Raw input indices stay intact for commit.
        for i_c in range(1, n_contacts):
            for i_reverse in range(i_c):
                i_right = i_c - i_reverse
                i_a = packing.indices[i_right - 1, i_b]
                i_z = packing.indices[i_right, i_b]
                pair_a, pair_z = raw_sources.pair[i_a, i_b], raw_sources.pair[i_z, i_b]
                geom_a, geom_z = raw_sources.geom[i_a, i_b], raw_sources.geom[i_z, i_b]
                if pair_a > pair_z or (pair_a == pair_z and geom_a > geom_z):
                    packing.indices[i_right - 1, i_b] = i_z
                    packing.indices[i_right, i_b] = i_a
        n_blocks, block_size, group_start = 0, 0, 0
        for i_c in range(n_contacts):
            i_raw = packing.indices[i_c, i_b]
            is_end = i_c + 1 == n_contacts
            if not is_end:
                i_next = packing.indices[i_c + 1, i_b]
                pair, geom = raw_sources.pair[i_raw, i_b], raw_sources.geom[i_raw, i_b]
                is_end = pair < 0 or geom < 0
                is_end = is_end or pair != raw_sources.pair[i_next, i_b] or geom != raw_sources.geom[i_next, i_b]
            if is_end:
                group_size = i_c + 1 - group_start
                # Scene construction uses the collider's authoritative pair cap.
                assert group_size <= n_chunk
                if block_size == 0 or block_size + group_size > n_chunk:
                    packing.block_start[n_blocks, i_b] = group_start
                    n_blocks += 1
                    block_size = 0
                block_size += group_size
                packing.block_end[n_blocks - 1, i_b] = i_c + 1
                group_start = i_c + 1
        qd.atomic_max(packing.maximum[None], n_blocks)


def kernel_bind_traction_support(
    contact_state: StressContactState,
    source_state: StressSourceState,
    footprint_state: StressFootprintState,
    traction_state: StressTractionState,
    body_union: bool = False,
):
    kernel_bind_traction_support_impl(contact_state, source_state, footprint_state, traction_state, body_union)


@qd.kernel
def kernel_bind_traction_support_impl(
    contact_state: StressContactState,
    source_state: StressSourceState,
    footprint_state: StressFootprintState,
    traction_state: StressTractionState,
    body_union: qd.template(),
):
    """Bind complete pair-union sources, including valid zero-load witnesses."""
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        start, end = i_c, i_c + 1
        valid = contact_state.valid[i_c, i_b] and source_state.status[i_c, i_b] == 0
        valid = valid and footprint_state.status[i_c, i_b] == 0
        pair, geom = source_state.pair[i_c, i_b], source_state.geom[i_c, i_b]
        if qd.static(body_union):
            start, end = 0, 0
            for i_s in range(contact_state.valid.shape[0]):
                if contact_state.valid[i_s, i_b]:
                    end = i_s + 1
        elif contact_state.valid[i_c, i_b] and pair >= 0 and geom >= 0:
            for i_s in range(contact_state.valid.shape[0]):
                if (
                    contact_state.valid[i_s, i_b]
                    and pair == source_state.pair[i_s, i_b]
                    and geom == source_state.geom[i_s, i_b]
                ):
                    start, end = qd.min(start, i_s), qd.max(end, i_s + 1)
            for i_s in range(start, end):
                # The packer establishes contiguity; a malformed standalone
                # subproblem must fail rather than borrow another pair's samples.
                valid = valid and contact_state.valid[i_s, i_b]
                valid = valid and pair == source_state.pair[i_s, i_b] and geom == source_state.geom[i_s, i_b]
        traction_state.support.start[i_c, i_b] = start
        traction_state.support.end[i_c, i_b] = end
        traction_state.support.valid[i_c, i_b] = valid
        traction_state.support.center[i_c, i_b] = contact_state.center[i_c, i_b]
        traction_state.support.radius[i_c, i_b] = contact_state.radius[i_c, i_b]
        traction_state.support.friction[i_c, i_b] = contact_state.friction[i_c, i_b]


@qd.kernel
def kernel_load_intrinsic_contacts(
    i_start: int,
    raw_contacts: StressContactState,
    raw_sources: StressSourceState,
    contact_state: StressContactState,
    source_state: StressSourceState,
    packing: StressContactPacking,
):
    for i_b in range(packing.count.shape[0]):
        contact_state.frame_position[i_b] = raw_contacts.frame_position[i_b]
        contact_state.frame_quaternion[i_b] = raw_contacts.frame_quaternion[i_b]
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        contact_state.valid[i_c, i_b] = False
        contact_state.status[i_c, i_b] = 0
        start, end = i_start, packing.count[i_b]
        if qd.static(len(packing.block_start.shape) == 2):
            start, end = packing.block_start[i_start, i_b], packing.block_end[i_start, i_b]
        if start + i_c < end:
            i_raw = packing.indices[start + i_c, i_b]
            contact_state.valid[i_c, i_b] = True
            contact_state.position[i_c, i_b] = raw_contacts.position[i_raw, i_b]
            contact_state.force[i_c, i_b] = raw_contacts.force[i_raw, i_b]
            contact_state.normal[i_c, i_b] = raw_contacts.normal[i_raw, i_b]
            contact_state.friction[i_c, i_b] = raw_contacts.friction[i_raw, i_b]
            contact_state.radius[i_c, i_b] = raw_contacts.radius[i_raw, i_b]
            source_state.geom[i_c, i_b] = raw_sources.geom[i_raw, i_b]
            source_state.pair[i_c, i_b] = raw_sources.pair[i_raw, i_b]
            source_state.method[i_c, i_b] = raw_sources.method[i_raw, i_b]
            source_state.vertex[i_c, i_b] = raw_sources.vertex[i_raw, i_b]
            source_state.witness[i_c, i_b] = raw_sources.witness[i_raw, i_b]
            source_state.witness_face[i_c, i_b] = raw_sources.witness_face[i_raw, i_b]
            source_state.witness_barycentric[i_c, i_b] = raw_sources.witness_barycentric[i_raw, i_b]
            source_state.uncertainty[i_c, i_b] = raw_sources.uncertainty[i_raw, i_b]
            source_state.free_moment[i_c, i_b] = raw_sources.free_moment[i_raw, i_b]
            source_state.has_vertex_ambiguity[i_c, i_b] = raw_sources.has_vertex_ambiguity[i_raw, i_b]


def kernel_commit_intrinsic_contacts(
    i_start: int,
    epsilon: float,
    raw_contacts: StressContactState,
    raw_sources: StressSourceState,
    contact_state: StressContactState,
    source_state: StressSourceState,
    footprint_state: StressFootprintState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
    packing: StressContactPacking,
    certificates: StressWrenchCertificates,
    cooperative: bool = True,
):
    if cooperative and gs.backend == gs.cuda:
        from .cooperative_scatter import kernel_commit_intrinsic_contacts as kernel_commit_cooperative

        kernel_commit_cooperative(
            i_start,
            epsilon,
            raw_contacts,
            raw_sources,
            contact_state,
            source_state,
            footprint_state,
            traction_state,
            traction_info,
            packing,
            certificates,
        )
    else:
        kernel_commit_intrinsic_contacts_impl(
            i_start,
            epsilon,
            raw_contacts,
            raw_sources,
            contact_state,
            source_state,
            footprint_state,
            traction_state,
            traction_info,
            packing,
            certificates,
        )


@qd.kernel
def kernel_commit_intrinsic_contacts_impl(
    i_start: int,
    epsilon: float,
    raw_contacts: StressContactState,
    raw_sources: StressSourceState,
    contact_state: StressContactState,
    source_state: StressSourceState,
    footprint_state: StressFootprintState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
    packing: StressContactPacking,
    certificates: StressWrenchCertificates,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b]:
            start = i_start
            if qd.static(len(packing.block_start.shape) == 2):
                start = packing.block_start[i_start, i_b]
            i_raw = packing.indices[start + i_c, i_b]
            raw_contacts.center[i_raw, i_b] = contact_state.center[i_c, i_b]
            raw_contacts.anchor_face[i_raw, i_b] = contact_state.anchor_face[i_c, i_b]
            raw_sources.barycentric[i_raw, i_b] = source_state.barycentric[i_c, i_b]
            if qd.static(len(traction_state.priors.force.shape) == 2):
                raw_sources.anchor_moment[i_raw, i_b] = traction_state.priors.anchor_moment[i_c, i_b]
            else:
                raw_sources.anchor_moment[i_raw, i_b] = source_state.anchor_moment[i_c, i_b]
            raw_sources.status[i_raw, i_b] = source_state.status[i_c, i_b]
            raw_contacts.candidate_count[i_raw, i_b] = footprint_state.samples[i_c, i_b]
            if qd.static(len(traction_state.fan_refined.shape) == 2):
                raw_contacts.is_refined[i_raw, i_b] = (
                    traction_state.circle.refined[i_c, i_b] or traction_state.fan_refined[i_c, i_b]
                )
            raw_contacts.evaluations[i_raw, i_b] = traction_state.evaluations[i_c, i_b]
            certificates.ray[i_raw, i_b] = traction_state.certificate[i_c, i_b]
            certificates.margin[i_raw, i_b] = traction_state.certificate_margin[i_c, i_b]
            force_scale = contact_state.force[i_c, i_b].norm()
            if force_scale == 0.0:
                force_scale = traction_state.scale[i_c, i_b]
            raw_contacts.force_error[i_raw, i_b] = traction_state.force_error[i_c, i_b] * force_scale
            raw_contacts.moment_error[i_raw, i_b] = (
                traction_state.moment_error[i_c, i_b] * force_scale * contact_state.radius[i_c, i_b]
            )
            status = 0
            if source_state.status[i_c, i_b] != 0:
                status = 10 + source_state.status[i_c, i_b]
            elif footprint_state.status[i_c, i_b] != 0:
                status = 20 + footprint_state.status[i_c, i_b]
            elif traction_state.status[i_c, i_b] != 0:
                status = 30 + traction_state.status[i_c, i_b]
            elif func_traction_constraint_active(i_c, i_b, traction_state):
                total, moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
                scale = qd.max(contact_state.force[i_c, i_b].norm(), traction_state.scale[i_c, i_b])
                cone_tolerance = 64.0 * epsilon * scale
                cone_valid = True
                first, last = func_traction_support_range(i_c, i_b, traction_state)
                for i_s in range(first, last):
                    friction = func_traction_source_friction(i_s, i_b, contact_state.friction[i_c, i_b], traction_state)
                    for i_q in range(traction_state.sample_count[i_s, i_b]):
                        force = func_source_traction(i_q, i_s, i_c, i_b, contact_state, traction_state, traction_info)
                        position, normal = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                        compression = force.dot(normal)
                        tangent = force - compression * normal
                        cone_valid = cone_valid and compression >= -cone_tolerance
                        cone_valid = cone_valid and tangent.norm() <= friction * compression + cone_tolerance
                        total += force
                        moment += (position - contact_state.position[i_c, i_b]).cross(force)
                force_error = (total - contact_state.force[i_c, i_b]).norm()
                moment_error = (moment - source_state.free_moment[i_c, i_b]).norm()
                raw_contacts.force_error[i_raw, i_b] = force_error
                raw_contacts.moment_error[i_raw, i_b] = moment_error
                budget_scale = contact_state.force[i_c, i_b].norm()
                if budget_scale == 0.0:
                    budget_scale = scale
                budget = 1e-8 * budget_scale
                if not cone_valid or force_error > budget or moment_error > budget * contact_state.radius[i_c, i_b]:
                    status = 40
            raw_contacts.status[i_raw, i_b] = status


@qd.kernel
def kernel_commit_body_partition(raw_contacts: StressContactState, packing: StressContactPacking):
    """A failed aggregate constraint invalidates all its physical member loads."""
    for i_b in range(packing.count.shape[0]):
        if packing.count[i_b] > 0:
            leader = packing.indices[0, i_b]
            for i_c in range(1, packing.count[i_b]):
                i_raw = packing.indices[i_c, i_b]
                # Retain a member's specific source/integration diagnostic.
                if raw_contacts.status[i_raw, i_b] == 0:
                    raw_contacts.status[i_raw, i_b] = raw_contacts.status[leader, i_b]
                raw_contacts.force_error[i_raw, i_b] = raw_contacts.force_error[leader, i_b]
                raw_contacts.moment_error[i_raw, i_b] = raw_contacts.moment_error[leader, i_b]
