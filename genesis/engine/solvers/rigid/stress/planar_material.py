"""Resolve original quadrilateral loading components independently of proxy task storage."""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V, of_kind

from .contact import StressContactState, func_contact_sample, func_contact_section, func_force_frame
from .planar_geometry import PlanarLoadingInfo
from .quadrilateral import func_quad_inverse
from .surface import StressSurfaceInfo


class PlanarMaterialStatus(IntEnum):
    SUCCESS = 0
    INVALID_ANCHOR = 1
    UNRESOLVED_SEED = 2
    AMBIGUOUS_ANCHOR = 3
    NONFINITE_FOOTPRINT = 4


@dataclass(frozen=True)
class PlanarMaterialState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    mask: qd.Tensor
    hits: qd.Tensor
    queried: qd.Tensor
    status: qd.Tensor
    selected_count: qd.Tensor
    excluded_count: qd.Tensor
    query_contacts: qd.Tensor = of_kind(DataKind.STATE)  # noqa: RUF009 - typed array metadata
    rejected_contacts: qd.Tensor = of_kind(DataKind.STATE)  # noqa: RUF009 - typed array metadata


def create_planar_material(n_parents: int, n_contacts: int, n_envs: int) -> PlanarMaterialState:
    """Allocate packed original-parent membership and source-level query diagnostics."""
    shape = (n_contacts, n_envs)
    state = PlanarMaterialState(
        mask=V_VEC(1, dtype=gs.qd_int, shape=((n_parents + 31) // 32, n_contacts, n_envs)),
        hits=V_VEC(1, dtype=gs.qd_int, shape=((n_parents + 31) // 32, n_contacts, n_envs)),
        queried=V(dtype=gs.qd_bool, shape=shape),
        status=V(dtype=gs.qd_int, shape=shape),
        selected_count=V(dtype=gs.qd_int, shape=shape),
        excluded_count=V(dtype=gs.qd_int, shape=shape),
        query_contacts=V(dtype=gs.qd_int, shape=(n_envs,)),
        rejected_contacts=V(dtype=gs.qd_int, shape=(n_envs,)),
    )
    state.query_contacts.fill(0)
    state.rejected_contacts.fill(0)
    return state


@qd.kernel
def kernel_planar_material(
    source_epsilon: float,
    contacts: StressContactState,
    fallback: qd.Tensor,
    surface: StressSurfaceInfo,
    geometry: PlanarLoadingInfo,
    state: PlanarMaterialState,
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        state.queried[i_c, i_b] = False
        state.status[i_c, i_b] = PlanarMaterialStatus.SUCCESS
        state.selected_count[i_c, i_b], state.excluded_count[i_c, i_b] = 0, 0
        if fallback[i_c, i_b] and contacts.status[i_c, i_b] == 0:
            state.queried[i_c, i_b] = True
            qd.atomic_add(state.query_contacts[i_b], 1)
            for i_word in range(state.mask.shape[0]):
                state.mask[i_word, i_c, i_b] = qd.Vector([0])
                state.hits[i_word, i_c, i_b] = qd.Vector([0])
            center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
            first, second = func_force_frame(contacts.force[i_c, i_b])
            n_q = surface.shape.shape[0]
            for i_face in range(surface.quad_vertices.shape[0]):
                separation = qd.max(surface.face_bounds_low[i_face] - center, 0.0)
                separation += qd.max(center - surface.face_bounds_high[i_face], 0.0)
                if separation.norm_sqr() < radius * radius:
                    _closest, remaining, contained = func_contact_section(i_face, center, radius, surface)
                    if remaining > 0.0:
                        positive = False
                        for i_sample in range(qd.select(contained, n_q, 4 * n_q)):
                            if not positive:
                                i_q = surface.positions.shape[0] + i_face * 4 * n_q + i_sample
                                _coordinates, weight, _position, _shape, _face = func_contact_sample(
                                    i_q, i_c, i_b, first, second, contacts, surface
                                )
                                if not (weight >= 0.0 and weight < float("inf")):
                                    state.status[i_c, i_b] = PlanarMaterialStatus.NONFINITE_FOOTPRINT
                                positive = weight > 0.0 and weight < float("inf")
                        if positive:
                            parent = geometry.parent_face[i_face]
                            i_word, bit = parent // 32, parent % 32
                            packed = state.hits[i_word, i_c, i_b]
                            packed[0] = packed[0] | (1 << bit)
                            state.hits[i_word, i_c, i_b] = packed
            i_anchor = contacts.anchor_face[i_c, i_b]
            if i_anchor < 0 or i_anchor >= geometry.parent_face.shape[0]:
                state.status[i_c, i_b] = PlanarMaterialStatus.INVALID_ANCHOR
            else:
                seed = geometry.parent_face[i_anchor]
                i_word, bit = seed // 32, seed % 32
                seed_hits = state.hits[i_word, i_c, i_b]
                if (seed_hits[0] & (1 << bit)) == 0:
                    state.status[i_c, i_b] = PlanarMaterialStatus.UNRESOLVED_SEED
                else:
                    state.mask[i_word, i_c, i_b] = qd.Vector([1 << bit])
                    active = True
                    for _ in range(geometry.parent_neighbor_start.shape[0] - 1):
                        if active:
                            changed = False
                            for parent in range(geometry.parent_neighbor_start.shape[0] - 1):
                                i_parent_word, i_parent_bit = parent // 32, parent % 32
                                hit = state.hits[i_parent_word, i_c, i_b]
                                selected = state.mask[i_parent_word, i_c, i_b]
                                if (hit[0] & (1 << i_parent_bit)) != 0 and (selected[0] & (1 << i_parent_bit)) == 0:
                                    connected = False
                                    for i_n in range(
                                        geometry.parent_neighbor_start[parent],
                                        geometry.parent_neighbor_start[parent + 1],
                                    ):
                                        other = geometry.parent_neighbor_faces[i_n]
                                        i_other_word, i_other_bit = other // 32, other % 32
                                        other_selected = state.mask[i_other_word, i_c, i_b]
                                        connected = connected or (other_selected[0] & (1 << i_other_bit)) != 0
                                    if connected:
                                        selected[0] = selected[0] | (1 << i_parent_bit)
                                        state.mask[i_parent_word, i_c, i_b] = selected
                                        changed = True
                            active = changed
                    for parent in range(geometry.parent_neighbor_start.shape[0] - 1):
                        i_parent_word, i_parent_bit = parent // 32, parent % 32
                        hit = state.hits[i_parent_word, i_c, i_b]
                        selected = state.mask[i_parent_word, i_c, i_b]
                        state.selected_count[i_c, i_b] += gs.qd_int((selected[0] & (1 << i_parent_bit)) != 0)
                        state.excluded_count[i_c, i_b] += gs.qd_int(
                            (hit[0] & (1 << i_parent_bit)) != 0 and (selected[0] & (1 << i_parent_bit)) == 0
                        )
                    if state.excluded_count[i_c, i_b] > 0:
                        direction = -contacts.force[i_c, i_b].normalized()
                        point = contacts.position[i_c, i_b]
                        selected_distance = (contacts.center[i_c, i_b] - point).norm()
                        line_first, line_second = func_force_frame(direction)
                        line_basis = qd.Matrix.rows([line_first, line_second])
                        for i_face in range(surface.quad_vertices.shape[0]):
                            parent = geometry.parent_face[i_face]
                            i_parent_word, i_parent_bit = parent // 32, parent % 32
                            hit_parent = state.hits[i_parent_word, i_c, i_b]
                            selected_parent = state.mask[i_parent_word, i_c, i_b]
                            if (hit_parent[0] & (1 << i_parent_bit)) != 0 and (
                                selected_parent[0] & (1 << i_parent_bit)
                            ) == 0:
                                _shape, hit, outward, _ratio, regular = func_quad_inverse(
                                    surface.quad_coefficients[i_face], point, line_basis, qd.Vector([0.0, 0.0])
                                )
                                tolerance = 64.0 * source_epsilon * (point.norm() + center.norm())
                                tolerance += 4e-12 * surface.quad_scale[i_face]
                                distance = qd.abs((hit - point).dot(direction))
                                if (
                                    regular
                                    and outward.dot(direction) > 1e-12
                                    and qd.abs(distance - selected_distance) <= tolerance
                                ):
                                    state.status[i_c, i_b] = PlanarMaterialStatus.AMBIGUOUS_ANCHOR
            if state.status[i_c, i_b] != PlanarMaterialStatus.SUCCESS:
                contacts.status[i_c, i_b] = qd.select(
                    state.status[i_c, i_b] == PlanarMaterialStatus.AMBIGUOUS_ANCHOR, 9, 10
                )
                qd.atomic_add(state.rejected_contacts[i_b], 1)


@qd.kernel
def kernel_planar_material_reset(envs_idx: qd.types.ndarray(), state: PlanarMaterialState):
    for i_selected in range(envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        state.query_contacts[i_b], state.rejected_contacts[i_b] = 0, 0
    for i_c, i_selected in qd.ndrange(state.queried.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        state.queried[i_c, i_b] = False
        state.status[i_c, i_b] = PlanarMaterialStatus.SUCCESS
        state.selected_count[i_c, i_b], state.excluded_count[i_c, i_b] = 0, 0
