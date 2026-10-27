"""Transfer guarded spherical Hertz pressure through planar shell loading proxies."""

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V, iter_data, of_kind

from .contact import (
    StressContactState,
    func_force_frame,
    func_pressure_initial,
    func_pressure_step,
    kernel_pressure,
    kernel_scatter,
)
from .data import StressInfo, StressState, func_scatter_load
from .planar_geometry import (
    PlanarLoadingInfo,
    build_planar_geometry,
    func_cell_overlap,
    func_face_overlap,
    func_parent_adjacent,
    func_triangle_geometry,
)
from .planar_material import create_planar_material, kernel_planar_material, kernel_planar_material_reset
from .planar_moments import (
    func_planar_affine_minimum,
    func_planar_clip_geometry,
    func_planar_hertz_moments,
    func_planar_hertz_sample,
    func_planar_sample_count,
    func_planar_section,
)
from .scatter import kernel_pressure_moments, kernel_scatter_faces
from .surface import StressSurfaceInfo


class PlanarFallback(IntEnum):
    NONE = 0
    PROFILE = 1
    GEOMETRY = 2
    EMPTY = 3
    MATERIAL = 4
    CAPACITY = 5
    PRESSURE = 6
    WRENCH = 7
    ANCHOR = 8


@dataclass(frozen=True)
class PlanarLoadingState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    side_filter: qd.Tensor = of_kind(DataKind.INFO)  # noqa: RUF009 - typed array metadata
    attempted_loaded_contacts: qd.Tensor = of_kind(DataKind.STATE)  # noqa: RUF009 - typed array metadata
    proxy_contacts: qd.Tensor = of_kind(DataKind.STATE)  # noqa: RUF009 - typed array metadata
    fallback_contacts: qd.Tensor = of_kind(DataKind.STATE)  # noqa: RUF009 - typed array metadata
    analytic_sections: qd.Tensor = of_kind(DataKind.STATE)  # noqa: RUF009 - typed array metadata
    overflow_steps: qd.Tensor = of_kind(DataKind.STATE)  # noqa: RUF009 - typed array metadata
    eligible: qd.Tensor
    fallback: qd.Tensor
    reason: qd.Tensor
    affine: qd.Tensor
    task_start: qd.Tensor
    task_end: qd.Tensor
    parent_start: qd.Tensor
    parent_end: qd.Tensor
    sample_count: qd.Tensor
    analytic_count: qd.Tensor
    fit_evaluations: qd.Tensor
    fallback_count: qd.Tensor
    overflow: qd.Tensor
    task_count: qd.Tensor
    parent_count: qd.Tensor
    source_count: qd.Tensor
    source_slots: qd.Tensor
    tasks: qd.Tensor
    parent_faces: qd.Tensor
    parent_connected: qd.Tensor
    origin: qd.Tensor
    edges: qd.Tensor
    dual: qd.Tensor
    normal: qd.Tensor
    parent_shapes: qd.Tensor
    section_center: qd.Tensor
    plane: qd.Tensor
    polygon: qd.Tensor
    remaining: qd.Tensor
    angles: qd.Tensor
    angle_count: qd.Tensor
    contained: qd.Tensor
    gram: qd.Tensor
    load_moments: qd.Tensor
    node_weights: qd.Tensor


class PlanarLoading:
    """Prepare shared proxy geometry and per-entry guarded finite-pressure workspaces."""

    def __init__(
        self,
        surface: StressSurfaceInfo,
        load_info: StressInfo,
        contacts: StressContactState,
        subdivisions: int,
        geometry: PlanarLoadingInfo | None = None,
        *,
        tasks_per_env: int = 32,
    ):
        if tasks_per_env < 1:
            gs.raise_exception("Planar loading requires a positive per-environment task capacity.")
        self.info = geometry if geometry is not None else build_planar_geometry(surface, load_info, subdivisions)
        n_contacts, n_envs = contacts.valid.shape
        contact_shape, task_shape = (n_contacts, n_envs), (tasks_per_env, n_envs)
        self.state = PlanarLoadingState(
            side_filter=V(dtype=gs.qd_int, shape=contact_shape),
            attempted_loaded_contacts=V(dtype=gs.qd_int, shape=(n_envs,)),
            proxy_contacts=V(dtype=gs.qd_int, shape=(n_envs,)),
            fallback_contacts=V(dtype=gs.qd_int, shape=(n_envs,)),
            analytic_sections=V(dtype=gs.qd_int, shape=(n_envs,)),
            overflow_steps=V(dtype=gs.qd_int, shape=(n_envs,)),
            eligible=V(dtype=gs.qd_bool, shape=contact_shape),
            fallback=V(dtype=gs.qd_bool, shape=contact_shape),
            reason=V(dtype=gs.qd_int, shape=contact_shape),
            affine=V(dtype=gs.qd_bool, shape=contact_shape),
            task_start=V(dtype=gs.qd_int, shape=contact_shape),
            task_end=V(dtype=gs.qd_int, shape=contact_shape),
            parent_start=V(dtype=gs.qd_int, shape=contact_shape),
            parent_end=V(dtype=gs.qd_int, shape=contact_shape),
            sample_count=V(dtype=gs.qd_int, shape=contact_shape),
            analytic_count=V(dtype=gs.qd_int, shape=contact_shape),
            fit_evaluations=V(dtype=gs.qd_int, shape=contact_shape),
            fallback_count=V(dtype=gs.qd_int, shape=(n_envs,)),
            overflow=V(dtype=gs.qd_bool, shape=(n_envs,)),
            task_count=V(dtype=gs.qd_int, shape=(n_envs,)),
            parent_count=V(dtype=gs.qd_int, shape=(n_envs,)),
            source_count=V(dtype=gs.qd_int, shape=(n_envs,)),
            source_slots=V(dtype=gs.qd_int, shape=contact_shape),
            tasks=V_VEC(3, dtype=gs.qd_int, shape=task_shape),
            parent_faces=V(dtype=gs.qd_int, shape=task_shape),
            parent_connected=V(dtype=gs.qd_bool, shape=task_shape),
            origin=V_VEC(3, dtype=gs.qd_float, shape=task_shape),
            edges=V_MAT(2, 3, dtype=gs.qd_float, shape=task_shape),
            dual=V_MAT(2, 3, dtype=gs.qd_float, shape=task_shape),
            normal=V_VEC(3, dtype=gs.qd_float, shape=task_shape),
            parent_shapes=V_MAT(3, 4, dtype=gs.qd_float, shape=task_shape),
            section_center=V_VEC(3, dtype=gs.qd_float, shape=task_shape),
            plane=V_MAT(2, 3, dtype=gs.qd_float, shape=task_shape),
            polygon=V_MAT(3, 2, dtype=gs.qd_float, shape=task_shape),
            remaining=V(dtype=gs.qd_float, shape=task_shape),
            angles=V_VEC(11, dtype=gs.qd_float, shape=task_shape),
            angle_count=V(dtype=gs.qd_int, shape=task_shape),
            contained=V(dtype=gs.qd_bool, shape=task_shape),
            gram=V_MAT(3, 3, dtype=gs.qd_float, shape=task_shape),
            load_moments=V_MAT(4, 3, dtype=gs.qd_float, shape=task_shape),
            node_weights=V_VEC(4, dtype=gs.qd_float, shape=task_shape),
        )
        self.state.side_filter.fill(0)
        self.state.attempted_loaded_contacts.fill(0)
        self.state.proxy_contacts.fill(0)
        self.state.fallback_contacts.fill(0)
        self.state.analytic_sections.fill(0)
        self.state.overflow_steps.fill(0)
        self.fallback_contacts = replace(contacts, valid=self.state.fallback)
        self.material_state = create_planar_material(self.info.parent_neighbor_start.shape[0] - 1, n_contacts, n_envs)
        self.material_surface = replace(surface, face_mask=self.material_state.mask)
        self.source_epsilon = float(np.finfo(gs.np_float).eps)
        self.contacts = contacts

    def scatter(self, stress_state, load_info, surface_info, options, scatter_workspace, reuse_contact_moments: bool):
        """Fit current proxy loads, recover unsupported contacts on the original route and add accepted contributions."""
        if options.cooperative_pressure and gs.backend == gs.cuda:
            kernel_planar_query_warp(self.contacts, self.info, self.state)
            kernel_planar_prepare_moments(self.contacts, self.state)
            kernel_planar_moments_warp(self.contacts, self.state, surface_info)
        else:
            kernel_planar_query(self.contacts, self.info, self.state)
            kernel_planar_moments(self.contacts, self.state, surface_info)
        kernel_planar_fit(self.contacts, self.state, surface_info)
        kernel_planar_validate(self.contacts, self.state, load_info, surface_info)
        kernel_planar_material(
            self.source_epsilon, self.contacts, self.state.fallback, surface_info, self.info, self.material_state
        )
        if reuse_contact_moments:
            kernel_pressure_moments(
                self.fallback_contacts,
                stress_state,
                load_info,
                self.material_surface,
                scatter_workspace,
                options.cached_face_bounds,
            )
        else:
            kernel_pressure(self.fallback_contacts, self.material_surface, options.cooperative_pressure)
        if scatter_workspace.tasks.shape[0] and options.cooperative_scatter:
            kernel_scatter_faces(
                self.fallback_contacts,
                stress_state,
                load_info,
                self.material_surface,
                scatter_workspace,
                options.cached_face_bounds,
                reuse_contact_moments,
            )
        else:
            kernel_scatter(
                self.fallback_contacts,
                stress_state,
                load_info,
                self.material_surface,
                options.cooperative_scatter,
                options.cached_face_bounds,
            )
        kernel_planar_scatter(self.contacts, self.state, load_info, stress_state)

    def data(self, prefix: str, *, geometry: bool = True):
        if geometry:
            yield from iter_data(self.info, f"{prefix}.geometry")
        yield from iter_data(self.state, f"{prefix}.state")
        yield from iter_data(self.material_state, f"{prefix}.material")

    def reset(self, envs_idx):
        """Clear the selected environments' loading state and cumulative route counts."""
        kernel_planar_reset(envs_idx, self.state)
        kernel_planar_material_reset(envs_idx, self.material_state)


@qd.kernel
def kernel_planar_reset(envs_idx: qd.types.ndarray(), state: PlanarLoadingState):
    for i_selected in range(envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        state.attempted_loaded_contacts[i_b], state.proxy_contacts[i_b], state.fallback_contacts[i_b] = 0, 0, 0
        state.analytic_sections[i_b], state.overflow_steps[i_b] = 0, 0
        state.fallback_count[i_b], state.task_count[i_b], state.parent_count[i_b], state.source_count[i_b] = 0, 0, 0, 0
        state.overflow[i_b] = False
    for i_c, i_selected in qd.ndrange(state.eligible.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        state.eligible[i_c, i_b], state.fallback[i_c, i_b], state.affine[i_c, i_b] = False, False, False
        state.reason[i_c, i_b] = PlanarFallback.NONE
        state.task_start[i_c, i_b], state.task_end[i_c, i_b] = 0, 0
        state.parent_start[i_c, i_b], state.parent_end[i_c, i_b] = 0, 0
        state.sample_count[i_c, i_b], state.analytic_count[i_c, i_b], state.fit_evaluations[i_c, i_b] = 0, 0, 0


@qd.kernel
def kernel_planar_query(contacts: StressContactState, info: PlanarLoadingInfo, state: PlanarLoadingState):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        state.eligible[i_c, i_b] = False
        state.fallback[i_c, i_b] = contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 1e-30
        state.reason[i_c, i_b] = PlanarFallback.NONE
        state.affine[i_c, i_b] = False
        state.task_start[i_c, i_b], state.task_end[i_c, i_b] = 0, 0
        state.parent_start[i_c, i_b], state.parent_end[i_c, i_b] = 0, 0
        state.sample_count[i_c, i_b], state.analytic_count[i_c, i_b], state.fit_evaluations[i_c, i_b] = 0, 0, 0
    for i_b in range(contacts.valid.shape[1]):
        state.task_count[i_b], state.parent_count[i_b], state.fallback_count[i_b], state.source_count[i_b] = 0, 0, 0, 0
        state.overflow[i_b] = False
        for i_c in range(contacts.valid.shape[0]):
            state.task_start[i_c, i_b] = state.task_count[i_b]
            state.parent_start[i_c, i_b] = state.parent_count[i_b]
            if state.fallback[i_c, i_b]:
                i_source = state.source_count[i_b]
                state.source_slots[i_source, i_b] = i_c
                state.source_count[i_b] += 1
                if contacts.status[i_c, i_b] != 0:
                    state.reason[i_c, i_b] = PlanarFallback.ANCHOR
                elif not contacts.hertz_profile[None]:
                    state.reason[i_c, i_b] = PlanarFallback.PROFILE
                elif not state.overflow[i_b]:
                    state.eligible[i_c, i_b] = True
                    center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
                    for i_f in range(info.face_valid.shape[0]):
                        side = state.side_filter[i_c, i_b]
                        admitted_side = side == 0 or side == info.side[i_f]
                        if state.eligible[i_c, i_b] and admitted_side and func_face_overlap(i_f, center, radius, info):
                            if not info.face_valid[i_f] or info.normal_error[i_f] > 0.01 * radius:
                                state.eligible[i_c, i_b] = False
                                state.reason[i_c, i_b] = PlanarFallback.GEOMETRY
                            else:
                                hit = False
                                for i_cell in range(info.cell_nodes.shape[0]):
                                    if not state.overflow[i_b] and func_cell_overlap(i_f, i_cell, center, radius, info):
                                        for i_half in range(2):
                                            i_triangle = 2 * i_cell + i_half
                                            vertices, dual, normal, parent_shapes = func_triangle_geometry(
                                                i_f, i_triangle, info
                                            )
                                            edges = qd.Matrix.rows(
                                                [vertices[1, :] - vertices[0, :], vertices[2, :] - vertices[0, :]]
                                            )
                                            _section, _remaining, _contained, intersects = func_planar_section(
                                                vertices[0, :], edges, dual, normal, center, radius
                                            )
                                            if intersects:
                                                i_t = state.task_count[i_b]
                                                if i_t < state.tasks.shape[0]:
                                                    state.tasks[i_t, i_b] = qd.Vector([i_f, i_triangle, i_c])
                                                    state.origin[i_t, i_b] = vertices[0, :]
                                                    state.edges[i_t, i_b] = edges
                                                    state.dual[i_t, i_b] = dual
                                                    state.normal[i_t, i_b] = normal
                                                    state.parent_shapes[i_t, i_b] = parent_shapes
                                                    state.task_count[i_b] += 1
                                                    hit = True
                                                else:
                                                    state.overflow[i_b] = True
                                if hit:
                                    parent = info.parent_face[i_f]
                                    present = False
                                    for i_p in range(state.parent_start[i_c, i_b], state.parent_count[i_b]):
                                        present = present or state.parent_faces[i_p, i_b] == parent
                                    if not present:
                                        i_p = state.parent_count[i_b]
                                        if i_p < state.parent_faces.shape[0]:
                                            state.parent_faces[i_p, i_b] = parent
                                            state.parent_count[i_b] += 1
                                        else:
                                            state.overflow[i_b] = True
                    if state.eligible[i_c, i_b] and state.task_count[i_b] == state.task_start[i_c, i_b]:
                        state.eligible[i_c, i_b] = False
                        state.reason[i_c, i_b] = PlanarFallback.EMPTY
            state.task_end[i_c, i_b] = state.task_count[i_b]
            state.parent_end[i_c, i_b] = state.parent_count[i_b]
        for i_c in range(contacts.valid.shape[0]):
            func_planar_query_component(i_c, i_b, contacts, info, state)


@qd.func
def func_planar_query_component(
    i_c: int, i_b: int, contacts: StressContactState, info: PlanarLoadingInfo, state: PlanarLoadingState
):
    """Apply material connectivity and complete-environment overflow admission to one source."""
    if state.overflow[i_b] and state.fallback[i_c, i_b]:
        state.eligible[i_c, i_b] = False
        state.reason[i_c, i_b] = PlanarFallback.CAPACITY
    if state.eligible[i_c, i_b]:
        i_anchor = contacts.anchor_face[i_c, i_b]
        connected = i_anchor >= 0 and i_anchor < info.parent_face.shape[0]
        seed = -1
        if connected:
            seed = info.parent_face[i_anchor]
        for i_p in range(state.parent_start[i_c, i_b], state.parent_end[i_c, i_b]):
            state.parent_connected[i_p, i_b] = state.parent_faces[i_p, i_b] == seed
        active = True
        for _ in range(state.parent_end[i_c, i_b] - state.parent_start[i_c, i_b]):
            if active:
                changed = False
                for i_p in range(state.parent_start[i_c, i_b], state.parent_end[i_c, i_b]):
                    if not state.parent_connected[i_p, i_b]:
                        parent = state.parent_faces[i_p, i_b]
                        for i_other in range(state.parent_start[i_c, i_b], state.parent_end[i_c, i_b]):
                            if state.parent_connected[i_other, i_b]:
                                other = state.parent_faces[i_other, i_b]
                                if func_parent_adjacent(parent, other, info):
                                    state.parent_connected[i_p, i_b], changed = True, True
                active = changed
        for i_p in range(state.parent_start[i_c, i_b], state.parent_end[i_c, i_b]):
            connected = connected and state.parent_connected[i_p, i_b]
        if not connected:
            state.eligible[i_c, i_b] = False
            state.reason[i_c, i_b] = PlanarFallback.MATERIAL
        state.fallback[i_c, i_b] = not state.eligible[i_c, i_b]


@qd.kernel
def kernel_planar_query_warp(contacts: StressContactState, info: PlanarLoadingInfo, state: PlanarLoadingState):
    for i_b in range(contacts.valid.shape[1]):
        state.task_count[i_b], state.parent_count[i_b], state.fallback_count[i_b], state.source_count[i_b] = 0, 0, 0, 0
        state.overflow[i_b] = False
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        state.eligible[i_c, i_b] = False
        state.fallback[i_c, i_b] = contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 1e-30
        state.reason[i_c, i_b] = PlanarFallback.NONE
        state.affine[i_c, i_b] = False
        state.task_start[i_c, i_b], state.task_end[i_c, i_b] = 0, 0
        state.parent_start[i_c, i_b], state.parent_end[i_c, i_b] = 0, 0
        state.sample_count[i_c, i_b], state.analytic_count[i_c, i_b], state.fit_evaluations[i_c, i_b] = 0, 0, 0
        if state.fallback[i_c, i_b]:
            i_source = qd.atomic_add(state.source_count[i_b], 1)
            state.source_slots[i_source, i_b] = i_c
    # One warp owns each environment's pool, keeping every contact's task range contiguous.
    qd.loop_config(block_dim=32)
    for i_thread in range(contacts.valid.shape[1] * 32):
        lane, i_b = i_thread % 32, i_thread // 32
        sh_faces = qd.simt.block.SharedArray((32,), gs.qd_int)
        sh_count = qd.simt.block.SharedArray((1,), gs.qd_int)
        for i_source in range(state.source_count[i_b]):
            i_c = state.source_slots[i_source, i_b]
            if lane == 0:
                state.task_start[i_c, i_b] = state.task_count[i_b]
                state.parent_start[i_c, i_b] = state.parent_count[i_b]
                if contacts.status[i_c, i_b] != 0:
                    state.reason[i_c, i_b] = PlanarFallback.ANCHOR
                elif not contacts.hertz_profile[None]:
                    state.reason[i_c, i_b] = PlanarFallback.PROFILE
                elif not state.overflow[i_b]:
                    state.eligible[i_c, i_b] = True
            qd.simt.block.sync()
            center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
            side = state.side_filter[i_c, i_b]
            if state.eligible[i_c, i_b]:
                for i_face_chunk in range((info.face_valid.shape[0] + 31) // 32):
                    if state.eligible[i_c, i_b] and not state.overflow[i_b]:
                        if lane == 0:
                            sh_count[0] = 0
                        qd.simt.block.sync()
                        i_face = i_face_chunk * 32 + lane
                        invalid = 0
                        if i_face < info.face_valid.shape[0]:
                            admitted_side = side == 0 or side == info.side[i_face]
                            if admitted_side and func_face_overlap(i_face, center, radius, info):
                                if not info.face_valid[i_face] or info.normal_error[i_face] > 0.01 * radius:
                                    invalid = 1
                                else:
                                    i_candidate = qd.atomic_add(sh_count[0], 1)
                                    sh_faces[i_candidate] = i_face
                        invalid = qd.simt.subgroup.reduce_all_add(invalid)
                        if lane == 0 and invalid > 0:
                            state.eligible[i_c, i_b] = False
                            state.reason[i_c, i_b] = PlanarFallback.GEOMETRY
                        qd.simt.block.sync()
                        if state.eligible[i_c, i_b]:
                            for i_candidate in range(sh_count[0]):
                                if not state.overflow[i_b]:
                                    i_face = sh_faces[i_candidate]
                                    found, overflowed = 0, 0
                                    for i_cell_chunk in range((info.cell_nodes.shape[0] + 31) // 32):
                                        i_cell = i_cell_chunk * 32 + lane
                                        if i_cell < info.cell_nodes.shape[0]:  # noqa: SIM102 - bound the geometry index first
                                            if func_cell_overlap(i_face, i_cell, center, radius, info):
                                                for i_half in range(2):
                                                    i_triangle = 2 * i_cell + i_half
                                                    vertices, dual, normal, parent_shapes = func_triangle_geometry(
                                                        i_face, i_triangle, info
                                                    )
                                                    edges = qd.Matrix.rows(
                                                        [
                                                            vertices[1, :] - vertices[0, :],
                                                            vertices[2, :] - vertices[0, :],
                                                        ]
                                                    )
                                                    _section, _remaining, _contained, intersects = func_planar_section(
                                                        vertices[0, :], edges, dual, normal, center, radius
                                                    )
                                                    if intersects:
                                                        i_t = qd.atomic_add(state.task_count[i_b], 1)
                                                        if i_t < state.tasks.shape[0]:
                                                            state.tasks[i_t, i_b] = qd.Vector([i_face, i_triangle, i_c])
                                                            state.origin[i_t, i_b] = vertices[0, :]
                                                            state.edges[i_t, i_b] = edges
                                                            state.dual[i_t, i_b] = dual
                                                            state.normal[i_t, i_b] = normal
                                                            state.parent_shapes[i_t, i_b] = parent_shapes
                                                        else:
                                                            overflowed = 1
                                                        found += 1
                                    found = qd.simt.subgroup.reduce_all_add(found)
                                    overflowed = qd.simt.subgroup.reduce_all_add(overflowed)
                                    if lane == 0:
                                        if overflowed > 0:
                                            state.overflow[i_b] = True
                                            state.task_count[i_b] = state.tasks.shape[0]
                                        if found > 0 and not state.overflow[i_b]:
                                            i_p = state.parent_count[i_b]
                                            if i_p < state.parent_faces.shape[0]:
                                                state.parent_faces[i_p, i_b] = info.parent_face[i_face]
                                                state.parent_count[i_b] += 1
                                            else:
                                                state.overflow[i_b] = True
                                    qd.simt.block.sync()
                        qd.simt.block.sync()
            qd.simt.block.sync()
            if lane == 0:
                if state.eligible[i_c, i_b] and state.task_count[i_b] == state.task_start[i_c, i_b]:
                    state.eligible[i_c, i_b] = False
                    state.reason[i_c, i_b] = PlanarFallback.EMPTY
                state.task_end[i_c, i_b] = state.task_count[i_b]
                state.parent_end[i_c, i_b] = state.parent_count[i_b]
            qd.simt.block.sync()
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        func_planar_query_component(i_c, i_b, contacts, info, state)


@qd.func
def func_planar_task_sample(
    i_t: int,
    i_b: int,
    i_sample: int,
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    contacts: StressContactState,
    state: PlanarLoadingState,
    surface: StressSurfaceInfo,
):
    task = state.tasks[i_t, i_b]
    i_c = task[2]
    return func_planar_hertz_sample(
        i_sample,
        state.origin[i_t, i_b],
        state.dual[i_t, i_b],
        state.parent_shapes[i_t, i_b],
        state.section_center[i_t, i_b],
        state.plane[i_t, i_b],
        state.polygon[i_t, i_b],
        state.remaining[i_t, i_b],
        state.angles[i_t, i_b],
        state.angle_count[i_t, i_b],
        contacts.position[i_c, i_b],
        contacts.radius[i_c, i_b],
        first,
        second,
        surface.gauss,
        surface.gauss_weights,
        surface.gauss,
        surface.gauss_weights,
        state.contained[i_t, i_b],
    )


@qd.kernel
def kernel_planar_moments(contacts: StressContactState, state: PlanarLoadingState, surface: StressSurfaceInfo):
    for i_t, i_b in qd.ndrange(state.tasks.shape[0], state.tasks.shape[1]):
        if i_t < state.task_count[i_b]:
            task = state.tasks[i_t, i_b]
            i_c = task[2]
            if state.eligible[i_c, i_b]:
                first, second = func_force_frame(contacts.force[i_c, i_b])
                origin, edges = state.origin[i_t, i_b], state.edges[i_t, i_b]
                dual, normal, parent_shapes = (
                    state.dual[i_t, i_b],
                    state.normal[i_t, i_b],
                    state.parent_shapes[i_t, i_b],
                )
                center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
                section, plane, polygon, remaining, angles, count, contained = func_planar_clip_geometry(
                    origin, edges, normal, center, radius
                )
                state.section_center[i_t, i_b], state.plane[i_t, i_b], state.polygon[i_t, i_b] = section, plane, polygon
                state.remaining[i_t, i_b], state.angles[i_t, i_b], state.angle_count[i_t, i_b] = (
                    remaining,
                    angles,
                    count,
                )
                state.contained[i_t, i_b] = contained
                gram, load = qd.Matrix.zero(gs.qd_float, 3, 3), qd.Matrix.zero(gs.qd_float, 4, 3)
                samples = 0
                if contained:
                    gram, load, _mass, _contained = func_planar_hertz_moments(
                        origin,
                        edges,
                        dual,
                        normal,
                        parent_shapes,
                        center,
                        contacts.position[i_c, i_b],
                        radius,
                        first,
                        second,
                    )
                    qd.atomic_add(state.analytic_count[i_c, i_b], 1)
                else:
                    for i_sample in range(
                        func_planar_sample_count(count, surface.gauss.shape[0], surface.gauss.shape[0], False)
                    ):
                        _position, shape, weight, coordinates = func_planar_task_sample(
                            i_t, i_b, i_sample, first, second, contacts, state, surface
                        )
                        gram += weight * coordinates.outer_product(coordinates)
                        load += shape.outer_product(weight * coordinates)
                        samples += gs.qd_int(weight > 0.0)
                state.gram[i_t, i_b], state.load_moments[i_t, i_b] = gram, load
                qd.atomic_add(state.sample_count[i_c, i_b], samples)


@qd.kernel
def kernel_planar_prepare_moments(contacts: StressContactState, state: PlanarLoadingState):
    """Publish each task's clipping geometry and contained analytic moments before warp integration."""
    for i_t, i_b in qd.ndrange(state.tasks.shape[0], state.tasks.shape[1]):
        if i_t < state.task_count[i_b]:
            task = state.tasks[i_t, i_b]
            i_c = task[2]
            if state.eligible[i_c, i_b]:
                origin, edges = state.origin[i_t, i_b], state.edges[i_t, i_b]
                normal = state.normal[i_t, i_b]
                center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
                section, plane, polygon, remaining, angles, count, contained = func_planar_clip_geometry(
                    origin, edges, normal, center, radius
                )
                state.section_center[i_t, i_b], state.plane[i_t, i_b], state.polygon[i_t, i_b] = section, plane, polygon
                state.remaining[i_t, i_b], state.angles[i_t, i_b], state.angle_count[i_t, i_b] = (
                    remaining,
                    angles,
                    count,
                )
                state.contained[i_t, i_b] = contained
                if contained:
                    first, second = func_force_frame(contacts.force[i_c, i_b])
                    gram, load, _mass, _contained = func_planar_hertz_moments(
                        origin,
                        edges,
                        state.dual[i_t, i_b],
                        normal,
                        state.parent_shapes[i_t, i_b],
                        center,
                        contacts.position[i_c, i_b],
                        radius,
                        first,
                        second,
                    )
                    state.gram[i_t, i_b], state.load_moments[i_t, i_b] = gram, load
                    qd.atomic_add(state.analytic_count[i_c, i_b], 1)


@qd.kernel
def kernel_planar_moments_warp(contacts: StressContactState, state: PlanarLoadingState, surface: StressSurfaceInfo):
    qd.loop_config(block_dim=128)
    for i_thread in range(state.tasks.shape[0] * state.tasks.shape[1] * 32):
        lane = i_thread % 32
        i_t, i_b = (i_thread // 32) // state.tasks.shape[1], (i_thread // 32) % state.tasks.shape[1]
        if i_t < state.task_count[i_b]:
            task = state.tasks[i_t, i_b]
            i_c = task[2]
            if state.eligible[i_c, i_b]:
                contained = state.contained[i_t, i_b]
                if not contained:
                    # The preceding prepare kernel publishes the cache before any lane reads it.
                    first, second = func_force_frame(contacts.force[i_c, i_b])
                    origin, dual = state.origin[i_t, i_b], state.dual[i_t, i_b]
                    parent_shapes = state.parent_shapes[i_t, i_b]
                    section, plane, polygon = (
                        state.section_center[i_t, i_b],
                        state.plane[i_t, i_b],
                        state.polygon[i_t, i_b],
                    )
                    remaining, angles, count = (
                        state.remaining[i_t, i_b],
                        state.angles[i_t, i_b],
                        state.angle_count[i_t, i_b],
                    )
                    radius = contacts.radius[i_c, i_b]
                    gram, load = qd.Matrix.zero(gs.qd_float, 3, 3), qd.Matrix.zero(gs.qd_float, 4, 3)
                    samples = 0
                    n_samples = func_planar_sample_count(count, surface.gauss.shape[0], surface.gauss.shape[0], False)
                    for chunk in range((n_samples + 31) // 32):
                        i_sample = chunk * 32 + lane
                        if i_sample < n_samples:
                            _position, shape, weight, coordinates = func_planar_hertz_sample(
                                i_sample,
                                origin,
                                dual,
                                parent_shapes,
                                section,
                                plane,
                                polygon,
                                remaining,
                                angles,
                                count,
                                contacts.position[i_c, i_b],
                                radius,
                                first,
                                second,
                                surface.gauss,
                                surface.gauss_weights,
                                surface.gauss,
                                surface.gauss_weights,
                                contained,
                            )
                            gram += weight * coordinates.outer_product(coordinates)
                            load += shape.outer_product(weight * coordinates)
                            samples += gs.qd_int(weight > 0.0)
                    for a, b in qd.static(qd.ndrange(3, 3)):
                        gram[a, b] = qd.simt.subgroup.reduce_all_add(gram[a, b])
                    for a, b in qd.static(qd.ndrange(4, 3)):
                        load[a, b] = qd.simt.subgroup.reduce_all_add(load[a, b])
                    samples = qd.simt.subgroup.reduce_all_add(samples)
                    if lane == 0:
                        state.gram[i_t, i_b], state.load_moments[i_t, i_b] = gram, load
                        qd.atomic_add(state.sample_count[i_c, i_b], samples)


@qd.func
def func_planar_pressure_evaluate(
    i_c: int,
    i_b: int,
    coefficient: qd.types.vector(3),
    normalizer: float,
    derivatives: bool,
    contacts: StressContactState,
    state: PlanarLoadingState,
    surface: StressSurfaceInfo,
):
    first, second = func_force_frame(contacts.force[i_c, i_b])
    objective = coefficient[0]
    gradient = qd.Vector([-1.0, 0.0, 0.0])
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i_t in range(state.task_start[i_c, i_b], state.task_end[i_c, i_b]):
        count = func_planar_sample_count(
            state.angle_count[i_t, i_b], surface.gauss.shape[0], surface.gauss.shape[0], state.contained[i_t, i_b]
        )
        for i_sample in range(count):
            _position, _shape, weight, coordinates = func_planar_task_sample(
                i_t, i_b, i_sample, first, second, contacts, state, surface
            )
            weight /= normalizer
            pressure = qd.max(0.0, coordinates.dot(coefficient))
            objective -= 0.5 * weight * pressure * pressure
            if derivatives:
                gradient += weight * pressure * coordinates
                if pressure > 0.0:
                    hessian += weight * coordinates.outer_product(coordinates)
    return objective, gradient, hessian


@qd.kernel
def kernel_planar_fit(contacts: StressContactState, state: PlanarLoadingState, surface: StressSurfaceInfo):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if state.eligible[i_c, i_b]:
            gram = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i_t in range(state.task_start[i_c, i_b], state.task_end[i_c, i_b]):
                gram += state.gram[i_t, i_b]
            coefficient, weight_sum, status = func_pressure_initial(gram)
            accepted = False
            if status != 3:
                residual = (gram / weight_sum) @ (coefficient * weight_sum) - qd.Vector([1.0, 0.0, 0.0])
                accepted = residual.norm() <= 1e-9
            first, second = func_force_frame(contacts.force[i_c, i_b])
            if accepted:
                for i_t in range(state.task_start[i_c, i_b], state.task_end[i_c, i_b]):
                    minimum = func_planar_affine_minimum(
                        state.section_center[i_t, i_b],
                        state.normal[i_t, i_b],
                        contacts.position[i_c, i_b],
                        state.remaining[i_t, i_b],
                        contacts.radius[i_c, i_b],
                        first,
                        second,
                        coefficient * weight_sum,
                    )
                    accepted = accepted and minimum >= -2e-12
            state.affine[i_c, i_b] = accepted
            if status != 3 and not accepted:
                candidate = coefficient * weight_sum
                line = False
                iterations, line_evaluations = 0, 0
                fraction, reference_objective, slope = gs.qd_float(1.0), gs.qd_float(0.0), gs.qd_float(0.0)
                step = qd.Vector.zero(gs.qd_float, 3)
                active = True
                for _ in range(80 * 41):
                    active = active and not accepted and (line or iterations < 80)
                    if active:
                        current = candidate
                        if line:
                            current = candidate - fraction * step
                        objective, gradient, hessian = func_planar_pressure_evaluate(
                            i_c, i_b, current, weight_sum, not line, contacts, state, surface
                        )
                        if not line:
                            iterations += 1
                            accepted = gradient.norm() <= 1e-9
                            proposed, nonsingular = func_pressure_step(hessian, gradient)
                            if not accepted:
                                if nonsingular:
                                    step, slope, reference_objective = proposed, gradient.dot(proposed), objective
                                    fraction, line, line_evaluations = 1.0, True, 0
                                else:
                                    active = False
                        else:
                            line_evaluations += 1
                            if objective >= reference_objective + 1e-4 * fraction * slope - 1e-15:
                                candidate, line = current, False
                            else:
                                fraction *= 0.5
                                if line_evaluations == 40:
                                    active = False
                coefficient = candidate / weight_sum
                state.fit_evaluations[i_c, i_b] = iterations
            if accepted:
                contacts.coefficient[i_c, i_b], contacts.weight_sum[i_c, i_b] = coefficient, weight_sum
            else:
                state.eligible[i_c, i_b], state.fallback[i_c, i_b] = False, True
                state.reason[i_c, i_b] = PlanarFallback.PRESSURE


@qd.kernel
def kernel_planar_validate(
    contacts: StressContactState, state: PlanarLoadingState, load_info: StressInfo, surface: StressSurfaceInfo
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if state.eligible[i_c, i_b]:
            force, original = contacts.force[i_c, i_b], contacts.position[i_c, i_b]
            first, second = func_force_frame(force)
            total, moment = gs.qd_float(0.0), qd.Vector.zero(gs.qd_float, 3)
            for i_t in range(state.task_start[i_c, i_b], state.task_end[i_c, i_b]):
                weights = qd.Vector.zero(gs.qd_float, 4)
                coefficient = contacts.coefficient[i_c, i_b]
                if state.affine[i_c, i_b]:
                    weights = state.load_moments[i_t, i_b] @ coefficient
                else:
                    count = func_planar_sample_count(
                        state.angle_count[i_t, i_b],
                        surface.gauss.shape[0],
                        surface.gauss.shape[0],
                        state.contained[i_t, i_b],
                    )
                    for i_sample in range(count):
                        _position, shape, weight, coordinates = func_planar_task_sample(
                            i_t, i_b, i_sample, first, second, contacts, state, surface
                        )
                        weights += shape * (weight * qd.max(0.0, coordinates.dot(coefficient)))
                state.node_weights[i_t, i_b] = weights
                task = state.tasks[i_t, i_b]
                i_face = task[0]
                for i_local in qd.static(range(4)):
                    i_node = load_info.surface_nodes[i_face, i_local]
                    moment += (load_info.vertices[i_node] - original).cross(weights[i_local] * force)
                    total += weights[i_local]
            force_error = qd.abs(total - 1.0) * force.norm()
            moment_error = moment.norm()
            accepted = (
                force_error <= 1e-8 * force.norm() and moment_error <= 1e-8 * force.norm() * contacts.radius[i_c, i_b]
            )
            if accepted:
                contacts.force_error[i_c, i_b], contacts.moment_error[i_c, i_b] = force_error, moment_error
                contacts.candidate_count[i_c, i_b] = state.sample_count[i_c, i_b] + state.analytic_count[i_c, i_b]
                contacts.evaluations[i_c, i_b] = state.fit_evaluations[i_c, i_b]
            else:
                state.eligible[i_c, i_b], state.fallback[i_c, i_b] = False, True
                state.reason[i_c, i_b] = PlanarFallback.WRENCH
        if state.fallback[i_c, i_b]:
            contacts.candidate_count[i_c, i_b], contacts.evaluations[i_c, i_b] = 0, 0
            contacts.force_error[i_c, i_b], contacts.moment_error[i_c, i_b] = 0.0, 0.0
            qd.atomic_add(state.fallback_count[i_b], 1)
        if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 1e-30:
            qd.atomic_add(state.attempted_loaded_contacts[i_b], 1)
            qd.atomic_add(state.proxy_contacts[i_b], gs.qd_int(state.eligible[i_c, i_b]))
            qd.atomic_add(state.fallback_contacts[i_b], gs.qd_int(state.fallback[i_c, i_b]))
            qd.atomic_add(state.analytic_sections[i_b], state.analytic_count[i_c, i_b])
    for i_b in range(state.overflow_steps.shape[0]):
        state.overflow_steps[i_b] += gs.qd_int(state.overflow[i_b])


@qd.kernel
def kernel_planar_scatter(
    contacts: StressContactState, state: PlanarLoadingState, load_info: StressInfo, stress_state: StressState
):
    for i_t, i_b in qd.ndrange(state.tasks.shape[0], state.tasks.shape[1]):
        if i_t < state.task_count[i_b]:
            task = state.tasks[i_t, i_b]
            i_face, i_c = task[0], task[2]
            if state.eligible[i_c, i_b]:
                force, weights = contacts.force[i_c, i_b], state.node_weights[i_t, i_b]
                for i_local in qd.static(range(4)):
                    i_node = load_info.surface_nodes[i_face, i_local]
                    func_scatter_load(i_node, i_b, weights[i_local] * force, stress_state, load_info)
