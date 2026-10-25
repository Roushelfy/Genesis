"""Rigid solver lifecycle for optional native stress links."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.base_solver import StateChange, Subscriber
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, DataItem, DataKind, iter_data
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    tensor_to_array,
)

from .association import kernel_associate
from .atlas import StressAtlas
from .automatic_radius import StressRadiusInfo, StressRadiusStatus
from .curvature import StressCurvature
from .contact import StressContactState, create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from .data import StressState
from .history import StressHistory
from .intrinsic import IntrinsicContacts
from .lifecycle import func_accept, func_begin_step
from .model import StressModel
from .pipeline import kernel_pipeline
from .post import create_post_kernel
from .provenance import kernel_capture_rigid_sources
from .resolved import ResolvedContacts
from .response import StressResponse
from .scatter import StressScatterWorkspace, create_scatter_workspace, kernel_pressure_moments, kernel_scatter_faces
from .shell_model import ShellModel
from .surface import StressSurface

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


@dataclass
class StressLink:
    link: "RigidLink"
    model: StressModel
    surface: StressSurface
    state: StressState
    contacts: StressContactState
    omega: qd.Tensor
    history: StressHistory | None
    output_mode: str
    scatter: StressScatterWorkspace
    options: RigidStressOptions
    reuse_contact_wrench: bool
    packed_block_size: int
    fused_pipeline: bool
    coalesced_residual: bool
    reuse_contact_moments: bool
    intrinsic: IntrinsicContacts | None = None
    resolved: ResolvedContacts | None = None
    requested_step: int = -1
    automatic_radius: StressRadiusInfo | None = None
    atlas: StressAtlas | None = None
    response: StressResponse | None = None


class RigidStressRecovery:
    def __init__(self, solver: "RigidSolver"):
        self.solver = solver
        self.links: list[StressLink] = []
        self.curvature: StressCurvature | None = None
        self.source_epsilon = float(np.finfo(gs.np_float).eps)
        for entity in solver.entities:
            for link in entity.links:
                options = link.stress_options
                if options is not None:
                    if options.atlas_regions and (
                        options.contact_model != "pressure" or options.load_transfer != "patch"
                    ):
                        gs.raise_exception("A response atlas requires DPR finite-pressure patch loading.")
                    if options.backend == "shell" and (
                        options.contact_model != "pressure" or options.load_transfer != "patch"
                    ):
                        gs.raise_exception("The shell backend requires DPR finite-pressure patch loading.")
                    if (options.radius_model != "manual" or options.pressure_profile != "dpr") and (
                        options.contact_model != "pressure" or options.load_transfer != "patch"
                    ):
                        gs.raise_exception("Automatic radius and Hertz profiles require DPR patch transfer.")
                    if options.load_transfer == "resolved" and options.contact_model != "pressure":
                        gs.raise_exception("Resolved boundary transfer requires pressure-mode stress recovery.")
                    if options.history_size and options.method == "pcg":
                        gs.raise_exception("The native load history requires a direct or inverse stress method.")
                    # Equal geometry, constitutive parameters and mass share physical operators and factors.
                    model = None
                    surface = None
                    quadrature = 1 if options.load_transfer == "resolved" else options.quadrature
                    for existing in self.links:
                        previous = existing.model.options
                        if (
                            Path(previous.mesh).resolve() == Path(options.mesh).resolve()
                            and previous.backend == options.backend
                            and previous.shell_thickness == options.shell_thickness
                            and previous.shell_drilling_factor == options.shell_drilling_factor
                            and previous.shell_gauge == options.shell_gauge
                            and previous.young == options.young
                            and previous.poisson == options.poisson
                            and previous.density == options.density
                            and previous.method == options.method
                            and previous.factor_scaling == options.factor_scaling
                            and previous.factor_ordering == options.factor_ordering
                            and previous.inverse_precision == options.inverse_precision
                            and previous.inverse_max_bytes == options.inverse_max_bytes
                            and previous.surface_inverse == options.surface_inverse
                            and previous.stress_response == options.stress_response
                            and previous.response_precision == options.response_precision
                            and previous.response_accumulation == options.response_accumulation
                            and previous.response_max_bytes == options.response_max_bytes
                        ):
                            model = existing.model
                            if existing.surface.info.gauss.shape[0] == quadrature:
                                surface = existing.surface
                            break
                    if model is None:
                        model = ShellModel(options) if options.backend == "shell" else StressModel(options)
                    if surface is None:
                        surface = StressSurface(quadrature, model.load_info)
                    reuse_contact_moments = (
                        options.contact_model == "pressure"
                        and options.load_transfer == "patch"
                        and gs.backend == gs.cuda
                        and options.face_parallel_scatter
                        and options.cooperative_pressure
                        and options.cooperative_scatter
                        and (
                            options.contact_moment_reuse
                            if options.contact_moment_reuse is not None
                            else (options.radius_model == "hertz" and options.pressure_profile == "hertz")
                            or (options.backend == "volume" and solver._B >= 8192)
                        )
                    )
                    self.links.append(
                        StressLink(
                            link,
                            model,
                            surface,
                            model.create_state(solver._B, options.output_mode),
                            create_contacts(
                                solver.collider.collider_state.contact_sort_idx.shape[0],
                                solver._B,
                                options.contact_radius,
                                options.contact_model == "pressure"
                                and options.load_transfer == "patch"
                                and options.cooperative_pressure,
                                options.contact_model == "pressure"
                                and options.load_transfer == "patch"
                                and options.cooperative_scatter,
                                options.pressure_profile == "hertz",
                            ),
                            V_VEC(3, dtype=gs.qd_float, shape=(solver._B,)),
                            StressHistory(model.info.vertices.shape[0], solver._B, options.history_size)
                            if options.history_size
                            else None,
                            options.output_mode,
                            create_scatter_workspace(
                                solver._B,
                                options.scatter_tasks_per_env
                                if options.contact_model == "pressure"
                                and options.load_transfer == "patch"
                                and options.face_parallel_scatter
                                and options.cooperative_scatter
                                and gs.backend == gs.cuda
                                else 0,
                                reuse_contact_moments,
                            ),
                            options.model_copy(deep=True),
                            options.contact_wrench_reuse
                            if options.contact_wrench_reuse is not None
                            else solver._B >= 8192,
                            options.packed_block_size
                            if options.packed_block_size is not None
                            else 512
                            if solver._B >= 8192
                            else 0,
                            options.contact_model == "pressure"
                            and options.load_transfer == "patch"
                            and gs.backend == gs.cuda
                            and options.fused_pipeline
                            and options.backend == "volume"
                            and model.surface_inverse is not None
                            and options.packed_surface_loads
                            and options.cooperative_pressure
                            and options.cooperative_scatter
                            and options.cooperative_solve
                            and model.info.vertices.shape[0] <= 1024
                            and not options.history_size
                            and not options.atlas_regions
                            and options.stress_response == "none",
                            options.coalesced_residual if options.coalesced_residual is not None else solver._B >= 8192,
                            reuse_contact_moments,
                        )
                    )
        for entry in self.links:
            if entry.model.output_response is not None:
                entry.response = StressResponse(entry.model.output_response, solver._B)
            if entry.options.atlas_regions:
                entry.atlas = StressAtlas(
                    entry.model, entry.contacts, entry.options.atlas_regions, entry.options.atlas_transition
                )
            if entry.options.radius_model == "hertz":
                if self.curvature is None:
                    self.curvature = StressCurvature(solver.dyn_info, self.source_epsilon)
                entry.automatic_radius = StressRadiusInfo(
                    solver.dyn_info,
                    self.curvature.info,
                    entry.options.contact_modulus
                    if entry.options.contact_modulus is not None
                    else entry.options.young / (1.0 - entry.options.poisson**2),
                    entry.options.contact_radius,
                    self.source_epsilon,
                    entry.options.box_edge_policy == "planar",
                )
            if entry.options.load_transfer == "resolved":
                entry.resolved = ResolvedContacts(entry.link, entry.contacts, entry.model.info, entry.options)
            elif entry.options.contact_model == "intrinsic":
                paths = None
                for previous in self.links:
                    if (
                        previous.intrinsic is not None
                        and previous.model is entry.model
                        and previous.options.surface_edge_subdivisions == entry.options.surface_edge_subdivisions
                    ):
                        paths = previous.intrinsic.paths
                        break
                entry.intrinsic = IntrinsicContacts(
                    entry.link, entry.contacts, entry.model.info, entry.surface.info, entry.options, paths
                )
        self.subscribers = [
            Subscriber(
                to=frozenset((StateChange.GEOMETRY, StateChange.DYNAMICS)),
                callback=partial(self._state_changed, entry=entry),
                links_filter=[entry.link.idx],
            )
            for entry in self.links
        ]
        for subscriber in self.subscribers:
            solver.subscribe(subscriber)
        self.post_kernel = None
        if len(self.links) == 1:
            entry = self.links[0]
            requested = entry.options.fused_rigid_post
            selected = requested if requested is not None else solver._B < 8192
            if entry.fused_pipeline and selected:
                self.post_kernel = create_post_kernel()

    @property
    def is_update_requested(self) -> bool:
        return any(
            entry.options.update_mode == "every_step" or entry.requested_step == self.solver.sim.cur_step_global
            for entry in self.links
        )

    def request_update(self, link_idx: int) -> None:
        if self.solver.sim.cur_substep_global % self.solver.substeps:
            gs.raise_exception("Request stress updates before the first physical substep of a scene step.")
        for entry in self.links:
            if entry.link.idx == link_idx:
                if entry.link.stress_options != entry.options:
                    gs.raise_exception("Changing stress recovery options requires rebuilding the scene.")
                entry.requested_step = self.solver.sim.cur_step_global
                return
        gs.raise_exception("Stress recovery is not enabled for this link.")

    def _pipeline_arguments(self, entry: StressLink, i_substep: int):
        solver = self.solver
        options = entry.options
        return (
            options.young,
            options.poisson,
            options.tolerance,
            options.absolute_tolerance,
            self.source_epsilon,
            entry.link.idx,
            solver._links_offset_pos,
            solver._links_offset_quat,
            entry.omega,
            solver.dyn_state,
            solver.collider.collider_state,
            entry.state,
            entry.contacts,
            entry.model.info,
            entry.surface.info,
            entry.model.surface_inverse.info,
            entry.model.inverse.info,
            entry.model.factor.info,
            entry.scatter,
            solver._errno,
            solver._links_offset_quat.ndim == 3,
            i_substep == 0,
            not solver._disable_constraint,
            options.inverse_corrections,
            options.cached_peak,
            options.cached_face_bounds,
            entry.model.info.vertices.shape[0],
            entry.output_mode == "full",
            entry.scatter.tasks.shape[0] > 0,
            options.cooperative_balance,
            entry.reuse_contact_wrench,
            entry.packed_block_size,
            entry.coalesced_residual,
            entry.reuse_contact_moments,
            entry.automatic_radius,
        )

    def recover_and_integrate(self, i_substep: int) -> None:
        """Serial post graph; ordinary recover() remains stress-only."""
        entry = self.links[0]
        if entry.link.stress_options != entry.options:
            gs.raise_exception("Changing stress recovery options requires rebuilding the scene.")
        solver = self.solver
        self.post_kernel(
            *self._pipeline_arguments(entry, i_substep),
            solver.constraint_solver.constraint_state,
            solver.dyn_info,
            solver.rigid_info,
            solver.rigid_config,
            solver._options.noslip_iterations > 0,
            solver._is_backward,
            not solver._disable_constraint,
        )

    def recover(self, i_substep: int) -> None:
        solver = self.solver
        for entry in self.links:
            options = entry.link.stress_options
            if options != entry.options:
                gs.raise_exception("Changing stress recovery options requires rebuilding the scene.")
            if options.update_mode == "manual" and entry.requested_step != solver.sim.cur_step_global:
                continue
            if entry.fused_pipeline:
                kernel_pipeline(*self._pipeline_arguments(entry, i_substep))
                continue
            if i_substep == 0:
                kernel_begin_step(entry.state)
            kernel_associate(
                entry.link.idx,
                solver._links_offset_pos,
                solver._links_offset_quat,
                entry.omega,
                solver.dyn_state,
                entry.contacts,
                solver.collider.collider_state,
                solver._links_offset_quat.ndim == 3,
                not solver._disable_constraint,
                entry.state.step_valid,
                solver._errno,
                entry.automatic_radius,
            )
            if entry.resolved is not None:
                kernel_capture_rigid_sources(
                    entry.link.idx, entry.contacts, entry.resolved.raw_sources, solver.collider.collider_state
                )
                entry.resolved.scatter(
                    entry.contacts, entry.state, entry.model.load_info, entry.surface.info, options, self.source_epsilon
                )
            elif entry.intrinsic is not None:
                kernel_capture_rigid_sources(
                    entry.link.idx, entry.contacts, entry.intrinsic.raw_sources, solver.collider.collider_state
                )
                entry.intrinsic.scatter(
                    entry.contacts,
                    entry.state,
                    entry.model.load_info,
                    entry.surface.info,
                    options,
                    self.source_epsilon,
                    solver._errno,
                )
            else:
                kernel_anchor(self.source_epsilon, entry.contacts, entry.surface.info)
                if entry.reuse_contact_moments:
                    kernel_pressure_moments(
                        entry.contacts,
                        entry.state,
                        entry.model.load_info,
                        entry.surface.info,
                        entry.scatter,
                        options.cached_face_bounds,
                    )
                else:
                    kernel_pressure(entry.contacts, entry.surface.info, options.cooperative_pressure)
                if entry.atlas is not None:
                    entry.atlas.prepare(
                        entry.contacts, entry.model.load_info, entry.surface.info, options.atlas_transition
                    )
                if entry.scatter.tasks.shape[0] and options.cooperative_scatter:
                    kernel_scatter_faces(
                        entry.contacts,
                        entry.state,
                        entry.model.load_info,
                        entry.surface.info,
                        entry.scatter,
                        options.cached_face_bounds,
                        entry.reuse_contact_moments,
                        entry.atlas.scatter if entry.atlas is not None else None,
                    )
                else:
                    kernel_scatter(
                        entry.contacts,
                        entry.state,
                        entry.model.load_info,
                        entry.surface.info,
                        options.cooperative_scatter,
                        options.cached_face_bounds,
                        entry.atlas.scatter if entry.atlas is not None else None,
                    )
            entry.model.recover(
                entry.omega,
                entry.state,
                entry.link.stress_options,
                entry.history,
                surface_load=True,
                atlas=entry.atlas,
                response=entry.response,
            )
            kernel_accept(entry.state, entry.contacts, solver._errno)
            if entry.history is not None:
                entry.history.append(entry.state)

    def _state_changed(self, change: StateChange, envs_idx, *, entry: StressLink) -> None:
        # The outer native mutation emits both notices for a state restore.
        # Geometry already invalidates the complete observation/history state.
        if change is StateChange.DYNAMICS and StateChange.GEOMETRY in self.solver._mutation_changes:
            return
        self.reset(change, envs_idx, entry=entry)

    def reset(self, change: StateChange, envs_idx, *, entry: StressLink | None = None) -> None:
        envs_idx = self.solver.scene._sanitize_envs_idx(envs_idx)
        for affected in self.links if entry is None else (entry,):
            affected.requested_step = -1
            kernel_reset(envs_idx, affected.state)
            if affected.history is not None:
                affected.history.reset(envs_idx)

    def describe_load_failures(self) -> str:
        reasons = {
            1: "invalid input or force outside the supplied unilateral friction cone",
            2: "force line has no admissible exterior anchor",
            3: "nonnegative sampled pad fit unresolved after local integration; physical infeasibility is not established",
            4: "integrated contact force or moment exceeds its preservation budget",
            5: "constrained pressure fit unresolved",
            11: "invalid contact source input",
            12: "ambiguous material-boundary provenance",
            13: "collision witness is outside the declared reference-boundary projection limit",
            21: "intrinsic footprint source unresolved",
            22: "invalid intrinsic footprint input",
            23: "finite intrinsic footprint has no resolved positive integration samples",
            31: "invalid vector traction input",
            32: "vector traction integration unresolved",
            33: "vector traction numerical fit unresolved",
            34: "sampled wrench has a strict infeasibility certificate; continuum feasibility remains unresolved",
            35: "force is incompatible with the continuous reference-surface friction cones inside the physical radius",
            36: "original wrench is incompatible with the continuous reference-surface friction cones inside the physical radius",
            40: "vector traction fails final original wrench or local friction acceptance",
        }
        details = []
        for entry in self.links:
            if entry.intrinsic is not None:
                failure = entry.intrinsic.failure
                preserved = qd_to_numpy(failure.status, transpose=True)
                environments = np.flatnonzero(preserved != 0)
                if len(environments):
                    indices = qd_to_numpy(failure.contact, transpose=True)
                    radius = qd_to_numpy(failure.radius, transpose=True)
                    friction = qd_to_numpy(failure.friction, transpose=True)
                    group_count = qd_to_numpy(failure.group_count, transpose=True)
                    for env in environments[:8]:
                        code = int(preserved[env])
                        details.append(
                            f"link={entry.link.idx}, env={env}, contact={indices[env]}, status={code}: {reasons[code]}; "
                            f"radius={radius[env]:.9g} m, mu={friction[env]:.9g} (first preserved contact failure)"
                        )
                        if group_count[env] > 1:
                            details.append(
                                f"env={env}: body_source_prior conserves the aggregate wrench of "
                                f"{group_count[env]} original contacts; target_force and target_moment "
                                "are retained in the failure record about its original application point"
                            )
                    if len(environments) > 8:
                        details.append(
                            f"{len(environments) - 8} additional preserved environment failures on this link"
                        )
                    continue
            status = qd_to_numpy(entry.contacts.status, transpose=True)
            failed = np.argwhere(status != 0)
            if not len(failed):
                continue
            radius = qd_to_numpy(entry.contacts.radius, transpose=True)
            friction = qd_to_numpy(entry.contacts.friction, transpose=True)
            radius_status = qd_to_numpy(entry.contacts.radius_status, transpose=True)
            for env, contact in failed[:8]:
                code = int(status[env, contact])
                reason = reasons[code]
                if entry.automatic_radius is not None:
                    if radius_status[env, contact] == StressRadiusStatus.UNSUPPORTED:
                        reason = (
                            "automatic Hertz radius unsupported for the current surface geometry or relative curvature"
                        )
                    elif radius_status[env, contact] == StressRadiusStatus.INVALID:
                        reason = "automatic Hertz radius requires finite compressive contact inputs"
                details.append(
                    f"link={entry.link.idx}, env={env}, contact={contact}, status={code}: {reason}; "
                    f"radius={radius[env, contact]:.9g} m, mu={friction[env, contact]:.9g}"
                )
            if len(failed) > 8:
                details.append(f"{len(failed) - 8} additional current contact failures on this link")
        if not details:
            details.append(
                "No failed contact remains in the current buffers; the error may precede the latest recovery "
                "or involve incompatible applied rigid loads or mass properties."
            )
        return "; ".join(details)

    @property
    def data(self) -> Iterator[DataItem]:
        if self.curvature is not None:
            yield from iter_data(self.curvature.info, "stress.curvature")
        seen = set()
        seen_surfaces = set()
        seen_paths = set()
        for i, entry in enumerate(self.links):
            if entry.link.stress_options != entry.options:
                gs.raise_exception("Changing stress recovery options requires rebuilding the scene.")
            yield DataItem(
                f"stress.{i}.options",
                entry.options.model_copy(update={"mesh": str(Path(entry.options.mesh).resolve())}),
                DataKind.CONFIG,
            )
            if id(entry.model) not in seen:
                yield from entry.model.iter_info_data(f"stress.{i}")
                if entry.model.factor is not None:
                    yield from iter_data(entry.model.factor.info, f"stress.{i}.factor")
                if entry.model.inverse is not None:
                    yield from iter_data(entry.model.inverse.info, f"stress.{i}.inverse")
                if entry.model.surface_inverse is not None:
                    yield from iter_data(entry.model.surface_inverse.info, f"stress.{i}.surface_inverse")
                seen.add(id(entry.model))
            if id(entry.surface) not in seen_surfaces:
                yield from iter_data(entry.surface.info, f"stress.{i}.surface")
                seen_surfaces.add(id(entry.surface))
            for item in iter_data(entry.state, f"stress.{i}.state"):
                if 0 not in item.value.shape:
                    yield item
            for item in iter_data(entry.contacts, f"stress.{i}.contacts"):
                if 0 not in item.value.shape:
                    yield item
            for item in iter_data(entry.scatter, f"stress.{i}.scatter"):
                if 0 not in item.value.shape:
                    yield item
            if entry.intrinsic is not None:
                if id(entry.intrinsic.paths) not in seen_paths:
                    yield from iter_data(entry.intrinsic.paths.info, f"stress.{i}.intrinsic.paths")
                    seen_paths.add(id(entry.intrinsic.paths))
                yield from entry.intrinsic.data(f"stress.{i}.intrinsic")
            if entry.response is not None:
                yield from iter_data(entry.response.state, f"stress.{i}.output_response.state")
            if entry.atlas is not None:
                yield from iter_data(entry.atlas.info, f"stress.{i}.atlas.info")
                yield from iter_data(entry.atlas.state, f"stress.{i}.atlas.state")
            if entry.history is not None:
                yield from iter_data(entry.history.state, f"stress.{i}.history")
            if entry.resolved is not None:
                yield from iter_data(entry.resolved.raw_sources, f"stress.{i}.resolved.sources")
                if entry.resolved.vertices.info is not None:
                    yield from iter_data(entry.resolved.vertices.info, f"stress.{i}.resolved.vertices")
            yield DataItem(f"stress.{i}.omega", entry.omega, entry.state.kind)

    def get_peak(self, link_idx: int, envs_idx=None, *, copy: bool = True) -> torch.Tensor:
        for entry in self.links:
            if entry.link.idx == link_idx:
                if entry.link.stress_options != entry.options:
                    gs.raise_exception("Changing stress recovery options requires rebuilding the scene.")
                tensor = qd_to_torch(entry.state.step_peak, envs_idx, transpose=True, copy=copy)
                return tensor[0] if self.solver.n_envs == 0 else tensor
        gs.raise_exception("Stress recovery is not enabled for this link.")

    def get_field(self, link_idx: int, envs_idx=None, *, copy: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        for entry in self.links:
            if entry.link.idx == link_idx:
                if entry.link.stress_options != entry.options:
                    gs.raise_exception("Changing stress recovery options requires rebuilding the scene.")
                if entry.output_mode != "full":
                    gs.raise_exception("Configure output_mode='full' before building to observe the stress field.")
                tensor = qd_to_torch(entry.state.stress_tensor, envs_idx, transpose=True, copy=copy)
                von_mises = qd_to_torch(entry.state.von_mises, envs_idx, transpose=True, copy=copy).squeeze(-1)
                if entry.model.n_shell_elements:
                    tensor = tensor.reshape((tensor.shape[0], entry.model.n_shell_elements, 2, 4, 6))
                    von_mises = von_mises.reshape((von_mises.shape[0], entry.model.n_shell_elements, 2, 4))
                if self.solver.n_envs == 0:
                    tensor, von_mises = tensor[0], von_mises[0]
                return tensor, von_mises
        gs.raise_exception("Stress recovery is not enabled for this link.")

    def get_shell_resultants(
        self, link_idx: int, envs_idx=None, *, copy: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for entry in self.links:
            if entry.link.idx == link_idx:
                if entry.link.stress_options != entry.options:
                    gs.raise_exception("Changing stress recovery options requires rebuilding the scene.")
                if not entry.model.n_shell_elements or entry.output_mode != "full":
                    gs.raise_exception(
                        "Shell resultants require backend='shell' and output_mode='full' before building."
                    )
                values = tuple(
                    qd_to_torch(value, envs_idx, transpose=True, copy=copy)
                    for value in (
                        entry.state.membrane_resultant,
                        entry.state.bending_resultant,
                        entry.state.transverse_resultant,
                    )
                )
                return tuple(value[0] for value in values) if self.solver.n_envs == 0 else values
        gs.raise_exception("Stress recovery is not enabled for this link.")

    def set_radius(self, link_idx: int, radius, envs_idx=None) -> None:
        if envs_idx is not None and self.solver.n_envs == 0:
            gs.raise_exception("`envs_idx` is not supported for a scene without parallel environments.")
        if isinstance(envs_idx, (int, np.integer)) and not -self.solver._B <= envs_idx < self.solver._B:
            gs.raise_exception("Stress radius environment index is out of bounds.")
        for entry in self.links:
            if entry.link.idx == link_idx:
                if entry.options.load_transfer == "resolved":
                    gs.raise_exception("Resolved boundary samples have no reconstruction radius.")
                if entry.options.radius_model != "manual":
                    gs.raise_exception(
                        "Automatic contact radii are recomputed from current loads; select manual radii to set them."
                    )
                if gs.use_zerocopy:
                    view = qd_to_torch(entry.contacts.radius, transpose=True, copy=False)
                    mask = indices_to_mask(envs_idx)
                    values = broadcast_tensor(radius, gs.tc_float, view[mask].shape)
                    assign_indexed_tensor(view, mask, values)
                    if gs.backend == gs.metal:
                        torch.mps.synchronize()
                else:
                    envs = self.solver.scene._sanitize_envs_idx(envs_idx)
                    values = broadcast_tensor(radius, gs.tc_float, (envs.shape[0], entry.contacts.radius.shape[0]))
                    kernel_set_radius(envs, values.contiguous(), entry.contacts)
                return
        gs.raise_exception("Stress recovery is not enabled for this link.")

    def reject_link_mutation(self, links_idx, reason: str) -> None:
        if isinstance(links_idx, torch.Tensor):
            links_idx = tensor_to_array(links_idx)
        selected = np.atleast_1d(np.arange(self.solver.n_links)[indices_to_mask(links_idx)])
        if any(entry.link.idx in selected for entry in self.links):
            gs.raise_exception(reason)


@qd.kernel
def kernel_begin_step(stress_state: StressState):
    func_begin_step(stress_state)


def kernel_accept(stress_state: StressState, contact_state: StressContactState, errno: qd.Tensor):
    kernel_accept_impl(stress_state, contact_state, errno, stress_state.stress_tensor.shape[0] > 0)


@qd.kernel(graph=True)
def kernel_accept_impl(
    stress_state: StressState, contact_state: StressContactState, errno: qd.Tensor, full: qd.template()
):
    func_accept(stress_state, contact_state, errno, full)


def kernel_reset(envs_idx: qd.types.ndarray(), stress_state: StressState):
    kernel_reset_impl(envs_idx, stress_state, stress_state.stress_tensor.shape[0] > 0)


@qd.kernel
def kernel_reset_impl(envs_idx: qd.types.ndarray(), stress_state: StressState, full: qd.template()):
    for i_n, i_selected in qd.ndrange(stress_state.displacement.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        stress_state.displacement[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_selected in range(envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        stress_state.peak[i_b] = 0.0
        stress_state.step_peak[i_b] = 0.0
        stress_state.valid[i_b] = True
        stress_state.step_valid[i_b] = True
    if qd.static(full):
        for i_e, i_corner, i_selected in qd.ndrange(stress_state.stress_tensor.shape[0], 4, envs_idx.shape[0]):
            i_b = envs_idx[i_selected]
            stress_state.stress_tensor[i_e, i_corner, i_b] = qd.Vector([float("nan") for _ in qd.static(range(6))])
            stress_state.von_mises[i_e, i_corner, i_b] = qd.Vector([float("nan")])
        for i_e, i_q, i_selected in qd.ndrange(stress_state.membrane_resultant.shape[0], 4, envs_idx.shape[0]):
            i_b = envs_idx[i_selected]
            stress_state.membrane_resultant[i_e, i_q, i_b] = qd.Vector([float("nan")] * 6)
            stress_state.bending_resultant[i_e, i_q, i_b] = qd.Vector([float("nan")] * 6)
            stress_state.transverse_resultant[i_e, i_q, i_b] = qd.Vector([float("nan")] * 3)


@qd.kernel
def kernel_set_radius(envs_idx: qd.types.ndarray(), radius: qd.types.ndarray(), contact_state: StressContactState):
    for i_c, i_selected in qd.ndrange(contact_state.radius.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        contact_state.radius[i_c, i_b] = radius[i_selected, i_c]
