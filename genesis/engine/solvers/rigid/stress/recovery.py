"""Rigid solver lifecycle for optional native stress links."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.base_solver import StateChange, Subscriber
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
from .contact import StressContactState, create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from .data import StressState
from .history import StressHistory
from .lifecycle import func_accept, func_begin_step
from .model import StressModel
from .pipeline import kernel_pipeline
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


class RigidStressRecovery:
    def __init__(self, solver: "RigidSolver"):
        self.solver = solver
        self.links: list[StressLink] = []
        for entity in solver.entities:
            for link in entity.links:
                options = link.stress_options
                if options is not None:
                    if options.history_size and options.method == "pcg":
                        gs.raise_exception("The native load history requires a direct or inverse stress method.")
                    # Equal geometry, constitutive parameters and mass share physical operators and factors.
                    model = None
                    surface = None
                    for existing in self.links:
                        previous = existing.model.options
                        if (
                            Path(previous.mesh).resolve() == Path(options.mesh).resolve()
                            and previous.young == options.young
                            and previous.poisson == options.poisson
                            and previous.density == options.density
                            and previous.method == options.method
                            and previous.inverse_precision == options.inverse_precision
                            and previous.inverse_max_bytes == options.inverse_max_bytes
                            and previous.surface_inverse == options.surface_inverse
                        ):
                            model = existing.model
                            if previous.quadrature == options.quadrature:
                                surface = existing.surface
                            break
                    if model is None:
                        model = StressModel(options)
                    if surface is None:
                        surface = StressSurface(options.quadrature, model.info)
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
                                options.cooperative_pressure,
                                options.cooperative_scatter,
                            ),
                            V_VEC(3, dtype=gs.qd_float, shape=(solver._B,)),
                            StressHistory(model.info.vertices.shape[0], solver._B) if options.history_size else None,
                            options.output_mode,
                        )
                    )
        self.subscriber = Subscriber(
            to=frozenset((StateChange.GEOMETRY, StateChange.DYNAMICS)),
            callback=self.reset,
            links_filter=[entry.link.idx for entry in self.links],
        )
        solver.subscribe(self.subscriber)
        self.source_epsilon = float(np.finfo(gs.np_float).eps)

    def recover(self, i_substep: int) -> None:
        solver = self.solver
        for entry in self.links:
            options = entry.link.stress_options
            if options.output_mode != entry.output_mode:
                gs.raise_exception("Changing stress output mode requires rebuilding the scene.")
            if (
                gs.backend == gs.cuda
                and options.fused_pipeline
                and entry.model.surface_inverse is not None
                and options.packed_surface_loads
                and options.cooperative_pressure
                and options.cooperative_scatter
                and options.cooperative_solve
                and entry.model.info.vertices.shape[0] <= 1024
                and entry.history is None
            ):
                kernel_pipeline(
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
                    solver._errno,
                    solver._links_offset_quat.ndim == 3,
                    i_substep == 0,
                    not solver._disable_constraint,
                    options.inverse_corrections,
                    options.cached_peak,
                    options.cached_face_bounds,
                    entry.model.info.vertices.shape[0],
                    entry.output_mode == "full",
                )
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
            )
            kernel_anchor(self.source_epsilon, entry.contacts, entry.surface.info)
            kernel_pressure(entry.contacts, entry.surface.info, entry.link.stress_options.cooperative_pressure)
            kernel_scatter(
                entry.contacts,
                entry.state,
                entry.model.info,
                entry.surface.info,
                entry.link.stress_options.cooperative_scatter,
                entry.link.stress_options.cached_face_bounds,
            )
            entry.model.recover(entry.omega, entry.state, entry.link.stress_options, entry.history, surface_load=True)
            kernel_accept(entry.state, entry.contacts, solver._errno)
            if entry.history is not None:
                entry.history.append(entry.state)

    def reset(self, change: StateChange, envs_idx) -> None:
        envs_idx = self.solver.scene._sanitize_envs_idx(envs_idx)
        for entry in self.links:
            kernel_reset(envs_idx, entry.state)
            if entry.history is not None:
                entry.history.reset(envs_idx)

    def describe_load_failures(self) -> str:
        reasons = {
            1: "invalid input or force outside the supplied unilateral friction cone",
            2: "force line has no admissible exterior anchor",
            3: "nonnegative sampled pad fit unresolved after local integration; physical infeasibility is not established",
            4: "integrated contact force or moment exceeds its preservation budget",
            5: "constrained pressure fit unresolved",
        }
        details = []
        for entry in self.links:
            status = qd_to_numpy(entry.contacts.status, transpose=True)
            failed = np.argwhere(status != 0)
            if not len(failed):
                continue
            radius = qd_to_numpy(entry.contacts.radius, transpose=True)
            friction = qd_to_numpy(entry.contacts.friction, transpose=True)
            for env, contact in failed[:8]:
                code = int(status[env, contact])
                details.append(
                    f"link={entry.link.idx}, env={env}, contact={contact}, status={code}: {reasons[code]}; "
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
        seen = set()
        seen_surfaces = set()
        for i, entry in enumerate(self.links):
            yield DataItem(
                f"stress.{i}.options",
                entry.link.stress_options.model_copy(
                    update={"mesh": str(Path(entry.link.stress_options.mesh).resolve())}
                ),
                DataKind.CONFIG,
            )
            if id(entry.model) not in seen:
                yield from iter_data(entry.model.info, f"stress.{i}.info")
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
            if entry.history is not None:
                yield from iter_data(entry.history.state, f"stress.{i}.history")
            yield DataItem(f"stress.{i}.omega", entry.omega, entry.state.kind)

    def get_peak(self, link_idx: int, envs_idx=None, *, copy: bool = True) -> torch.Tensor:
        for entry in self.links:
            if entry.link.idx == link_idx:
                tensor = qd_to_torch(entry.state.step_peak, envs_idx, transpose=True, copy=copy)
                return tensor[0] if self.solver.n_envs == 0 else tensor
        gs.raise_exception("Stress recovery is not enabled for this link.")

    def get_field(self, link_idx: int, envs_idx=None, *, copy: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        for entry in self.links:
            if entry.link.idx == link_idx:
                if entry.link.stress_options.output_mode != entry.output_mode:
                    gs.raise_exception("Changing stress output mode requires rebuilding the scene.")
                if entry.output_mode != "full":
                    gs.raise_exception("Configure output_mode='full' before building to observe the stress field.")
                tensor = qd_to_torch(entry.state.stress_tensor, envs_idx, transpose=True, copy=copy)
                von_mises = qd_to_torch(entry.state.von_mises, envs_idx, transpose=True, copy=copy).squeeze(-1)
                if self.solver.n_envs == 0:
                    tensor, von_mises = tensor[0], von_mises[0]
                return tensor, von_mises
        gs.raise_exception("Stress recovery is not enabled for this link.")

    def set_radius(self, link_idx: int, radius, envs_idx=None) -> None:
        if envs_idx is not None and self.solver.n_envs == 0:
            gs.raise_exception("`envs_idx` is not supported for a scene without parallel environments.")
        if isinstance(envs_idx, (int, np.integer)) and not -self.solver._B <= envs_idx < self.solver._B:
            gs.raise_exception("Stress radius environment index is out of bounds.")
        for entry in self.links:
            if entry.link.idx == link_idx:
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


@qd.kernel
def kernel_set_radius(envs_idx: qd.types.ndarray(), radius: qd.types.ndarray(), contact_state: StressContactState):
    for i_c, i_selected in qd.ndrange(contact_state.radius.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        contact_state.radius[i_c, i_b] = radius[i_selected, i_c]
