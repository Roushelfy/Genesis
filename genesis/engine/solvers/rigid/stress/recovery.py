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
from genesis.utils.array_class import V_VEC, DataItem, ErrorCode, iter_data
from genesis.utils.misc import assign_indexed_tensor, broadcast_tensor, indices_to_mask, qd_to_torch

from .association import kernel_associate
from .contact import StressContactState, create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from .data import StressState
from .model import StressModel
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


class RigidStressRecovery:
    def __init__(self, solver: "RigidSolver"):
        self.solver = solver
        self.links: list[StressLink] = []
        for entity in solver.entities:
            for link in entity.links:
                options = link.stress_options
                if options is not None:
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
                            model.create_state(solver._B),
                            create_contacts(
                                solver.collider.collider_state.contact_sort_idx.shape[0],
                                solver._B,
                                options.contact_radius,
                            ),
                            V_VEC(3, dtype=gs.qd_float, shape=(solver._B,)),
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
            )
            kernel_anchor(self.source_epsilon, entry.contacts, entry.surface.info)
            kernel_pressure(entry.contacts, entry.surface.info)
            kernel_scatter(entry.contacts, entry.state, entry.model.info, entry.surface.info)
            entry.model.recover(entry.omega, entry.state, entry.link.stress_options)
            kernel_accept(entry.state, entry.contacts, solver._errno)

    def reset(self, change: StateChange, envs_idx) -> None:
        envs_idx = self.solver.scene._sanitize_envs_idx(envs_idx)
        for entry in self.links:
            kernel_reset(envs_idx, entry.state)

    @property
    def data(self) -> Iterator[DataItem]:
        seen = set()
        seen_surfaces = set()
        for i, entry in enumerate(self.links):
            if id(entry.model) not in seen:
                yield from iter_data(entry.model.info, f"stress.{i}.info")
                if entry.model.factor is not None:
                    yield from iter_data(entry.model.factor.info, f"stress.{i}.factor")
                seen.add(id(entry.model))
            if id(entry.surface) not in seen_surfaces:
                yield from iter_data(entry.surface.info, f"stress.{i}.surface")
                seen_surfaces.add(id(entry.surface))
            yield from iter_data(entry.state, f"stress.{i}.state")
            yield from iter_data(entry.contacts, f"stress.{i}.contacts")
            yield DataItem(f"stress.{i}.omega", entry.omega, entry.state.kind)

    def get_peak(self, link_idx: int, envs_idx=None, *, copy: bool = True) -> torch.Tensor:
        for entry in self.links:
            if entry.link.idx == link_idx:
                tensor = qd_to_torch(entry.state.step_peak, envs_idx, transpose=True, copy=copy)
                return tensor[0] if self.solver.n_envs == 0 else tensor
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
                    kernel_set_radius(envs, values, entry.contacts)
                return
        gs.raise_exception("Stress recovery is not enabled for this link.")


@qd.kernel
def kernel_begin_step(stress_state: StressState):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.step_peak[i_b] = 0.0
        stress_state.step_valid[i_b] = True


@qd.kernel(graph=True)
def kernel_accept(stress_state: StressState, contact_state: StressContactState, errno: qd.Tensor):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b] and contact_state.status[i_c, i_b] != 0:
            stress_state.step_valid[i_b] = False
            qd.atomic_or(errno[i_b], ErrorCode.INVALID_STRESS_LOAD)
    for i_b in range(stress_state.active.shape[0]):
        stress_state.step_valid[i_b] = stress_state.step_valid[i_b] and stress_state.valid[i_b]
        if not stress_state.valid[i_b]:
            qd.atomic_or(errno[i_b], ErrorCode.INVALID_STRESS_SOLVE)
        if not stress_state.step_valid[i_b]:
            stress_state.step_peak[i_b] = float("nan")


@qd.kernel
def kernel_reset(envs_idx: qd.types.ndarray(), stress_state: StressState):
    for i_n, i_selected in qd.ndrange(stress_state.displacement.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        stress_state.displacement[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_selected in range(envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        stress_state.peak[i_b] = 0.0
        stress_state.step_peak[i_b] = 0.0
        stress_state.valid[i_b] = True
        stress_state.step_valid[i_b] = True


@qd.kernel
def kernel_set_radius(envs_idx: qd.types.ndarray(), radius: qd.types.ndarray(), contact_state: StressContactState):
    for i_c, i_selected in qd.ndrange(contact_state.radius.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        contact_state.radius[i_c, i_b] = radius[i_selected, i_c]
