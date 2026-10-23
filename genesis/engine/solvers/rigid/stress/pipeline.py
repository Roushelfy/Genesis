"""Serial native graph for complete exterior-load recovery and acceptance."""

import quadrants as qd

from genesis.utils import array_class

from .association import func_associate
from .balance import func_balance_contacts
from .contact import (
    StressContactState,
    func_anchor,
    func_pack_contacts,
    func_pressure_correct_warp,
    func_pressure_warp,
    func_refine_contacts,
    func_scatter_warp,
)
from .data import StressInfo, StressState
from .factor import StressFactorInfo, func_solve_cooperative
from .inverse import StressInverseInfo, func_apply_inverse
from .lifecycle import func_accept, func_begin_step
from .scatter import StressScatterWorkspace, func_pressure_moments, func_scatter_faces
from .solve import func_balance, func_direct_init, func_full_residual, func_peak_impl
from .surface import StressSurfaceInfo
from .surface_inverse import StressSurfaceInverseInfo, func_surface_apply_packed, func_surface_pack


@qd.func
def func_pipeline(
    young: float,
    poisson: float,
    tolerance: float,
    absolute: float,
    epsilon: float,
    link: int,
    offsets_pos: qd.types.ndarray(),
    offsets_quat: qd.types.ndarray(),
    omega: qd.Tensor,
    dyn: array_class.DynState,
    collider: array_class.ColliderState,
    state: StressState,
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    boundary: StressSurfaceInverseInfo,
    inverse: StressInverseInfo,
    factor: StressFactorInfo,
    scatter: StressScatterWorkspace,
    errno: qd.Tensor,
    batch_offsets: qd.template(),
    first: qd.template(),
    enable: bool,
    corrections: qd.template(),
    cached_peak: qd.template(),
    cached_bounds: qd.template(),
    n_nodes: qd.template(),
    full: qd.template(),
    face_parallel: qd.template(),
    is_cooperative_balance: qd.template(),
    reuse_contact_wrench: qd.template(),
    packed_block_size: qd.template(),
    coalesced_residual: qd.template(),
    reuse_contact_moments: qd.template(),
    automatic_radius: qd.types.Template | None,
):
    if qd.static(first):
        func_begin_step(state)
    func_associate(
        link,
        offsets_pos,
        offsets_quat,
        omega,
        dyn,
        contacts,
        collider,
        batch_offsets,
        enable,
        state.step_valid,
        errno,
        automatic_radius,
    )
    func_anchor(epsilon, contacts, surface)
    func_pack_contacts(contacts)
    if qd.static(reuse_contact_moments):
        func_pressure_moments(contacts, state, info, surface, scatter, cached_bounds)
    else:
        func_pressure_warp(contacts, surface, False)
    func_pressure_correct_warp(contacts, surface)
    func_refine_contacts(contacts)
    func_pressure_warp(contacts, surface, True)
    func_pressure_correct_warp(contacts, surface)
    if qd.static(face_parallel):
        func_scatter_faces(contacts, state, info, surface, scatter, cached_bounds, reuse_contact_moments)
    else:
        func_scatter_warp(contacts, state, info, surface, cached_bounds)
    if qd.static(face_parallel and is_cooperative_balance and reuse_contact_wrench):
        func_balance_contacts(omega, state, info, contacts, scatter)
    else:
        func_balance(omega, state, info, is_cooperative_balance)
    func_direct_init(state)
    func_surface_pack(state, boundary)
    func_surface_apply_packed(young, omega, state, boundary, packed_block_size)
    func_full_residual(young, tolerance, absolute, state, info, False, coalesced_residual)
    for _ in qd.static(range(corrections)):
        func_apply_inverse(young, state, inverse, True, False)
        func_full_residual(young, tolerance, absolute, state, info, True, coalesced_residual)
    func_solve_cooperative(young, state, info, factor, n_nodes, True)
    func_full_residual(young, tolerance, absolute, state, info, True, coalesced_residual)
    func_peak_impl(young, poisson, state, info, cached_peak, full)
    func_accept(state, contacts, errno, full)


@qd.kernel(graph=True)
def kernel_pipeline(
    young: float,
    poisson: float,
    tolerance: float,
    absolute: float,
    epsilon: float,
    link: int,
    offsets_pos: qd.types.ndarray(),
    offsets_quat: qd.types.ndarray(),
    omega: qd.Tensor,
    dyn: array_class.DynState,
    collider: array_class.ColliderState,
    state: StressState,
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    boundary: StressSurfaceInverseInfo,
    inverse: StressInverseInfo,
    factor: StressFactorInfo,
    scatter: StressScatterWorkspace,
    errno: qd.Tensor,
    batch_offsets: qd.template(),
    first: qd.template(),
    enable: bool,
    corrections: qd.template(),
    cached_peak: qd.template(),
    cached_bounds: qd.template(),
    n_nodes: qd.template(),
    full: qd.template(),
    face_parallel: qd.template(),
    is_cooperative_balance: qd.template(),
    reuse_contact_wrench: qd.template(),
    packed_block_size: qd.template(),
    coalesced_residual: qd.template(),
    reuse_contact_moments: qd.template(),
    automatic_radius: qd.types.Template | None,
):
    func_pipeline(
        young,
        poisson,
        tolerance,
        absolute,
        epsilon,
        link,
        offsets_pos,
        offsets_quat,
        omega,
        dyn,
        collider,
        state,
        contacts,
        info,
        surface,
        boundary,
        inverse,
        factor,
        scatter,
        errno,
        batch_offsets,
        first,
        enable,
        corrections,
        cached_peak,
        cached_bounds,
        n_nodes,
        full,
        face_parallel,
        is_cooperative_balance,
        reuse_contact_wrench,
        packed_block_size,
        coalesced_residual,
        reuse_contact_moments,
        automatic_radius,
    )
