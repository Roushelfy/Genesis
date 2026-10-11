"""Serial contact-force, stress and rigid integration graph factory."""

from functools import lru_cache

import quadrants as qd

from genesis.utils import array_class

from .contact import StressContactState
from .data import StressInfo, StressState
from .factor import StressFactorInfo
from .inverse import StressInverseInfo
from .pipeline import func_pipeline
from .scatter import StressScatterWorkspace
from .surface import StressSurfaceInfo
from .surface_inverse import StressSurfaceInverseInfo


@lru_cache(maxsize=1)
def create_post_kernel():
    # Resolve the rigid functions after RigidSolver finishes importing recovery.
    from ..rigid_solver import func_resolve_post, func_step_2

    @qd.kernel(graph=True)
    def kernel_rigid_stress_post(
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
        constraint_state: array_class.ConstraintState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        rigid_config: qd.template(),
        noslip: qd.template(),
        is_backward: qd.template(),
        constraints_enabled: qd.template(),
    ):
        if qd.static(constraints_enabled):
            func_resolve_post(dyn, collider, constraint_state, dyn_info, rigid_info, rigid_config, noslip, errno)
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
        )
        func_step_2(dyn, constraint_state, dyn_info, rigid_info, rigid_config, is_backward, errno)

    return kernel_rigid_stress_post
