"""Native graph for shell recovery with complete residual checks and sampled output."""

import quadrants as qd

from .data import StressInfo, StressState
from .factor import StressFactorInfo
from .inverse import StressInverseInfo
from .pipeline import func_solve_surface
from .shell_operators import ShellInfo, func_peak
from .solve import func_balance
from .surface_inverse import StressSurfaceInverseInfo


@qd.kernel(graph=True)
def kernel_recover(
    young: float,
    poisson: float,
    thickness: float,
    tolerance: float,
    absolute: float,
    omega: qd.Tensor,
    state: StressState,
    info: StressInfo,
    shell: ShellInfo,
    boundary: StressSurfaceInverseInfo,
    inverse: StressInverseInfo,
    factor: StressFactorInfo,
    corrections: qd.template(),
    capacity: qd.template(),
    packed_block_size: qd.template(),
    coalesced_residual: qd.template(),
    cooperative_balance: qd.template(),
):
    func_balance(omega, state, info, cooperative_balance)
    func_solve_surface(
        young,
        tolerance,
        absolute,
        omega,
        state,
        info,
        boundary,
        inverse,
        factor,
        corrections,
        capacity,
        packed_block_size,
        coalesced_residual,
    )
    func_peak(young, poisson, thickness, state, shell)
