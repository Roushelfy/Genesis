"""Native complete tensors for an explicitly supplied elliptic tangential traction."""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz_tangential import func_hertz_tangential_tensor
from genesis.utils.array_class import V, V_MAT, V_VEC

from .native_hertz import HertzState
from .native_hertz_stress import kernel_hertz_stress_quadrature


@dataclass(frozen=True)
class HertzTangentialProbeState:
    position: qd.Tensor
    poisson: qd.Tensor
    tangent: qd.Tensor
    tensor: qd.Tensor
    status: qd.Tensor


class NativeHertzTangentialStress:
    """Evaluate q = tangent * normal Hertz pressure in the ellipse frame.

    The two-component tangent is a traction ratio, not a velocity or friction
    coefficient. No contact/friction history is inferred here. Status0 is a
    finite solution,1 unloaded,2 invalid inputs and3 a failed confocal root.
    """

    def __init__(self, count: int, probes: int, order: int = 16):
        assert count > 0 and probes > 0 and order >= 8
        self.quadrature = V_VEC(2, dtype=gs.qd_float, shape=(order,))
        self.state = HertzTangentialProbeState(
            position=V_VEC(3, dtype=gs.qd_float, shape=(probes, count)),
            poisson=V(dtype=gs.qd_float, shape=(count,)),
            tangent=V_VEC(2, dtype=gs.qd_float, shape=(count,)),
            tensor=V_MAT(3, 3, dtype=gs.qd_float, shape=(probes, count)),
            status=V(dtype=gs.qd_int, shape=(probes, count)),
        )
        kernel_hertz_stress_quadrature(self.quadrature)

    def update(self, hertz: HertzState):
        kernel_hertz_tangential_stress(self.state, hertz, self.quadrature)


@qd.kernel
def kernel_hertz_tangential_stress(probe: HertzTangentialProbeState, hertz: HertzState, quadrature: qd.types.ndarray()):
    for i_q, i_p in qd.ndrange(probe.status.shape[0], probe.status.shape[1]):
        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        status = 2
        if hertz.status[i_p] == 1:
            status = 1
        elif hertz.status[i_p] == 0:
            point, axes, poisson = probe.position[i_q, i_p], hertz.axes[i_p], probe.poisson[i_p]
            tangent, peak = probe.tangent[i_p], hertz.peak_pressure[i_p]
            is_valid = point.norm() < float("inf") and point[2] >= 0 and tangent.norm() < float("inf")
            is_valid = is_valid and axes.norm() < float("inf") and axes[0] > 0
            is_valid = is_valid and axes[1] >= 0.05 * axes[0] and axes[1] <= axes[0]
            is_valid = is_valid and poisson > -1 and poisson < 0.5 and peak >= 0 and peak < float("inf")
            if is_valid:
                candidate, is_resolved = func_hertz_tangential_tensor(
                    point / axes[0], axes[1] / axes[0], tangent, poisson, quadrature
                )
                status = 3
                if is_resolved and candidate.norm() < float("inf"):
                    tensor, status = candidate * peak, 0
        probe.tensor[i_q, i_p], probe.status[i_q, i_p] = tensor, status
