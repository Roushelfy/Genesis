import quadrants as qd

from genesis.utils.array_class import ErrorCode

from .contact import StressContactState
from .data import StressState


@qd.func
def func_begin_step(stress_state: StressState):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.step_peak[i_b] = 0.0
        stress_state.step_valid[i_b] = True
        stress_state.corrections[i_b] = 0
        stress_state.fallbacks[i_b] = 0
        stress_state.invalid_reported[i_b] = False


@qd.func
def func_accept(
    stress_state: StressState, contact_state: StressContactState, errno: qd.Tensor, full: qd.template() = False
):
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
            if not stress_state.invalid_reported[i_b]:
                stress_state.invalid_steps[i_b] += 1
                stress_state.invalid_reported[i_b] = True
    if qd.static(full):
        for i_e, i_corner, i_b in qd.ndrange(stress_state.stress_tensor.shape[0], 4, stress_state.active.shape[0]):
            if not stress_state.step_valid[i_b]:
                stress_state.stress_tensor[i_e, i_corner, i_b] = qd.Vector([float("nan") for _ in qd.static(range(6))])
                stress_state.von_mises[i_e, i_corner, i_b] = qd.Vector([float("nan")])
