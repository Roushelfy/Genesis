"""Native comparison of independent regional tensors with merged-displacement recovery."""

from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions

from .atlas import StressAtlasState
from .data import StressState
from .response import func_von_mises

if TYPE_CHECKING:
    from .model import StressModel


class StressAtlasOutput:
    def __init__(self, model: "StressModel", n_envs: int, n_groups: int):
        # The final contribution retains spin, numerical corrections and accepted
        # history states. All contributions share the mother's output frame.
        self.state = model.create_state(n_envs * (n_groups + 1), "full")

    def evaluate(self, options: RigidStressOptions, model: "StressModel", state: StressState, atlas: StressAtlasState):
        kernel_expand(state, self.state, atlas)
        model.evaluate_stress(options, self.state, self.state.valid)
        kernel_combine(state, self.state)


@qd.kernel(graph=True)
def kernel_expand(state: StressState, parts: StressState, atlas: StressAtlasState):
    for i_n, i_b in qd.ndrange(state.displacement.shape[0], state.active.shape[0]):
        residual = state.displacement[i_n, i_b]
        for i_g in range(atlas.counts.shape[0]):
            value = qd.Vector.zero(gs.qd_float, 3)
            if atlas.counts[i_g, i_b]:
                value = atlas.displacement[i_n, i_g, i_b]
            parts.displacement[i_n, i_g * state.active.shape[0] + i_b] = value
            residual -= value
        parts.displacement[i_n, atlas.counts.shape[0] * state.active.shape[0] + i_b] = residual
    for i_g, i_b in qd.ndrange(atlas.counts.shape[0] + 1, state.active.shape[0]):
        active = True
        if i_g < atlas.counts.shape[0]:
            active = atlas.counts[i_g, i_b] > 0
        parts.valid[i_g * state.active.shape[0] + i_b] = active


@qd.kernel(graph=True)
def kernel_combine(state: StressState, parts: StressState):
    for i_b in range(state.active.shape[0]):
        state.peak[i_b] = 0.0
    for i_e, i_q, i_b in qd.ndrange(parts.stress_tensor.shape[0], 4, state.active.shape[0]):
        tensor = qd.Vector.zero(gs.qd_float, 6)
        for i_g in range(parts.active.shape[0] // state.active.shape[0]):
            slot = i_g * state.active.shape[0] + i_b
            if parts.valid[slot]:
                tensor += parts.stress_tensor[i_e, i_q, slot]
        value = func_von_mises(tensor)
        qd.atomic_max(state.peak[i_b], value)
        if state.stress_tensor.shape[0]:
            state.stress_tensor[i_e, i_q, i_b] = tensor
            state.von_mises[i_e, i_q, i_b] = qd.Vector([value])
    for i_e, i_q, i_b in qd.ndrange(state.membrane_resultant.shape[0], 4, state.active.shape[0]):
        membrane, bending = qd.Vector.zero(gs.qd_float, 6), qd.Vector.zero(gs.qd_float, 6)
        shear = qd.Vector.zero(gs.qd_float, 3)
        for i_g in range(parts.active.shape[0] // state.active.shape[0]):
            slot = i_g * state.active.shape[0] + i_b
            if parts.valid[slot]:
                membrane += parts.membrane_resultant[i_e, i_q, slot]
                bending += parts.bending_resultant[i_e, i_q, slot]
                shear += parts.transverse_resultant[i_e, i_q, slot]
        state.membrane_resultant[i_e, i_q, i_b] = membrane
        state.bending_resultant[i_e, i_q, i_b] = bending
        state.transverse_resultant[i_e, i_q, i_b] = shear
    for i_b in range(state.active.shape[0]):
        state.step_peak[i_b] = qd.max(state.step_peak[i_b], state.peak[i_b])
