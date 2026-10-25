"""Complete native stress responses with unchanged displacement acceptance."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import quadrants as qd

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, DataKind, V

from .data import StressState
from .history import StressHistory
from .surface_inverse import StressSurfaceInverseInfo

if TYPE_CHECKING:
    from .model import StressModel


@dataclass(frozen=True)
class StressResponseInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    values: qd.Tensor
    bounds: qd.Tensor


@dataclass(frozen=True)
class StressResponseState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    loads: qd.Tensor
    columns: qd.Tensor
    counts: qd.Tensor
    recover_displacement: qd.Tensor
    evaluated: qd.Tensor


class StressResponseInventory:
    def __init__(self, model: "StressModel", options: RigidStressOptions):
        if model.surface_inverse is None:
            gs.raise_exception("Complete stress responses require the native FP64 surface response.")
        if options.stress_response == "screen" and (
            options.response_precision != "64" or options.response_accumulation != "64" or gs.qd_float != qd.f64
        ):
            gs.raise_exception("Conservative response screening currently requires FP64 storage and accumulation.")
        self.n_stress = model.n_output_elements * 4
        self.n_resultants = model.n_shell_elements * 4
        n_output = self.n_stress + 3 * self.n_resultants
        n_input = 3 * model.surface_inverse.info.nodes.shape[0] + 6
        n_blocks = (self.n_stress + 31) // 32
        self.capacity = next((size for size in (32, 64, 128, 256, 512, 1024) if size >= n_blocks), 0)
        required_bytes = n_output * n_input * 6 * (8 if options.response_precision == "64" else 4)
        required_bytes += n_blocks * n_input * 8
        if required_bytes > options.response_max_bytes:
            gs.raise_exception(
                f"Complete stress responses require {required_bytes} bytes, exceeding response_max_bytes."
            )
        self.info = StressResponseInfo(
            values=V_VEC(
                6, dtype=gs.qd_float if options.response_precision == "64" else qd.f32, shape=(n_output, n_input)
            ),
            bounds=V(dtype=gs.qd_float, shape=(n_blocks, n_input)),
        )
        temporary = model.create_state(n_input, "full")
        kernel_unit_displacements(options.young, temporary, model.surface_inverse.info)
        model.evaluate_stress(options, temporary)
        kernel_store(temporary, self.info)
        kernel_bounds(self.n_stress, self.info)
        qd.sync()


class StressResponse:
    def __init__(self, inventory: StressResponseInventory, n_envs: int):
        self.inventory = inventory
        n_input = inventory.info.values.shape[1]
        self.state = StressResponseState(
            loads=V(dtype=gs.qd_float, shape=(n_input, n_envs)),
            columns=V(dtype=gs.qd_int, shape=(n_input, n_envs)),
            counts=V(dtype=gs.qd_int, shape=(n_envs,)),
            recover_displacement=V(dtype=gs.qd_bool, shape=(n_envs,)),
            evaluated=V(dtype=gs.qd_int, shape=(n_envs,)),
        )

    def evaluate(
        self,
        options: RigidStressOptions,
        omega: qd.Tensor,
        state: StressState,
        model: "StressModel",
        history: StressHistory | None,
    ):
        kernel_inputs(
            omega, state, self.state, model.surface_inverse.info, history.state.hit if history is not None else None
        )
        # Corrected/predicted/invalid states retain ordinary recovery from their actual
        # displacement. H is used only for an accepted uncorrected exact response.
        model.evaluate_stress(options, state, self.state.recover_displacement)
        if (
            options.stress_response == "screen"
            and state.stress_tensor.shape[0] == 0
            and self.inventory.capacity
            and gs.backend == gs.cuda
        ):
            kernel_screen(self.inventory.n_stress, state, self.state, self.inventory.info, self.inventory.capacity)
        else:
            n_output = self.inventory.info.values.shape[0] if state.stress_tensor.shape[0] else self.inventory.n_stress
            mode = 0 if options.stress_response == "dense" else 2 if options.stress_response == "hybrid" else 1
            kernel_apply(
                self.inventory.n_stress,
                self.inventory.n_resultants,
                n_output,
                state,
                self.state,
                self.inventory.info,
                mode,
                qd.f32 if options.response_accumulation == "32" else gs.qd_float,
                options.response_block_size,
            )
        kernel_finish(state)


@qd.kernel
def kernel_unit_displacements(young: float, state: StressState, response: StressSurfaceInverseInfo):
    for i_n, i_b in qd.ndrange(state.displacement.shape[0], state.active.shape[0]):
        value = qd.Vector.zero(gs.qd_float, 3)
        if i_b < 3 * response.nodes.shape[0]:
            for a in qd.static(range(3)):
                value[a] = response.force_blocks[i_n, i_b // 3][a, i_b % 3]
        else:
            for a in qd.static(range(3)):
                value[a] = response.centrifugal[i_n][a, i_b - 3 * response.nodes.shape[0]]
        state.displacement[i_n, i_b] = value / young


@qd.kernel
def kernel_store(state: StressState, response: StressResponseInfo):
    for i_p, i_j in qd.ndrange(response.values.shape[0], response.values.shape[1]):
        n_stress = state.stress_tensor.shape[0] * 4
        value = qd.Vector.zero(gs.qd_float, 6)
        if i_p < n_stress:
            value = state.stress_tensor[i_p // 4, i_p % 4, i_j]
        else:
            n_resultants = state.membrane_resultant.shape[0] * 4
            i_field, i_q = (i_p - n_stress) // n_resultants, (i_p - n_stress) % n_resultants
            if i_field == 0:
                value = state.membrane_resultant[i_q // 4, i_q % 4, i_j]
            elif i_field == 1:
                value = state.bending_resultant[i_q // 4, i_q % 4, i_j]
            else:
                for a in qd.static(range(3)):
                    value[a] = state.transverse_resultant[i_q // 4, i_q % 4, i_j][a]
        response.values[i_p, i_j] = value


@qd.func
def func_von_mises(value: qd.types.vector(6)):
    squared = 0.5 * ((value[0] - value[1]) ** 2 + (value[1] - value[2]) ** 2 + (value[2] - value[0]) ** 2)
    squared += 3.0 * (value[3] ** 2 + value[4] ** 2 + value[5] ** 2)
    return qd.sqrt(qd.max(squared, 0.0))


@qd.kernel(graph=True)
def kernel_inputs(
    omega: qd.Tensor,
    state: StressState,
    response: StressResponseState,
    boundary: StressSurfaceInverseInfo,
    history_hits: qd.Tensor | None,
):
    for i_b in range(state.active.shape[0]):
        fallback = not state.valid[i_b] or state.corrections[i_b] > 0 or state.fallbacks[i_b] > 0
        if qd.static(history_hits is not None):
            fallback = fallback or history_hits[i_b]
        response.recover_displacement[i_b] = fallback
        response.evaluated[i_b] = 0
        count = 0
        w = omega[i_b]
        terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        if not fallback:
            for j in range(response.loads.shape[0]):
                value = gs.qd_float(0.0)
                if j < 3 * boundary.nodes.shape[0]:
                    value = state.force[boundary.nodes[j // 3], i_b][j % 3]
                else:
                    value = terms[j - 3 * boundary.nodes.shape[0]]
                response.loads[j, i_b] = value
                if value != 0.0:
                    response.columns[count, i_b] = j
                    count += 1
        response.counts[i_b] = count


@qd.kernel(graph=True)
def kernel_apply(
    n_stress: int,
    n_resultants: int,
    n_output: int,
    state: StressState,
    workspace: StressResponseState,
    response: StressResponseInfo,
    mode: qd.template(),
    accumulation: qd.template(),
    block_size: qd.template(),
):
    qd.loop_config(block_dim=block_size)
    for i_p, i_b in qd.ndrange(n_output, state.active.shape[0]):
        if not workspace.recover_displacement[i_b]:
            value = qd.Vector.zero(accumulation, 6)
            sparse = True
            if qd.static(mode == 0):
                sparse = False
            elif qd.static(mode == 2):
                sparse = workspace.counts[i_b] < response.values.shape[1] // 2
            n_columns = qd.select(sparse, workspace.counts[i_b], response.values.shape[1])
            for slot in range(n_columns):
                j = slot
                if sparse:
                    j = workspace.columns[slot, i_b]
                value += qd.cast(response.values[i_p, j], accumulation) * qd.cast(workspace.loads[j, i_b], accumulation)
            components = value.cast(gs.qd_float)
            if i_p < n_stress:
                vm = func_von_mises(components)
                qd.atomic_max(state.peak[i_b], vm)
                if state.stress_tensor.shape[0]:
                    state.stress_tensor[i_p // 4, i_p % 4, i_b] = components
                    state.von_mises[i_p // 4, i_p % 4, i_b] = qd.Vector([vm])
            else:
                i_field, i_q = (i_p - n_stress) // n_resultants, (i_p - n_stress) % n_resultants
                if i_field == 0:
                    state.membrane_resultant[i_q // 4, i_q % 4, i_b] = components
                elif i_field == 1:
                    state.bending_resultant[i_q // 4, i_q % 4, i_b] = components
                else:
                    state.transverse_resultant[i_q // 4, i_q % 4, i_b] = components[0:3]


@qd.kernel
def kernel_bounds(n_stress: int, response: StressResponseInfo):
    for i_block, j in qd.ndrange(response.bounds.shape[0], response.bounds.shape[1]):
        value, magnitude = gs.qd_float(0.0), gs.qd_float(0.0)
        for lane in range(32):
            i_p = 32 * i_block + lane
            if i_p < n_stress:
                tensor = response.values[i_p, j].cast(gs.qd_float)
                value = qd.max(value, func_von_mises(tensor))
                norm = qd.sqrt(
                    tensor[0] ** 2
                    + tensor[1] ** 2
                    + tensor[2] ** 2
                    + 2.0 * (tensor[3] ** 2 + tensor[4] ** 2 + tensor[5] ** 2)
                )
                magnitude = qd.max(magnitude, norm)
        # The tensor norm term covers cancellation/hydrostatic columns, for which
        # a relative VM margin alone is insufficient. Screening uses FP64 only.
        roundoff = (2.0 * response.values.shape[1] + 32.0) * 1.1102230246251565e-16
        allowance = 16.0 * roundoff / (1.0 - roundoff)
        response.bounds[i_block, j] = (value + allowance * magnitude) * (1.0 + 1e-10)


@qd.kernel(graph=True)
def kernel_screen(
    n_stress: int,
    state: StressState,
    workspace: StressResponseState,
    response: StressResponseInfo,
    capacity: qd.template(),
):
    qd.loop_config(block_dim=32)
    for i_thread in range(32 * state.active.shape[0]):
        lane, i_b = i_thread % 32, i_thread // 32
        sh_upper = qd.simt.block.SharedArray((capacity,), gs.qd_float)
        if not workspace.recover_displacement[i_b]:
            for chunk in range((response.bounds.shape[0] + 31) // 32):
                i_block = 32 * chunk + lane
                if i_block < response.bounds.shape[0]:
                    upper = gs.qd_float(0.0)
                    for slot in range(workspace.counts[i_b]):
                        j = workspace.columns[slot, i_b]
                        upper += qd.abs(workspace.loads[j, i_b]) * response.bounds[i_block, j]
                    sh_upper[i_block] = upper * (1.0 + 1e-10)
            qd.simt.block.sync()
            maximum, n_evaluated = gs.qd_float(0.0), 0
            for _ in range(response.bounds.shape[0]):
                best, chosen = gs.qd_float(-1.0), response.bounds.shape[0]
                for chunk in range((response.bounds.shape[0] + 31) // 32):
                    i_block = 32 * chunk + lane
                    if i_block < response.bounds.shape[0] and sh_upper[i_block] > best:
                        best, chosen = sh_upper[i_block], i_block
                upper = qd.simt.subgroup.reduce_all_max(best)
                chosen = qd.simt.subgroup.reduce_all_min(qd.select(best >= upper, chosen, response.bounds.shape[0]))
                if upper <= maximum:
                    break
                i_p = chosen * 32 + lane
                value = qd.Vector.zero(gs.qd_float, 6)
                if i_p < n_stress:
                    for slot in range(workspace.counts[i_b]):
                        j = workspace.columns[slot, i_b]
                        value += response.values[i_p, j] * workspace.loads[j, i_b]
                maximum = qd.max(maximum, qd.simt.subgroup.reduce_all_max(func_von_mises(value)))
                n_evaluated += 1
                if lane == 0:
                    sh_upper[chosen] = -1.0
                qd.simt.block.sync()
            if lane == 0:
                state.peak[i_b], workspace.evaluated[i_b] = maximum, n_evaluated


@qd.kernel(graph=True)
def kernel_finish(state: StressState):
    for i_b in range(state.active.shape[0]):
        state.step_peak[i_b] = qd.max(state.step_peak[i_b], state.peak[i_b])
