"""Native free-link compliance, contact-basis factorization and parallel-pad cold starts."""

from dataclasses import dataclass

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.solve import kernel_direct_init, kernel_full_residual
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy

from .native_pad_newton import PadInfo, PadState


@dataclass(frozen=True)
class PadComplianceInfo:
    nodes: qd.Tensor
    directions: qd.Tensor
    matrix: qd.Tensor


class NativePadCompliance:
    """Condense supplied nodal directions with native force balance and mass-gauge alignment."""

    def __init__(self, model: StressModel, nodes: np.ndarray, directions: np.ndarray, young: float):
        assert model.factor is not None and young > 0
        assert nodes.ndim == 1 and directions.shape == (len(nodes), 3)
        assert len(nodes) and nodes.min() >= 0 and nodes.max() < model.info.vertices.shape[0]
        assert np.isfinite(directions).all()
        self.info = PadComplianceInfo(
            nodes=V(dtype=gs.qd_int, shape=nodes.shape),
            directions=V_VEC(3, dtype=gs.qd_float, shape=(len(nodes),)),
            matrix=V(dtype=gs.qd_float, shape=(len(nodes), len(nodes))),
        )
        self.info.nodes.from_numpy(nodes.astype(gs.np_int, copy=False))
        self.info.directions.from_numpy(directions)
        temporary = model.create_state(len(nodes))
        kernel_compliance_loads(temporary, model.info, self.info)
        kernel_direct_init(temporary)
        model.factor.solve(young, temporary, model.info)
        kernel_full_residual(young, 1e-7, 1e-12, temporary, model.info, False)
        if not qd_to_numpy(temporary.valid, transpose=True).all():
            raise RuntimeError("Native compliance unit loads fail complete elastic equilibrium.")
        kernel_compliance_store(temporary, model.info, self.info)
        self.reciprocity_error = kernel_reciprocity(self.info.matrix)
        if not self.reciprocity_error <= 1e-7:
            raise RuntimeError("Native compliance fails reciprocity.")


@dataclass(frozen=True)
class PadBasisWorkspace:
    compliance: qd.Tensor
    lower: qd.Tensor
    row_start: qd.Tensor
    columns: qd.Tensor
    weights: qd.Tensor
    rigid: qd.Tensor
    valid: qd.Tensor


class NativePadBasis:
    """Factor supplied compliance and assemble the sparse work-conjugate map in Quadrants."""

    def __init__(self, compliance: qd.Tensor, interpolation: np.ndarray, rigid: np.ndarray):
        n_nodes = compliance.shape[0]
        assert compliance.shape == (n_nodes, n_nodes)
        assert interpolation.ndim == 2 and interpolation.shape[1] == n_nodes
        assert rigid.ndim == 2 and rigid.shape[0] == n_nodes
        assert np.isfinite(interpolation).all() and np.isfinite(rigid).all()
        rows, columns = np.nonzero(interpolation)
        starts = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(interpolation)))]
        self.workspace = PadBasisWorkspace(
            compliance=compliance,
            lower=V(dtype=gs.qd_float, shape=(n_nodes, n_nodes)),
            row_start=V(dtype=gs.qd_int, shape=starts.shape),
            columns=V(dtype=gs.qd_int, shape=columns.shape),
            weights=V(dtype=gs.qd_float, shape=columns.shape),
            rigid=V(dtype=gs.qd_float, shape=rigid.shape),
            valid=V(dtype=gs.qd_bool, shape=()),
        )
        self.workspace.row_start.from_numpy(starts.astype(gs.np_int, copy=False))
        self.workspace.columns.from_numpy(columns.astype(gs.np_int, copy=False))
        self.workspace.weights.from_numpy(interpolation[rows, columns])
        self.workspace.rigid.from_numpy(rigid)
        self.workspace.lower.fill(0)
        self.workspace.valid.fill(True)
        reciprocity = kernel_reciprocity(compliance)
        if not reciprocity <= 1e-7:
            raise RuntimeError("Supplied compliance fails reciprocity.")
        for column in range(n_nodes):
            kernel_cholesky_column(column, self.workspace)
        if not qd_to_numpy(self.workspace.valid, transpose=True):
            raise RuntimeError("Contact compliance has a nonpositive or nonfinite Cholesky pivot.")
        self.info = PadInfo(
            mapping=V(dtype=gs.qd_float, shape=(len(interpolation), n_nodes + rigid.shape[1])),
            elastic=V(dtype=gs.qd_float, shape=(n_nodes + rigid.shape[1],)),
        )
        kernel_pad_mapping(self.workspace, self.info)


@qd.kernel
def kernel_compliance_loads(stress_state: StressState, stress_info: StressInfo, compliance_info: PadComplianceInfo):
    for i_b in range(compliance_info.nodes.shape[0]):
        node = compliance_info.nodes[i_b]
        stress_state.wrench[i_b] = stress_info.modes[node].transpose() @ compliance_info.directions[i_b]
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], compliance_info.nodes.shape[0]):
        node = compliance_info.nodes[i_b]
        value = -stress_info.relief[i_n] @ stress_state.wrench[i_b]
        if i_n == node:
            value += compliance_info.directions[i_b]
        stress_state.rhs[i_n, i_b] = value


@qd.kernel
def kernel_compliance_store(stress_state: StressState, stress_info: StressInfo, compliance_info: PadComplianceInfo):
    for i_b in range(compliance_info.nodes.shape[0]):
        stress_state.wrench[i_b] = qd.Vector.zero(gs.qd_float, 6)
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], compliance_info.nodes.shape[0]):
        moment = stress_info.mass_modes[i_n].transpose() @ stress_state.displacement[i_n, i_b]
        for a in qd.static(range(6)):
            qd.atomic_add(stress_state.wrench[i_b][a], moment[a])
    for i, j in qd.ndrange(compliance_info.nodes.shape[0], compliance_info.nodes.shape[0]):
        node = compliance_info.nodes[i]
        coordinate = stress_info.gram_inverse[None] @ stress_state.wrench[j]
        displacement = stress_state.displacement[node, j] - stress_info.modes[node] @ coordinate
        compliance_info.matrix[i, j] = compliance_info.directions[i].dot(displacement)


@qd.kernel
def kernel_reciprocity(matrix: qd.Tensor) -> float:
    error, norm = 0.0, 0.0
    for i, j in qd.ndrange(matrix.shape[0], matrix.shape[1]):
        error += (matrix[i, j] - matrix[j, i]) ** 2
        norm += matrix[i, j] ** 2
    return qd.sqrt(error / norm)


@qd.kernel
def kernel_cholesky_column(column: int, basis: PadBasisWorkspace):
    for _ in range(1):
        value = basis.compliance[column, column]
        for k in range(column):
            value -= basis.lower[column, k] ** 2
        if value > 0 and value < float("inf") and basis.valid[None]:
            basis.lower[column, column] = qd.sqrt(value)
        else:
            basis.valid[None] = False
    for row in range(column + 1, basis.lower.shape[0]):
        if basis.valid[None]:
            value = 0.5 * (basis.compliance[row, column] + basis.compliance[column, row])
            for k in range(column):
                value -= basis.lower[row, k] * basis.lower[column, k]
            basis.lower[row, column] = value / basis.lower[column, column]


@qd.kernel
def kernel_pad_mapping(basis: PadBasisWorkspace, pad_info: PadInfo):
    for j in range(pad_info.elastic.shape[0]):
        pad_info.elastic[j] = gs.qd_float(j < basis.lower.shape[0])
    for i, j in qd.ndrange(pad_info.mapping.shape[0], pad_info.mapping.shape[1]):
        value = 0.0
        for entry in range(basis.row_start[i], basis.row_start[i + 1]):
            node = basis.columns[entry]
            coefficient = 0.0
            if j < basis.lower.shape[0]:
                coefficient = basis.lower[node, j]
            else:
                coefficient = basis.rigid[node, j - basis.lower.shape[0]]
            value += basis.weights[entry] * coefficient
        pad_info.mapping[i, j] = value


@qd.kernel
def kernel_parallel_pad_cold_start(requested: qd.types.ndarray(), pad_state: PadState, pad_info: PadInfo):
    """Start both opposing pads in compression without a CPU equilibrium solve."""
    for i_b in range(pad_state.status.shape[0]):
        if requested[i_b]:
            top_gap, bottom_gap, top_weight, bottom_weight = 0.0, 0.0, 0.0, 0.0
            n_dofs = pad_state.coordinate.shape[0]
            for i in range(pad_state.gap.shape[0]):
                if pad_info.mapping[i, n_dofs - 1] > 0.5:
                    top_gap = qd.max(top_gap, pad_state.gap[i, i_b])
                    top_weight += pad_state.weight[i, i_b]
                else:
                    bottom_gap = qd.max(bottom_gap, pad_state.gap[i, i_b])
                    bottom_weight += pad_state.weight[i, i_b]
            for j in range(n_dofs):
                pad_state.coordinate[j, i_b] = 0.0
            if top_weight > 0 and bottom_weight > 0:
                target = pad_state.linear[n_dofs - 1, i_b]
                vertical = -bottom_gap - target / bottom_weight
                pad_state.coordinate[n_dofs - 4, i_b] = vertical
                pad_state.coordinate[n_dofs - 1, i_b] = -top_gap + vertical - target / top_weight
            else:
                pad_state.coordinate[n_dofs - 1, i_b] = float("nan")
