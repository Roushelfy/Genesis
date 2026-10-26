"""Native Galerkin operators for sparse common-mother feature transforms."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, DataKind, V
from genesis.utils.misc import qd_to_numpy

from .data import StressInfo, allocate_stress_info
from .factor import StressFactor
from .operators import kernel_diagonal


@dataclass(frozen=True)
class FeatureTransformInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    row_start: qd.Tensor
    columns: qd.Tensor
    values: qd.Tensor
    column_start: qd.Tensor
    rows: qd.Tensor
    transpose_entries: qd.Tensor


@dataclass(frozen=True)
class FeatureAssemblyInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    row_start: qd.Tensor
    mother_entries: qd.Tensor
    left_entries: qd.Tensor
    right_entries: qd.Tensor
    transpose_entries: qd.Tensor


class FeatureOperator:
    """Project one mother's stiffness into a sparse three-component block space.

    ``row_start``, ``columns`` and ``coefficients`` describe the block CSR
    transform from reduced coordinates to mother coordinates. Both pinned
    mother rows and pinned reduced columns are zero in the stored transform.
    ``info.stiffness`` retains the unscaled, unit-Young-modulus Galerkin
    operator; ``factor`` applies symmetric diagonal equilibration separately.
    Geometry, mass and inertia relief remain properties of the mother model.
    """

    def __init__(
        self,
        mother_info: StressInfo,
        row_start: np.ndarray,
        columns: np.ndarray,
        coefficients: qd.Tensor,
        n_reduced_blocks: int,
        reduced_free: np.ndarray,
    ):
        n_mother_blocks = mother_info.vertices.shape[0]
        if (
            row_start.ndim != 1
            or row_start.shape[0] != n_mother_blocks + 1
            or not np.issubdtype(row_start.dtype, np.integer)
            or row_start[0] != 0
            or (np.diff(row_start) < 0).any()
        ):
            gs.raise_exception("Feature transform row_start must be a valid mother-block CSR row pointer.")
        if (
            n_reduced_blocks < 1
            or columns.ndim != 1
            or not np.issubdtype(columns.dtype, np.integer)
            or row_start[-1] != len(columns)
            or not len(columns)
            or columns.min() < 0
            or columns.max() >= n_reduced_blocks
        ):
            gs.raise_exception("Feature transform columns must contain in-range reduced block indices.")
        if coefficients.shape != (len(columns),) or coefficients.element_shape != (3, 3):
            gs.raise_exception("Feature transform coefficients must contain one 3-by-3 matrix per CSR entry.")
        if (
            reduced_free.shape != (n_reduced_blocks, 3)
            or not np.issubdtype(reduced_free.dtype, np.integer)
            or ((reduced_free != 0) & (reduced_free != 1)).any()
        ):
            gs.raise_exception("Feature reduced_free must contain one three-component 0/1 mask per reduced block.")

        mother_row_start = qd_to_numpy(mother_info.row_start, transpose=True)
        mother_columns = qd_to_numpy(mother_info.columns, transpose=True)
        mother_rows = np.repeat(np.arange(n_mother_blocks), np.diff(mother_row_start))
        transform_counts = np.diff(row_start)
        left_counts, right_counts = transform_counts[mother_rows], transform_counts[mother_columns]
        term_counts = left_counts * right_counts
        mother_entries = np.repeat(np.arange(len(mother_columns)), term_counts)
        term_starts = np.cumsum(term_counts) - term_counts
        offsets = np.arange(len(mother_entries)) - np.repeat(term_starts, term_counts)
        right_counts = np.repeat(right_counts, term_counts)
        left_entries = np.repeat(row_start[mother_rows], term_counts) + offsets // right_counts
        right_entries = np.repeat(row_start[mother_columns], term_counts) + offsets % right_counts
        entries, inverse = np.unique(
            np.column_stack((columns[left_entries], columns[right_entries])), axis=0, return_inverse=True
        )
        if not len(entries):
            gs.raise_exception("Feature transform has no stiffness contributions from the mother operator.")
        reduced_row_start = np.r_[0, np.cumsum(np.bincount(entries[:, 0], minlength=n_reduced_blocks))]
        assembly_row_start = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(entries)))]
        assembly_order = np.argsort(inverse, kind="stable")
        transpose_entries = np.lexsort((entries[:, 0], entries[:, 1]))
        assert np.array_equal(entries[transpose_entries], entries[:, ::-1])
        transform_rows = np.repeat(np.arange(n_mother_blocks), transform_counts)
        transpose_order = np.lexsort((transform_rows, columns))
        column_start = np.r_[0, np.cumsum(np.bincount(columns, minlength=n_reduced_blocks))]

        self.n_transform_entries = len(columns)
        self.n_assembly_terms = len(mother_entries)
        self.info = allocate_stress_info(n_reduced_blocks, 0, 0, len(entries), 0)
        self.info.vertices.fill(0)
        self.info.pins.fill(-1)
        self.info.row_start.from_numpy(reduced_row_start.astype(gs.np_int, copy=False))
        self.info.columns.from_numpy(entries[:, 1].astype(gs.np_int, copy=False))
        self.info.is_free.from_numpy(reduced_free.astype(gs.np_int, copy=False))
        self.transform = FeatureTransformInfo(
            row_start=V(dtype=gs.qd_int, shape=(n_mother_blocks + 1,)),
            columns=V(dtype=gs.qd_int, shape=(len(columns),)),
            values=V_MAT(3, 3, dtype=gs.qd_float, shape=(len(columns),)),
            column_start=V(dtype=gs.qd_int, shape=(n_reduced_blocks + 1,)),
            rows=V(dtype=gs.qd_int, shape=(len(columns),)),
            transpose_entries=V(dtype=gs.qd_int, shape=(len(columns),)),
        )
        self.transform.row_start.from_numpy(row_start.astype(gs.np_int, copy=False))
        self.transform.columns.from_numpy(columns.astype(gs.np_int, copy=False))
        self.transform.column_start.from_numpy(column_start.astype(gs.np_int, copy=False))
        self.transform.rows.from_numpy(transform_rows[transpose_order].astype(gs.np_int, copy=False))
        self.transform.transpose_entries.from_numpy(transpose_order.astype(gs.np_int, copy=False))
        assembly = FeatureAssemblyInfo(
            row_start=V(dtype=gs.qd_int, shape=(len(entries) + 1,)),
            mother_entries=V(dtype=gs.qd_int, shape=(len(mother_entries),)),
            left_entries=V(dtype=gs.qd_int, shape=(len(mother_entries),)),
            right_entries=V(dtype=gs.qd_int, shape=(len(mother_entries),)),
            transpose_entries=V(dtype=gs.qd_int, shape=(len(entries),)),
        )
        assembly.row_start.from_numpy(assembly_row_start.astype(gs.np_int, copy=False))
        assembly.mother_entries.from_numpy(mother_entries[assembly_order].astype(gs.np_int, copy=False))
        assembly.left_entries.from_numpy(left_entries[assembly_order].astype(gs.np_int, copy=False))
        assembly.right_entries.from_numpy(right_entries[assembly_order].astype(gs.np_int, copy=False))
        assembly.transpose_entries.from_numpy(transpose_entries.astype(gs.np_int, copy=False))
        kernel_feature_transform(coefficients, mother_info, self.info, self.transform)
        kernel_feature_assemble(mother_info, self.info, self.transform, assembly)
        kernel_feature_symmetrize(self.info, assembly)
        kernel_diagonal(self.info)
        self.factor = StressFactor(self.info, scaling=True)

    def project(self, mother_vectors: qd.Tensor, reduced_vectors: qd.Tensor) -> None:
        """Overwrite batched reduced vectors with the transpose transform product."""
        if (
            len(mother_vectors.shape) != 2
            or len(reduced_vectors.shape) != 2
            or mother_vectors.shape[0] + 1 != self.transform.row_start.shape[0]
            or reduced_vectors.shape[0] != self.info.vertices.shape[0]
            or mother_vectors.shape[1] != reduced_vectors.shape[1]
        ):
            gs.raise_exception("Feature projection requires matching (mother/reduced blocks, environments) vectors.")
        kernel_feature_project(mother_vectors, reduced_vectors, self.transform)

    def lift(self, reduced_vectors: qd.Tensor, mother_vectors: qd.Tensor, add: bool = False) -> None:
        """Lift batched reduced vectors, optionally adding to existing mother vectors."""
        if (
            len(mother_vectors.shape) != 2
            or len(reduced_vectors.shape) != 2
            or mother_vectors.shape[0] + 1 != self.transform.row_start.shape[0]
            or reduced_vectors.shape[0] != self.info.vertices.shape[0]
            or mother_vectors.shape[1] != reduced_vectors.shape[1]
        ):
            gs.raise_exception("Feature lifting requires matching (mother/reduced blocks, environments) vectors.")
        kernel_feature_lift(reduced_vectors, mother_vectors, self.transform, add)

    def residual(
        self,
        young: float,
        reduced_rhs: qd.Tensor,
        reduced_solution: qd.Tensor,
        reduced_residual: qd.Tensor,
    ) -> None:
        """Write ``young * K_i @ z - b`` in the original free reduced coordinates."""
        if (
            len(reduced_rhs.shape) != 2
            or reduced_rhs.shape != reduced_solution.shape
            or reduced_rhs.shape != reduced_residual.shape
            or reduced_rhs.shape[0] != self.info.vertices.shape[0]
        ):
            gs.raise_exception("Feature residual vectors must have matching (reduced blocks, environments) shapes.")
        kernel_feature_residual(young, reduced_rhs, reduced_solution, reduced_residual, self.info)


@qd.kernel
def kernel_feature_transform(
    coefficients: qd.Tensor, mother_info: StressInfo, reduced_info: StressInfo, transform_info: FeatureTransformInfo
):
    for i_n in range(mother_info.vertices.shape[0]):
        for i_entry in range(transform_info.row_start[i_n], transform_info.row_start[i_n + 1]):
            i_r = transform_info.columns[i_entry]
            value = coefficients[i_entry]
            for a, b in qd.static(qd.ndrange(3, 3)):
                if not mother_info.is_free[i_n][a] or not reduced_info.is_free[i_r][b]:
                    value[a, b] = 0.0
            transform_info.values[i_entry] = value


@qd.kernel
def kernel_feature_assemble(
    mother_info: StressInfo,
    reduced_info: StressInfo,
    transform_info: FeatureTransformInfo,
    assembly_info: FeatureAssemblyInfo,
):
    for i_entry in range(reduced_info.stiffness.shape[0]):
        stiffness = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i_term in range(assembly_info.row_start[i_entry], assembly_info.row_start[i_entry + 1]):
            i_mother = assembly_info.mother_entries[i_term]
            i_left = assembly_info.left_entries[i_term]
            i_right = assembly_info.right_entries[i_term]
            stiffness += (
                transform_info.values[i_left].transpose()
                @ mother_info.stiffness[i_mother]
                @ transform_info.values[i_right]
            )
        reduced_info.stiffness[i_entry] = stiffness


@qd.kernel
def kernel_feature_symmetrize(reduced_info: StressInfo, assembly_info: FeatureAssemblyInfo):
    for i_entry in range(reduced_info.stiffness.shape[0]):
        i_transpose = assembly_info.transpose_entries[i_entry]
        if i_entry <= i_transpose:
            # Transposed sums can accumulate in different orders; one owner writes both blocks.
            value = 0.5 * (reduced_info.stiffness[i_entry] + reduced_info.stiffness[i_transpose].transpose())
            reduced_info.stiffness[i_entry] = value
            reduced_info.stiffness[i_transpose] = value.transpose()


@qd.kernel(graph=True)
def kernel_feature_project(mother_vectors: qd.Tensor, reduced_vectors: qd.Tensor, transform_info: FeatureTransformInfo):
    for i_r, i_b in qd.ndrange(reduced_vectors.shape[0], reduced_vectors.shape[1]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for i_column in range(transform_info.column_start[i_r], transform_info.column_start[i_r + 1]):
            i_n = transform_info.rows[i_column]
            i_entry = transform_info.transpose_entries[i_column]
            value += transform_info.values[i_entry].transpose() @ mother_vectors[i_n, i_b]
        reduced_vectors[i_r, i_b] = value


@qd.kernel(graph=True)
def kernel_feature_lift(
    reduced_vectors: qd.Tensor, mother_vectors: qd.Tensor, transform_info: FeatureTransformInfo, add: bool
):
    for i_n, i_b in qd.ndrange(mother_vectors.shape[0], mother_vectors.shape[1]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for i_entry in range(transform_info.row_start[i_n], transform_info.row_start[i_n + 1]):
            i_r = transform_info.columns[i_entry]
            value += transform_info.values[i_entry] @ reduced_vectors[i_r, i_b]
        if add:
            value += mother_vectors[i_n, i_b]
        mother_vectors[i_n, i_b] = value


@qd.kernel(graph=True)
def kernel_feature_residual(
    young: float,
    reduced_rhs: qd.Tensor,
    reduced_solution: qd.Tensor,
    reduced_residual: qd.Tensor,
    reduced_info: StressInfo,
):
    for i_r, i_b in qd.ndrange(reduced_rhs.shape[0], reduced_rhs.shape[1]):
        value = -reduced_rhs[i_r, i_b]
        for i_entry in range(reduced_info.row_start[i_r], reduced_info.row_start[i_r + 1]):
            j_r = reduced_info.columns[i_entry]
            value += young * reduced_info.stiffness[i_entry] @ reduced_solution[j_r, i_b]
        reduced_residual[i_r, i_b] = value * reduced_info.is_free[i_r]
