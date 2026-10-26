"""Bilinear shell prolongation and nodal bubbles on an unchanged mother mesh."""

from typing import NamedTuple

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT
from genesis.utils.misc import qd_to_numpy

from .model import StressModel


class FeatureMap(NamedTuple):
    """Store a mother-to-reduced block CSR map and its scalar free-coordinate mask."""

    row_start: np.ndarray
    columns: np.ndarray
    values: qd.Tensor
    n_reduced_blocks: int
    reduced_free: np.ndarray


class BilinearBasis:
    """Construct ``P_nodes tensor I6`` from integer Q4 parent coordinates.

    Parent corners use Q4 order (0, 0), (subdiv, 0), (subdiv, subdiv), (0, subdiv).
    Each physical node contributes adjacent translation and rotation blocks. Host work constructs only the
    integer sparse graph; Quadrants computes interpolation coefficients and local identity enrichments.
    """

    def __init__(
        self,
        model: StressModel,
        coarse_nodes: np.ndarray,
        parent_corners: np.ndarray,
        parent_ij: np.ndarray,
        subdiv: int,
    ):
        self.mother_info = model.info
        n_blocks = self.mother_info.vertices.shape[0]
        n_nodes = n_blocks // 2
        if not model.n_shell_elements or n_blocks != 2 * n_nodes:
            gs.raise_exception("Bilinear feature bases require a shell mother with translation and rotation blocks.")
        if not isinstance(subdiv, (int, np.integer)) or subdiv < 1:
            gs.raise_exception("Bilinear parent subdivision must be a positive integer.")
        for name, indices, shape in (
            ("coarse_nodes", coarse_nodes, None),
            ("parent_corners", parent_corners, (n_nodes, 4)),
            ("parent_ij", parent_ij, (n_nodes, 2)),
        ):
            if (
                not isinstance(indices, np.ndarray)
                or not np.issubdtype(indices.dtype, np.integer)
                or (shape is not None and indices.shape != shape)
                or (shape is None and indices.ndim != 1)
            ):
                gs.raise_exception(f"Bilinear {name} must be an integer topology array of the expected shape.")
        if (
            not len(coarse_nodes)
            or len(np.unique(coarse_nodes)) != len(coarse_nodes)
            or (coarse_nodes < 0).any()
            or (coarse_nodes >= n_nodes).any()
        ):
            gs.raise_exception("Bilinear coarse_nodes must contain distinct in-range physical mother nodes.")
        if (parent_corners < 0).any() or (parent_corners >= n_nodes).any():
            gs.raise_exception("Bilinear parent corners must contain in-range physical mother nodes.")
        if ((parent_ij < 0) | (parent_ij > subdiv)).any():
            gs.raise_exception("Bilinear parent_ij must lie between zero and subdiv, inclusive.")
        ordered_corners = np.sort(parent_corners, axis=1)
        if (ordered_corners[:, 1:] == ordered_corners[:, :-1]).any():
            gs.raise_exception("Bilinear Q4 parent corners must be distinct within each row.")
        self.coarse_nodes = coarse_nodes.astype(gs.np_int, copy=True)
        self.coarse_index = np.full(n_nodes, -1, dtype=gs.np_int)
        self.coarse_index[self.coarse_nodes] = np.arange(len(self.coarse_nodes))
        if (self.coarse_index[parent_corners] < 0).any():
            gs.raise_exception("Bilinear parent corners must reference the supplied coarse_nodes.")
        self.parent_corners = parent_corners.astype(gs.np_int, copy=True)
        self.parent_ij = parent_ij.astype(gs.np_int, copy=True)
        self.subdiv = int(subdiv)
        self.coarse_blocks = (2 * self.coarse_nodes[:, None] + np.arange(2)).reshape(-1)
        self.mother_free = qd_to_numpy(self.mother_info.is_free, transpose=True, copy=True)
        pinned_nodes = np.flatnonzero(np.any(self.mother_free.reshape((n_nodes, 2, 3)) == 0, axis=(1, 2)))
        if (self.coarse_index[pinned_nodes] < 0).any():
            gs.raise_exception("Every pinned mother node must be a cardinal node of the bilinear coarse space.")

        # Zero weights follow solely from integer coordinates. This matches CSR zero elimination without
        # computing any physical interpolation values on the host.
        i, j = self.parent_ij[:, 0], self.parent_ij[:, 1]
        active = np.column_stack(
            ((i < subdiv) & (j < subdiv), (i > 0) & (j < subdiv), (i > 0) & (j > 0), (i < subdiv) & (j > 0))
        )
        cardinal = active[self.coarse_nodes]
        if not np.all(np.count_nonzero(cardinal, axis=1) == 1):
            gs.raise_exception("Bilinear coarse nodes must have cardinal parent coordinates.")
        cardinal_parents = self.parent_corners[self.coarse_nodes][cardinal]
        if not np.array_equal(cardinal_parents, self.coarse_nodes):
            gs.raise_exception("Bilinear coarse cardinal rows must interpolate their own physical nodes.")
        parent_rows, parent_slots = np.nonzero(active)
        parent_columns = self.coarse_index[self.parent_corners[parent_rows, parent_slots]]
        rows = np.concatenate((2 * parent_rows, 2 * parent_rows + 1))
        columns = np.concatenate((2 * parent_columns, 2 * parent_columns + 1))
        physical_rows = np.tile(parent_rows, 2)
        slots = np.tile(parent_slots, 2)
        order = np.lexsort((columns, rows))
        self.rows = rows[order].astype(gs.np_int, copy=False)
        row_start = np.r_[0, np.cumsum(np.bincount(self.rows, minlength=n_blocks))].astype(gs.np_int, copy=False)
        columns = columns[order].astype(gs.np_int, copy=False)
        values = V_MAT(3, 3, dtype=gs.qd_float, shape=(len(columns),))
        kernel_bilinear_values(
            self.parent_ij,
            physical_rows[order].astype(gs.np_int, copy=False),
            slots[order].astype(gs.np_int, copy=False),
            self.subdiv,
            values,
        )
        self.transform = FeatureMap(
            row_start, columns, values, len(self.coarse_blocks), self.mother_free[self.coarse_blocks].copy()
        )

    def feature(self, local_nodes: np.ndarray) -> FeatureMap:
        """Append six independent coordinates at each distinct noncoarse local physical node."""
        if (
            not isinstance(local_nodes, np.ndarray)
            or local_nodes.ndim != 1
            or not np.issubdtype(local_nodes.dtype, np.integer)
        ):
            gs.raise_exception("Local bilinear feature nodes must be a one-dimensional integer topology array.")
        if ((local_nodes < 0) | (local_nodes >= len(self.coarse_index))).any():
            gs.raise_exception("Local bilinear feature nodes must be in-range physical mother nodes.")
        local_nodes = np.unique(local_nodes[self.coarse_index[local_nodes] < 0])
        if not len(local_nodes):
            return self.transform
        local_blocks = (2 * local_nodes[:, None] + np.arange(2)).reshape(-1)
        rows = np.concatenate((self.rows, local_blocks))
        columns = np.concatenate((self.transform.columns, len(self.coarse_blocks) + np.arange(len(local_blocks))))
        order = np.lexsort((columns, rows))
        row_start = np.r_[0, np.cumsum(np.bincount(rows, minlength=self.mother_info.vertices.shape[0]))].astype(
            gs.np_int, copy=False
        )
        columns = columns[order].astype(gs.np_int, copy=False)
        source_entries = np.concatenate(
            (np.arange(len(self.transform.columns), dtype=gs.np_int), np.full(len(local_blocks), -1, dtype=gs.np_int))
        )
        values = V_MAT(3, 3, dtype=gs.qd_float, shape=(len(columns),))
        kernel_bilinear_feature_values(source_entries[order], values, self.transform.values)
        return FeatureMap(
            row_start,
            columns,
            values,
            len(self.coarse_blocks) + len(local_blocks),
            np.concatenate((self.transform.reduced_free, self.mother_free[local_blocks])),
        )


@qd.kernel
def kernel_bilinear_values(
    parent_ij: qd.types.ndarray(),
    physical_rows: qd.types.ndarray(),
    slots: qd.types.ndarray(),
    subdiv: int,
    values: qd.Tensor,
):
    for i_entry in range(values.shape[0]):
        i_n, slot = physical_rows[i_entry], slots[i_entry]
        xi = 2.0 * qd.cast(parent_ij[i_n, 0], gs.qd_float) / subdiv - 1.0
        eta = 2.0 * qd.cast(parent_ij[i_n, 1], gs.qd_float) / subdiv - 1.0
        signs_x = qd.Vector([-1.0, 1.0, 1.0, -1.0])
        signs_y = qd.Vector([-1.0, -1.0, 1.0, 1.0])
        weight = 0.25 * (1.0 + signs_x[slot] * xi) * (1.0 + signs_y[slot] * eta)
        values[i_entry] = weight * qd.Matrix.identity(gs.qd_float, 3)


@qd.kernel
def kernel_bilinear_feature_values(source_entries: qd.types.ndarray(), values: qd.Tensor, coarse_values: qd.Tensor):
    for i_entry in range(values.shape[0]):
        value = qd.Matrix.identity(gs.qd_float, 3)
        i_source = source_entries[i_entry]
        if i_source >= 0:
            value = coarse_values[i_source]
        values[i_entry] = value
