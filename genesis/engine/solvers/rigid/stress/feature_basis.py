"""Build coarse and local displacement spaces on an unchanged mother discretization."""

from typing import NamedTuple

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT
from genesis.utils.misc import qd_to_numpy

from .data import StressInfo
from .model import StressModel
from .shell_operators import ShellInfo


class FeatureMap(NamedTuple):
    """Store a block compressed sparse row (CSR) displacement map and its mother scalar constraints."""

    row_start: np.ndarray
    columns: np.ndarray
    values: qd.Tensor
    n_reduced_blocks: int
    reduced_free: np.ndarray


class FeatureBasis:
    """Construct a shared coarse prolongation and independent local nodal enrichments.

    ``shell_parents`` gives up to three physical mother parent nodes per shell node, padded with -1. Cardinal
    self-parent rows identify the original coarse nodes. Parents receive equal weights. Volume parents come from
    the mother quadratic tetrahedra. ``fixed_nodes`` promotes additional physical nodes into the shared coarse set.
    The mother's scalar constraints remain identical in every feature. ``shell_bending`` adds an experimental
    normal displacement correction that reproduces constant-curvature bending on planar triangle macroelements.
    """

    def __init__(
        self,
        model: StressModel,
        shell_parents: np.ndarray | None = None,
        fixed_nodes: np.ndarray | None = None,
        shell_bending: bool = False,
    ):
        self.mother_info = model.info
        self.blocks_per_node = 2 if model.n_shell_elements else 1
        if shell_bending and self.blocks_per_node != 2:
            gs.raise_exception("The shell bending feature correction requires a shell mother model.")
        n_blocks = self.mother_info.vertices.shape[0]
        n_nodes = n_blocks // self.blocks_per_node
        assert n_nodes > 0 and n_blocks == self.blocks_per_node * n_nodes
        self.mother_free = qd_to_numpy(self.mother_info.is_free, transpose=True, copy=True)

        if self.blocks_per_node == 2:
            if shell_parents is None:
                gs.raise_exception("Shell feature spaces require physical mother parent-node indices.")
            if not isinstance(shell_parents, np.ndarray):
                gs.raise_exception("Feature basis indices must be NumPy topology arrays.")
            if shell_parents.shape != (n_nodes, 3) or not np.issubdtype(shell_parents.dtype, np.integer):
                gs.raise_exception("Shell parents must be an integer (n_physical_nodes, 3) array.")
            if (shell_parents < -1).any() or (shell_parents >= n_nodes).any():
                gs.raise_exception("Shell parents must contain in-range physical mother nodes or -1 padding.")
            self.parent_nodes = shell_parents.astype(gs.np_int, copy=True)
            parent_count = np.count_nonzero(self.parent_nodes >= 0, axis=1)
            if (parent_count == 0).any():
                gs.raise_exception("Every shell node must have at least one coarse parent.")
            sorted_parents = np.sort(self.parent_nodes, axis=1)
            if ((sorted_parents[:, 1:] == sorted_parents[:, :-1]) & (sorted_parents[:, 1:] >= 0)).any():
                gs.raise_exception("Shell parent rows must contain distinct physical nodes.")
            original_coarse = np.flatnonzero(
                (parent_count == 1) & (self.parent_nodes.max(axis=1) == np.arange(n_nodes))
            )
            is_original_coarse = np.zeros(n_nodes, dtype=gs.np_bool)
            is_original_coarse[original_coarse] = True
            if not is_original_coarse[self.parent_nodes[self.parent_nodes >= 0]].all():
                gs.raise_exception("Shell parents must reference the original cardinal coarse nodes.")
        else:
            if shell_parents is not None:
                gs.raise_exception("Volume feature spaces derive their parents from the mother tetrahedra.")
            elements = qd_to_numpy(self.mother_info.elements)
            edges = qd_to_numpy(self.mother_info.edges)
            assert elements.ndim == 2 and elements.shape[1] == 10 and len(elements) > 0
            assert edges.shape == (6, 2)
            original_coarse = np.unique(elements[:, :4])
            n_vertices = original_coarse[-1] + 1
            assert np.array_equal(original_coarse, np.arange(n_vertices))
            edge_nodes = elements[:, 4:].reshape((6 * len(elements),))
            edge_vertices = np.sort(elements[:, edges].reshape((6 * len(elements), 2)), axis=1)
            edge_rows = np.unique(np.column_stack((edge_nodes, edge_vertices)), axis=0)
            assert np.array_equal(edge_rows[:, 0], np.arange(n_vertices, n_nodes))
            self.parent_nodes = np.full((n_nodes, 3), -1, dtype=gs.np_int)
            self.parent_nodes[:n_vertices, 0] = original_coarse
            self.parent_nodes[edge_rows[:, 0], :2] = edge_rows[:, 1:]

        if fixed_nodes is None:
            fixed_nodes = np.empty(0, dtype=gs.np_int)
        else:
            if not isinstance(fixed_nodes, np.ndarray):
                gs.raise_exception("Feature basis indices must be NumPy topology arrays.")
            if fixed_nodes.ndim != 1 or not np.issubdtype(fixed_nodes.dtype, np.integer):
                gs.raise_exception("Fixed feature nodes must be a one-dimensional integer array.")
            if ((fixed_nodes < 0) | (fixed_nodes >= n_nodes)).any():
                gs.raise_exception("Fixed feature nodes must be in-range physical mother nodes.")
        pinned_nodes = np.flatnonzero(
            np.any(self.mother_free.reshape((n_nodes, self.blocks_per_node, 3)) == 0, axis=(1, 2))
        )
        self.coarse_nodes = np.union1d(np.union1d(original_coarse, pinned_nodes), fixed_nodes).astype(
            gs.np_int, copy=False
        )
        self.coarse_blocks = (
            self.blocks_per_node * self.coarse_nodes[:, None] + np.arange(self.blocks_per_node)
        ).reshape((self.blocks_per_node * len(self.coarse_nodes),))
        self.parent_nodes[self.coarse_nodes] = -1
        self.parent_nodes[self.coarse_nodes, 0] = self.coarse_nodes
        self.coarse_index = np.full(n_nodes, -1, dtype=gs.np_int)
        self.coarse_index[self.coarse_nodes] = np.arange(len(self.coarse_nodes))

        parent_rows, parent_slots = np.nonzero(self.parent_nodes >= 0)
        parent_columns = self.coarse_index[self.parent_nodes[parent_rows, parent_slots]]
        if self.blocks_per_node == 2:
            has_offset = self.coarse_index[parent_rows] < 0
            offset_rows = parent_rows[has_offset]
            offset_columns = parent_columns[has_offset]
            rows = np.concatenate((2 * parent_rows, 2 * parent_rows + 1, 2 * offset_rows))
            columns = np.concatenate((2 * parent_columns, 2 * parent_columns + 1, 2 * offset_columns + 1))
            is_offset = np.concatenate(
                (np.zeros(2 * len(parent_rows), dtype=gs.np_int), np.ones(len(offset_rows), dtype=gs.np_int))
            )
        else:
            rows = parent_rows
            columns = parent_columns
            is_offset = np.zeros(len(rows), dtype=gs.np_int)
        order = np.lexsort((columns, rows))
        self.rows = rows[order].astype(gs.np_int, copy=False)
        row_start = np.r_[0, np.cumsum(np.bincount(self.rows, minlength=n_blocks))].astype(gs.np_int, copy=False)
        columns = columns[order].astype(gs.np_int, copy=False)
        values = V_MAT(3, 3, dtype=gs.qd_float, shape=(len(columns),))
        kernel_coarse_values(
            self.parent_nodes,
            self.rows,
            is_offset[order],
            values,
            self.mother_info,
            self.blocks_per_node,
        )
        if shell_bending:
            entry_parents = np.full(len(rows), -1, dtype=gs.np_int)
            entry_parents[2 * len(parent_rows) :] = self.parent_nodes[parent_rows[has_offset], parent_slots[has_offset]]
            kernel_shell_bending_values(
                self.parent_nodes, self.rows, entry_parents[order], values, self.mother_info, model.shell
            )
        self.transform = FeatureMap(
            row_start, columns, values, len(self.coarse_blocks), self.mother_free[self.coarse_blocks].copy()
        )

    def feature(self, local_nodes: np.ndarray) -> FeatureMap:
        """Append independent displacement coordinates at distinct noncoarse physical mother nodes.

        Coarse cardinal rows and disjoint local identity columns make the combined map full column rank. Shell
        nodes append adjacent translation and rotation blocks. Repeated nodes and coarse nodes are removed.
        """
        if not isinstance(local_nodes, np.ndarray):
            gs.raise_exception("Feature basis indices must be NumPy topology arrays.")
        if local_nodes.ndim != 1 or not np.issubdtype(local_nodes.dtype, np.integer):
            gs.raise_exception("Local feature nodes must be a one-dimensional integer array.")
        if ((local_nodes < 0) | (local_nodes >= len(self.parent_nodes))).any():
            gs.raise_exception("Local feature nodes must be in-range physical mother nodes.")
        local_nodes = np.unique(local_nodes[self.coarse_index[local_nodes] < 0])
        if not len(local_nodes):
            return self.transform
        local_blocks = (self.blocks_per_node * local_nodes[:, None] + np.arange(self.blocks_per_node)).reshape(
            (self.blocks_per_node * len(local_nodes),)
        )
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
        kernel_feature_values(source_entries[order], values, self.transform.values)
        return FeatureMap(
            row_start,
            columns,
            values,
            len(self.coarse_blocks) + len(local_blocks),
            np.concatenate((self.transform.reduced_free, self.mother_free[local_blocks])),
        )


@qd.kernel
def kernel_coarse_values(
    parent_nodes: qd.types.ndarray(),
    rows: qd.types.ndarray(),
    is_offset: qd.types.ndarray(),
    values: qd.Tensor,
    stress_info: StressInfo,
    blocks_per_node: int,
):
    for i_e in range(values.shape[0]):
        i_n = rows[i_e] // blocks_per_node
        n_parents = 0
        for i_p in qd.static(range(3)):
            if parent_nodes[i_n, i_p] >= 0:
                n_parents += 1
        weight = 1.0 / n_parents
        value = weight * qd.Matrix.identity(gs.qd_float, 3)
        if is_offset[i_e]:
            parent_position = qd.Vector.zero(gs.qd_float, 3)
            for i_p in qd.static(range(3)):
                i_a = parent_nodes[i_n, i_p]
                if i_a >= 0:
                    parent_position += weight * stress_info.vertices[blocks_per_node * i_a]
            delta = stress_info.vertices[blocks_per_node * i_n] - parent_position
            # Rotation about the parent mean reproduces rigid motion at the unchanged mother coordinates.
            value = weight * qd.Matrix(
                [
                    [0.0, delta[2], -delta[1]],
                    [-delta[2], 0.0, delta[0]],
                    [delta[1], -delta[0], 0.0],
                ]
            )
        values[i_e] = value


@qd.kernel
def kernel_shell_bending_values(
    parent_nodes: qd.types.ndarray(),
    rows: qd.types.ndarray(),
    entry_parents: qd.types.ndarray(),
    values: qd.Tensor,
    stress_info: StressInfo,
    shell_info: ShellInfo,
):
    for i_e in range(values.shape[0]):
        i_parent = entry_parents[i_e]
        if i_parent >= 0:
            i_n = rows[i_e] // 2
            parents = qd.Vector([parent_nodes.shape[0], parent_nodes.shape[0], parent_nodes.shape[0]])
            n_parents = 0
            for i_p in qd.static(range(3)):
                i_a = parent_nodes[i_n, i_p]
                if i_a >= 0:
                    parents[n_parents] = i_a
                    n_parents += 1
            for a, b in qd.static(((0, 1), (1, 2), (0, 1))):
                if parents[a] > parents[b]:
                    i_swap = parents[a]
                    parents[a] = parents[b]
                    parents[b] = i_swap
            if n_parents > 1:
                i_a, i_b = parents[0], parents[1]
                x_a, x_b = stress_info.vertices[2 * i_a], stress_info.vertices[2 * i_b]
                normal = shell_info.directors[i_n]
                # Normal projection removes tangential displacement from drilling rotation gradients.
                # Quadratic midpoint and centroid interpolation errors require edge pair weights 1/8 and 1/18.
                first = normal.outer_product((x_b - x_a).cross(normal)) / 8.0
                second = -first
                if n_parents == 3:
                    i_c = parents[2]
                    x_c = stress_info.vertices[2 * i_c]
                    first = normal.outer_product(((x_b - x_a) + (x_c - x_a)).cross(normal)) / 18.0
                    second = normal.outer_product(((x_a - x_b) + (x_c - x_b)).cross(normal)) / 18.0
                # Parent columns follow physical-node order. Their negative sum cancels constant rotations.
                correction = -(first + second)
                if i_parent == i_a:
                    correction = first
                elif i_parent == i_b:
                    correction = second
                values[i_e] += correction


@qd.kernel
def kernel_feature_values(source_entries: qd.types.ndarray(), values: qd.Tensor, coarse_values: qd.Tensor):
    for i_e in range(values.shape[0]):
        i_source = source_entries[i_e]
        value = qd.Matrix.identity(gs.qd_float, 3)
        if i_source >= 0:
            value = coarse_values[i_source]
        values[i_e] = value
