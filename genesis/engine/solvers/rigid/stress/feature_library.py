"""Prepare angular macro supports and complete-incident shell bubble spaces."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V
from genesis.utils.misc import qd_to_numpy

from .bilinear_basis import BilinearBasis


@dataclass(frozen=True)
class FeatureLibraryInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    directions: qd.Tensor
    coarse_directions: qd.Tensor
    centers: qd.Tensor
    macro_quads: qd.Tensor
    macro_centers: qd.Tensor
    macro_radii: qd.Tensor
    macro_angles: qd.Tensor
    node_macro_row_start: qd.Tensor
    node_macro_columns: qd.Tensor
    macro_neighbor_row_start: qd.Tensor
    macro_neighbor_columns: qd.Tensor
    active_macros: qd.Tensor
    local_mask: qd.Tensor
    is_valid: qd.Tensor


def _validate_csr(name: str, row_start: np.ndarray, columns: np.ndarray, n_rows: int, n_columns: int) -> None:
    if (
        not isinstance(row_start, np.ndarray)
        or not isinstance(columns, np.ndarray)
        or row_start.shape != (n_rows + 1,)
        or columns.ndim != 1
        or not np.issubdtype(row_start.dtype, np.integer)
        or not np.issubdtype(columns.dtype, np.integer)
        or row_start[0] != 0
        or row_start[-1] != len(columns)
        or (np.diff(row_start) < 1).any()
        or (columns < 0).any()
        or (columns >= n_columns).any()
    ):
        gs.raise_exception(f"Feature library {name} must be an in-range integer CSR graph with nonempty rows.")


class FeatureLibrary:
    """Build native macro geometry, angular supports, halo expansion and local bubble maps.

    Direction arguments are Quadrants vector tensors. ``macro_quads`` indexes the supplied coarse direction
    ordering. Both CSR graphs are integer topology: physical node to incident macros, and macro to neighbors
    sharing at least one coarse vertex. Frozen support/core angles are physical configuration values.
    """

    def __init__(
        self,
        basis: BilinearBasis,
        directions: qd.Tensor,
        coarse_directions: qd.Tensor,
        macro_quads: np.ndarray,
        node_macro_row_start: np.ndarray,
        node_macro_columns: np.ndarray,
        macro_neighbor_row_start: np.ndarray,
        macro_neighbor_columns: np.ndarray,
        *,
        core_angle: float,
        support_angle: float,
        halo: int = 1,
        centers: qd.Tensor | None = None,
    ):
        n_nodes, n_coarse = len(basis.coarse_index), len(basis.coarse_nodes)
        if directions.shape != (n_nodes,) or directions.element_shape != (3,):
            gs.raise_exception("Feature library directions must provide one native three-vector per physical node.")
        if coarse_directions.shape != (n_coarse,) or coarse_directions.element_shape != (3,):
            gs.raise_exception("Feature library coarse directions must follow the bilinear coarse_nodes ordering.")
        if (
            not isinstance(macro_quads, np.ndarray)
            or macro_quads.ndim != 2
            or macro_quads.shape[1] != 4
            or not len(macro_quads)
            or not np.issubdtype(macro_quads.dtype, np.integer)
            or (macro_quads < 0).any()
            or (macro_quads >= n_coarse).any()
        ):
            gs.raise_exception("Feature macro_quads must be a nonempty integer array of coarse corner indices.")
        if core_angle <= 0 or support_angle <= 0:
            gs.raise_exception("Feature core and support angles must be positive physical configuration values.")
        if not isinstance(halo, (int, np.integer)) or halo < 0:
            gs.raise_exception("Feature macro halo must be a nonnegative integer.")
        if centers is not None and (len(centers.shape) != 1 or not centers.shape[0] or centers.element_shape != (3,)):
            gs.raise_exception("Explicit feature centers must be a nonempty native vector tensor.")
        n_macros = len(macro_quads)
        n_features = 12 if centers is None else centers.shape[0]
        _validate_csr("node-macro incidence", node_macro_row_start, node_macro_columns, n_nodes, n_macros)
        _validate_csr("macro neighbors", macro_neighbor_row_start, macro_neighbor_columns, n_macros, n_macros)
        self.basis = basis
        self.core_angle = core_angle
        self.support_angle = support_angle
        self.halo = int(halo)
        self.info = FeatureLibraryInfo(
            directions=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes,)),
            coarse_directions=V_VEC(3, dtype=gs.qd_float, shape=(n_coarse,)),
            centers=V_VEC(3, dtype=gs.qd_float, shape=(n_features,)),
            macro_quads=V(dtype=gs.qd_int, shape=(n_macros, 4)),
            macro_centers=V_VEC(3, dtype=gs.qd_float, shape=(n_macros,)),
            macro_radii=V(dtype=gs.qd_float, shape=(n_macros,)),
            macro_angles=V(dtype=gs.qd_float, shape=(n_features, n_macros)),
            node_macro_row_start=V(dtype=gs.qd_int, shape=(n_nodes + 1,)),
            node_macro_columns=V(dtype=gs.qd_int, shape=(len(node_macro_columns),)),
            macro_neighbor_row_start=V(dtype=gs.qd_int, shape=(n_macros + 1,)),
            macro_neighbor_columns=V(dtype=gs.qd_int, shape=(len(macro_neighbor_columns),)),
            active_macros=V(dtype=gs.qd_int, shape=(n_features, n_macros)),
            local_mask=V(dtype=gs.qd_int, shape=(n_features, n_nodes)),
            is_valid=V(dtype=gs.qd_bool, shape=()),
        )
        self.info.macro_quads.from_numpy(macro_quads.astype(gs.np_int, copy=False))
        self.info.node_macro_row_start.from_numpy(node_macro_row_start.astype(gs.np_int, copy=False))
        self.info.node_macro_columns.from_numpy(node_macro_columns.astype(gs.np_int, copy=False))
        self.info.macro_neighbor_row_start.from_numpy(macro_neighbor_row_start.astype(gs.np_int, copy=False))
        self.info.macro_neighbor_columns.from_numpy(macro_neighbor_columns.astype(gs.np_int, copy=False))
        self.info.is_valid.fill(True)
        kernel_library_normalize(directions, self.info.directions, self.info)
        kernel_library_normalize(coarse_directions, self.info.coarse_directions, self.info)
        if centers is None:
            kernel_library_icosahedron(self.info)
        else:
            kernel_library_normalize(centers, self.info.centers, self.info)
        kernel_library_geometry(self.info)
        kernel_library_active(core_angle, support_angle, self.info)
        expanded = V(dtype=gs.qd_int, shape=(n_features, n_macros))
        for _ in range(self.halo):
            kernel_library_halo(self.info, expanded)
            kernel_library_halo_copy(expanded, self.info)
        kernel_library_local_nodes(basis.coarse_index, self.info)
        if not qd_to_numpy(self.info.is_valid):
            gs.raise_exception("Feature directions, centers, macro geometry and angular parameters must be finite.")
        active = qd_to_numpy(self.info.active_macros, transpose=True, copy=True)
        local = qd_to_numpy(self.info.local_mask, transpose=True, copy=True)
        # Only integer selection masks cross to the host; B's symbolic columns follow those masks.
        self.active_macros = tuple(
            np.flatnonzero(active[:, i_f]).astype(gs.np_int, copy=False) for i_f in range(n_features)
        )
        self.local_nodes = tuple(
            np.flatnonzero(local[:, i_f]).astype(gs.np_int, copy=False) for i_f in range(n_features)
        )
        self.maps = tuple(basis.feature(nodes) for nodes in self.local_nodes)


@qd.kernel
def kernel_library_normalize(source: qd.Tensor, target: qd.Tensor, library: FeatureLibraryInfo):
    for i in range(target.shape[0]):
        vector = source[i]
        length = vector.norm()
        if length > 0.0 and length < 1e30:
            target[i] = vector / length
        else:
            target[i] = qd.Vector.zero(gs.qd_float, 3)
            library.is_valid[None] = False


@qd.kernel
def kernel_library_icosahedron(library: FeatureLibraryInfo):
    for i in range(12):
        golden = 0.5 * (1.0 + qd.sqrt(gs.qd_float(5.0)))
        vector = qd.Vector.zero(gs.qd_float, 3)
        # Lexicographic order of (0, +/-1, +/-golden) and its cyclic permutations.
        if i < 2:
            vector[0], vector[2] = -golden, 2.0 * i - 1.0
        elif i < 4:
            vector[0], vector[1] = -1.0, (2.0 * (i - 2) - 1.0) * golden
        elif i < 8:
            vector[1] = qd.select(i < 6, -1.0, 1.0)
            vector[2] = (2.0 * (i % 2) - 1.0) * golden
        elif i < 10:
            vector[0], vector[1] = 1.0, (2.0 * (i - 8) - 1.0) * golden
        else:
            vector[0], vector[2] = golden, 2.0 * (i - 10) - 1.0
        library.centers[i] = vector / vector.norm()


@qd.kernel
def kernel_library_geometry(library: FeatureLibraryInfo):
    for i_m in range(library.macro_centers.shape[0]):
        center = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(4)):
            center += library.coarse_directions[library.macro_quads[i_m, a]]
        length = center.norm()
        if length > 0.0 and length < 1e30:
            center /= length
        else:
            library.is_valid[None] = False
        radius = gs.qd_float(0.0)
        for a in qd.static(range(4)):
            corner = library.coarse_directions[library.macro_quads[i_m, a]]
            radius = qd.max(radius, qd.acos(qd.min(1.0, qd.max(-1.0, corner.dot(center)))))
        library.macro_centers[i_m] = center
        library.macro_radii[i_m] = radius


@qd.kernel
def kernel_library_active(core_angle: float, support_angle: float, library: FeatureLibraryInfo):
    for i_f, i_m in qd.ndrange(library.active_macros.shape[0], library.active_macros.shape[1]):
        cosine = library.macro_centers[i_m].dot(library.centers[i_f])
        angle = qd.acos(qd.min(1.0, qd.max(-1.0, cosine)))
        library.macro_angles[i_f, i_m] = angle
        library.active_macros[i_f, i_m] = qd.cast(angle <= core_angle + library.macro_radii[i_m] + 1e-14, gs.qd_int)
    for _ in range(1):
        if not (core_angle > 0.0 and core_angle < 1e30 and support_angle > 0.0 and support_angle < 1e30):
            library.is_valid[None] = False


@qd.kernel
def kernel_library_halo(library: FeatureLibraryInfo, expanded: qd.Tensor):
    for i_f, i_m in qd.ndrange(library.active_macros.shape[0], library.active_macros.shape[1]):
        expanded[i_f, i_m] = library.active_macros[i_f, i_m]
    for i_f, i_m in qd.ndrange(library.active_macros.shape[0], library.active_macros.shape[1]):
        if library.active_macros[i_f, i_m]:
            for i_e in range(library.macro_neighbor_row_start[i_m], library.macro_neighbor_row_start[i_m + 1]):
                other = library.macro_neighbor_columns[i_e]
                qd.atomic_max(expanded[i_f, other], 1)


@qd.kernel
def kernel_library_halo_copy(expanded: qd.Tensor, library: FeatureLibraryInfo):
    for i_f, i_m in qd.ndrange(library.active_macros.shape[0], library.active_macros.shape[1]):
        library.active_macros[i_f, i_m] = expanded[i_f, i_m]


@qd.kernel
def kernel_library_local_nodes(coarse_index: qd.types.ndarray(), library: FeatureLibraryInfo):
    for i_f, i_n in qd.ndrange(library.local_mask.shape[0], library.local_mask.shape[1]):
        inside = coarse_index[i_n] < 0
        for i_e in range(library.node_macro_row_start[i_n], library.node_macro_row_start[i_n + 1]):
            inside = inside and library.active_macros[i_f, library.node_macro_columns[i_e]] != 0
        library.local_mask[i_f, i_n] = qd.cast(inside, gs.qd_int)
