import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_MAT, V_VEC, V
from genesis.utils.misc import qd_to_numpy

from .data import StressInfo, StressState
from .factor import StressFactor
from .history import StressHistory
from .inverse import StressInverse
from .operators import (
    kernel_assemble,
    kernel_corner_gradients,
    kernel_diagonal,
    kernel_gauge,
    kernel_geometry,
    kernel_gram_inverse,
    kernel_mass_modes,
    kernel_mass_properties,
    kernel_midpoints,
    kernel_modes,
    kernel_valid_geometry,
)
from .solve import (
    kernel_active,
    kernel_balance,
    kernel_direct_init,
    kernel_full_residual,
    kernel_pcg_chunk,
    kernel_pcg_init,
    kernel_peak,
)
from .surface_inverse import StressSurfaceInverse


@dataclass(frozen=True)
class StressBuildTimings:
    topology_seconds: float
    operators_seconds: float
    factor_seconds: float
    inverse_seconds: float
    surface_inverse_seconds: float


class StressModel:
    """Own shared immutable operators assembled from a tetrahedral mesh asset."""

    def __init__(self, options: RigidStressOptions):
        started = time.perf_counter()
        self.options = options.model_copy(deep=True)
        with np.load(Path(options.mesh), allow_pickle=False) as asset:
            vertices = asset["vertices"]
            tetrahedra = asset["tetrahedra"]
            surface = asset["surface_triangles"]
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            gs.raise_exception("Stress mesh vertices must be a finite (n, 3) array.")
        if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or not len(tetrahedra):
            gs.raise_exception("Stress mesh tetrahedra must be a nonempty (n, 4) array.")
        if surface.ndim != 2 or surface.shape[1] != 3 or not len(surface):
            gs.raise_exception("Stress mesh surface_triangles must be a nonempty (n, 3) array.")
        for indices in (tetrahedra, surface):
            if not np.issubdtype(indices.dtype, np.integer) or indices.min() < 0 or indices.max() >= len(vertices):
                gs.raise_exception("Stress mesh connectivity must contain in-range integer vertex indices.")
        if len(np.unique(tetrahedra)) != len(vertices):
            gs.raise_exception("Every stress mesh vertex must belong to a tetrahedron.")

        # Connectivity processing handles asset topology; physical operators are assembled in Quadrants below.
        edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=gs.np_int)
        edge_vertices, edge_inverse = np.unique(
            np.sort(tetrahedra[:, edges].reshape((-1, 2)), axis=1), axis=0, return_inverse=True
        )
        elements = np.column_stack((tetrahedra, len(vertices) + edge_inverse.reshape((-1, 6))))
        xyz = np.zeros((len(vertices) + len(edge_vertices), 3), dtype=gs.np_float)
        xyz[: len(vertices)] = vertices
        face_edges = np.sort(surface[:, [[0, 1], [0, 2], [1, 2]]], axis=2)
        edge_keys = edge_vertices[:, 0] * len(vertices) + edge_vertices[:, 1]
        face_keys = face_edges[:, :, 0] * len(vertices) + face_edges[:, :, 1]
        mids = np.searchsorted(edge_keys, face_keys)
        if (mids >= len(edge_keys)).any() or not np.array_equal(edge_keys[mids], face_keys):
            gs.raise_exception("Stress exterior faces must use tetrahedral mesh edges.")
        surface_nodes = np.column_stack((surface, len(vertices) + mids))
        boundary_nodes = np.unique(surface_nodes)
        self.n_boundary_nodes = len(boundary_nodes)
        pairs = np.stack(np.broadcast_arrays(elements[:, :, None], elements[:, None, :]), axis=-1)
        entries, inverse = np.unique(pairs.reshape((-1, 2)), axis=0, return_inverse=True)
        row_start = np.r_[0, np.cumsum(np.bincount(entries[:, 0], minlength=len(xyz)))]
        n_nodes, n_elements, n_entries = len(xyz), len(elements), len(entries)

        self.info = StressInfo(
            vertices=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes,)),
            elements=V(dtype=gs.qd_int, shape=(n_elements, 10)),
            surface_nodes=V(dtype=gs.qd_int, shape=(len(surface), 6)),
            edges=V(dtype=gs.qd_int, shape=(6, 2)),
            gradients=V_MAT(4, 3, dtype=gs.qd_float, shape=(n_elements,)),
            corner_gradients=V_VEC(3, dtype=gs.qd_float, shape=(n_elements, 4, 7)),
            corner_nodes=V(dtype=gs.qd_int, shape=(n_elements, 4, 7)),
            volumes=V(dtype=gs.qd_float, shape=(n_elements,)),
            row_start=V(dtype=gs.qd_int, shape=(n_nodes + 1,)),
            columns=V(dtype=gs.qd_int, shape=(n_entries,)),
            element_entries=V(dtype=gs.qd_int, shape=(n_elements, 10, 10)),
            stiffness=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_entries,)),
            mass=V(dtype=gs.qd_float, shape=(n_entries,)),
            modes=V_MAT(3, 6, dtype=gs.qd_float, shape=(n_nodes,)),
            mass_modes=V_MAT(3, 6, dtype=gs.qd_float, shape=(n_nodes,)),
            centrifugal=V_MAT(3, 6, dtype=gs.qd_float, shape=(n_nodes,)),
            gram=V_MAT(6, 6, dtype=gs.qd_float, shape=()),
            gram_inverse=V_MAT(6, 6, dtype=gs.qd_float, shape=()),
            pins=V(dtype=gs.qd_int, shape=(6,)),
            is_free=V_VEC(3, dtype=gs.qd_int, shape=(n_nodes,)),
            diagonal_inverse=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_nodes,)),
            scalar_diagonal_inverse=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes,)),
            mass_properties=V(dtype=gs.qd_float, shape=(4,)),
        )
        self.info.vertices.from_numpy(xyz.astype(gs.np_float, copy=False))
        self.info.elements.from_numpy(elements.astype(gs.np_int, copy=False))
        self.info.surface_nodes.from_numpy(surface_nodes.astype(gs.np_int, copy=False))
        self.info.edges.from_numpy(edges)
        self.info.row_start.from_numpy(row_start.astype(gs.np_int, copy=False))
        self.info.columns.from_numpy(entries[:, 1].astype(gs.np_int, copy=False))
        self.info.element_entries.from_numpy(inverse.reshape((n_elements, 10, 10)).astype(gs.np_int, copy=False))
        self.info.stiffness.fill(0)
        self.info.mass.fill(0)
        self.info.mass_properties.fill(0)
        self.info.gram.fill(0)
        self.info.is_free.fill(1)
        topology_done = time.perf_counter()
        kernel_midpoints(len(vertices), edge_vertices.astype(gs.np_int, copy=False), self.info)
        kernel_geometry(self.info)
        kernel_corner_gradients(self.info)
        if kernel_valid_geometry(self.info):
            gs.raise_exception("Stress tetrahedra must have positive volume.")
        kernel_assemble(options.poisson, options.density, self.info)
        kernel_mass_properties(self.info)
        kernel_modes(self.info)
        kernel_mass_modes(self.info)
        kernel_gram_inverse(self.info)
        kernel_gauge(self.info)
        kernel_diagonal(self.info)
        qd.sync()
        operators_done = time.perf_counter()
        inverse_bytes = n_nodes * n_nodes * 9 * (8 if options.inverse_precision == "64" else 4)
        self.selected_method = options.method
        if options.method == "auto":
            self.selected_method = "inverse" if inverse_bytes <= options.inverse_max_bytes else "direct"
        self.factor = StressFactor(self.info) if self.selected_method != "pcg" else None
        qd.sync()
        factor_done = time.perf_counter()
        self.inverse = (
            StressInverse(
                self.info, self.factor, self.create_state, options.inverse_max_bytes, options.inverse_precision
            )
            if self.selected_method == "inverse"
            else None
        )
        inverse_done = time.perf_counter()
        boundary_bytes = (n_nodes * len(boundary_nodes) * 9 + n_nodes * 18) * np.dtype(gs.np_float).itemsize
        boundary_bytes += boundary_nodes.nbytes
        self.surface_inverse = None
        if (
            self.inverse is not None
            and options.surface_inverse
            and options.inverse_precision == "64"
            and inverse_bytes + boundary_bytes <= options.inverse_max_bytes
        ):
            self.surface_inverse = StressSurfaceInverse(self.info, self.factor, self.create_state, boundary_nodes)
        self.build_timings = StressBuildTimings(
            topology_done - started,
            operators_done - topology_done,
            factor_done - operators_done,
            inverse_done - factor_done,
            time.perf_counter() - inverse_done,
        )

    def create_state(self, n_envs: int, output_mode: str = "max") -> StressState:
        n_nodes = self.info.vertices.shape[0]
        n_krylov_nodes = n_nodes if self.selected_method == "pcg" else 0
        n_output_elements = self.info.elements.shape[0] if output_mode == "full" else 0
        state = StressState(
            force=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes, n_envs)),
            rhs=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes, n_envs)),
            displacement=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes, n_envs)),
            residual=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes, n_envs)),
            direction=V_VEC(3, dtype=gs.qd_float, shape=(n_krylov_nodes, n_envs)),
            product=V_VEC(3, dtype=gs.qd_float, shape=(n_krylov_nodes, n_envs)),
            preconditioned=V_VEC(3, dtype=gs.qd_float, shape=(n_krylov_nodes, n_envs)),
            boundary_columns=V(dtype=gs.qd_int, shape=(self.n_boundary_nodes, n_envs)),
            boundary_count=V(dtype=gs.qd_int, shape=(n_envs,)),
            wrench=V_VEC(6, dtype=gs.qd_float, shape=(n_envs,)),
            rhs_norm_squared=V(dtype=gs.qd_float, shape=(n_envs,)),
            residual_norm_squared=V(dtype=gs.qd_float, shape=(n_envs,)),
            residual_preconditioned=V(dtype=gs.qd_float, shape=(n_envs,)),
            next_residual_preconditioned=V(dtype=gs.qd_float, shape=(n_envs,)),
            direction_product=V(dtype=gs.qd_float, shape=(n_envs,)),
            active=V(dtype=gs.qd_int, shape=(n_envs,)),
            iterations=V(dtype=gs.qd_int, shape=(n_envs,)),
            peak=V(dtype=gs.qd_float, shape=(n_envs,)),
            step_peak=V(dtype=gs.qd_float, shape=(n_envs,)),
            step_valid=V(dtype=gs.qd_bool, shape=(n_envs,)),
            stress_tensor=V_VEC(6, dtype=gs.qd_float, shape=(n_output_elements, 4, n_envs)),
            von_mises=V_VEC(1, dtype=gs.qd_float, shape=(n_output_elements, 4, n_envs)),
            valid=V(dtype=gs.qd_bool, shape=(n_envs,)),
            corrections=V(dtype=gs.qd_int, shape=(n_envs,)),
            fallbacks=V(dtype=gs.qd_int, shape=(n_envs,)),
            invalid_steps=V(dtype=qd.i64, shape=(n_envs,)),
            invalid_reported=V(dtype=gs.qd_bool, shape=(n_envs,)),
        )
        state.displacement.fill(0)
        state.step_peak.fill(0)
        state.valid.fill(True)
        state.step_valid.fill(True)
        state.invalid_steps.fill(0)
        state.invalid_reported.fill(False)
        if n_output_elements:
            state.stress_tensor.fill(float("nan"))
            state.von_mises.fill(float("nan"))
        return state

    def recover(
        self,
        omega,
        state: StressState,
        options: RigidStressOptions | None = None,
        history: StressHistory | None = None,
        surface_load: bool = False,
    ) -> None:
        """Balance the complete load, recover displacement and scan the full-domain peak."""
        if options is None:
            options = self.options
        block = options.preconditioner == "block"
        kernel_balance(omega, state, self.info)
        if self.factor is not None:
            kernel_direct_init(state)
            if history is None:
                self.solve(options, state, omega, surface_load)
            else:
                history.predict(state)
                kernel_full_residual(
                    options.young, options.tolerance, options.absolute_tolerance, state, self.info, False
                )
                history.mark_hits(state)
                if self.inverse is not None:
                    self.inverse.apply(options.young, state, only_failed=True)
                else:
                    self.factor.solve(options.young, state, self.info, options.cooperative_solve, only_failed=True)
        else:
            kernel_pcg_init(options.young, state, self.info, block, options.warm_start)
            for _ in range((options.max_iterations + 15) // 16):
                kernel_pcg_chunk(
                    options.young,
                    0.01 * options.tolerance,
                    options.absolute_tolerance,
                    options.max_iterations,
                    state,
                    self.info,
                    block,
                )
                if not kernel_active(state):
                    break
        kernel_full_residual(
            options.young, options.tolerance, options.absolute_tolerance, state, self.info, history is not None
        )
        if self.inverse is not None:
            for _ in range(options.inverse_corrections):
                self.inverse.apply(options.young, state, correction=True)
                kernel_full_residual(
                    options.young, options.tolerance, options.absolute_tolerance, state, self.info, only_active=True
                )
            self.factor.solve(options.young, state, self.info, options.cooperative_solve, only_failed=True)
            kernel_full_residual(
                options.young, options.tolerance, options.absolute_tolerance, state, self.info, only_active=True
            )
        kernel_peak(options.young, options.poisson, state, self.info, options.cached_peak)

    def solve(
        self,
        options: RigidStressOptions,
        state: StressState,
        omega: qd.Tensor | None = None,
        surface_load: bool = False,
    ) -> None:
        if surface_load and self.surface_inverse is not None:
            assert omega is not None
            self.surface_inverse.apply(options.young, omega, state, options.packed_surface_loads)
        elif self.inverse is not None:
            self.inverse.apply(options.young, state)
        else:
            self.factor.solve(options.young, state, self.info, options.cooperative_solve)
