"""Shared native quadrilateral shell assembly and prefactorized recovery."""

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_MAT, V_VEC, DataItem, V, iter_data
from genesis.utils.misc import qd_to_numpy

from . import shell_operators
from .balance import kernel_centrifugal_wrench, kernel_relief_projection
from .data import StressState, allocate_stress_info
from .factor import StressFactor
from .history import StressHistory
from .inverse import StressInverse
from .model import StressBuildTimings, StressModel
from .operators import kernel_diagonal, kernel_gram_inverse
from .shell_operators import ShellInfo
from .shell_pipeline import kernel_recover
from .shell_surface import build_shell_surface
from .surface_inverse import StressSurfaceInverse


class ShellModel(StressModel):
    """Recover a fixed linear shell using translation and rotation block rows."""

    def __init__(self, options: RigidStressOptions):
        started = time.perf_counter()
        self.options = options.model_copy(deep=True)
        with np.load(Path(options.mesh), allow_pickle=False) as asset:
            vertices, quads = asset["vertices"], asset["quads"]
            directors = asset["directors"] if "directors" in asset else None
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            gs.raise_exception("Shell midsurface vertices must be a finite (n, 3) array.")
        if quads.ndim != 2 or quads.shape[1] != 4 or not len(quads):
            gs.raise_exception("Shell connectivity must be a nonempty (n, 4) array.")
        if not np.issubdtype(quads.dtype, np.integer) or quads.min() < 0 or quads.max() >= len(vertices):
            gs.raise_exception("Shell connectivity must contain in-range integer vertex indices.")
        if len(np.unique(quads)) != len(vertices):
            gs.raise_exception("Every shell vertex must belong to a quadrilateral.")
        if directors is not None and (directors.shape != vertices.shape or not np.isfinite(directors).all()):
            gs.raise_exception("Shell directors must match the finite midsurface coordinates.")
        n_nodes, n_elements = len(vertices), len(quads)
        self.recovery_capacity = next((size for size in (128, 256, 512, 1024) if 2 * n_nodes <= size), 0)
        # Each node contributes adjacent translation and rotation triplets to the shared block solver.
        element_blocks = (2 * quads[:, :, None] + np.arange(2)).reshape((-1, 8))
        pairs = np.stack(np.broadcast_arrays(element_blocks[:, :, None], element_blocks[:, None, :]), axis=-1)
        entries, inverse = np.unique(pairs.reshape((-1, 2)), axis=0, return_inverse=True)
        row_start = np.r_[0, np.cumsum(np.bincount(entries[:, 0], minlength=2 * n_nodes))]
        assembly_start = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(entries)))]
        assembly_idx = np.argsort(inverse, kind="stable")
        self.n_boundary_nodes = 2 * n_nodes
        self.info = allocate_stress_info(2 * n_nodes, 0, 0, len(entries), len(inverse))
        self.info.vertices.from_numpy(np.repeat(vertices, 2, axis=0).astype(gs.np_float, copy=False))
        self.info.row_start.from_numpy(row_start.astype(gs.np_int, copy=False))
        self.info.columns.from_numpy(entries[:, 1].astype(gs.np_int, copy=False))
        self.info.assembly_start.from_numpy(assembly_start.astype(gs.np_int, copy=False))
        self.info.assembly_idx.from_numpy(assembly_idx.astype(gs.np_int, copy=False))
        self.info.is_free.fill(1)
        self.shell = ShellInfo(
            quads=V(dtype=gs.qd_int, shape=(n_elements, 4)),
            directors=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes,)),
            membrane=V_VEC(3, dtype=gs.qd_float, shape=(n_elements, 4, 24)),
            bending=V_VEC(3, dtype=gs.qd_float, shape=(n_elements, 4, 24)),
            shear=V_VEC(2, dtype=gs.qd_float, shape=(n_elements, 4, 24)),
            drilling=V(dtype=gs.qd_float, shape=(n_elements, 4, 24)),
            axes=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_elements, 4)),
            inverse=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_elements, 4)),
            derivative=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_elements, 4)),
            tangents=V_MAT(3, 2, dtype=gs.qd_float, shape=(n_elements, 4)),
            interpolated_director=V_VEC(3, dtype=gs.qd_float, shape=(n_elements, 4)),
            weights=V(dtype=gs.qd_float, shape=(n_elements, 4)),
            nodal_mass=V(dtype=gs.qd_float, shape=(n_nodes,)),
            is_valid=V(dtype=gs.qd_bool, shape=()),
        )
        self.shell.quads.from_numpy(quads.astype(gs.np_int, copy=False))
        if directors is not None:
            self.shell.directors.from_numpy(directors.astype(gs.np_float, copy=False))
        self.shell.is_valid.fill(True)
        topology_done = time.perf_counter()
        shell_operators.kernel_directors(self.info, self.shell, supplied=directors is not None)
        shell_operators.kernel_geometry(self.info, self.shell)
        if not qd_to_numpy(self.shell.is_valid):
            gs.raise_exception("Shell directors and quadrilateral Jacobians must have consistent positive orientation.")
        shell_operators.kernel_operators(self.info, self.shell)
        shell_operators.kernel_assemble(
            options.shell_thickness, options.poisson, options.shell_drilling_factor, self.info, self.shell
        )
        shell_operators.kernel_mass(options.shell_thickness, options.density, self.info, self.shell)
        shell_operators.kernel_modes(options.shell_thickness, self.info, self.shell)
        kernel_gram_inverse(self.info)
        kernel_relief_projection(self.info)
        kernel_centrifugal_wrench(self.info)
        shell_operators.kernel_gauge(self.info, self.shell, translation_only=options.shell_gauge == "translation")
        if not qd_to_numpy(self.shell.is_valid):
            gs.raise_exception("Shell gauge must span all six rigid modes.")
        kernel_diagonal(self.info)
        qd.sync()
        operators_done = time.perf_counter()
        inverse_bytes = (6 * n_nodes) ** 2 * (8 if options.inverse_precision == "64" else 4)
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
        self.load_info = build_shell_surface(quads, options.shell_thickness, self.info, self.shell)
        boundary_bytes = ((2 * n_nodes) ** 2 * 9 + 2 * n_nodes * 18) * np.dtype(gs.np_float).itemsize
        boundary_nodes = np.arange(2 * n_nodes, dtype=gs.np_int)
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

    def iter_info_data(self, prefix: str) -> Iterator[DataItem]:
        yield from super().iter_info_data(prefix)
        yield from iter_data(self.shell, f"{prefix}.shell")
        yield from iter_data(self.load_info, f"{prefix}.load_info")

    @property
    def n_output_elements(self) -> int:
        return 2 * self.shell.quads.shape[0]

    @property
    def n_shell_elements(self) -> int:
        return self.shell.quads.shape[0]

    def evaluate_stress(self, options: RigidStressOptions, state: StressState) -> None:
        shell_operators.kernel_peak(options.young, options.poisson, options.shell_thickness, state, self.shell)

    def is_fused_recovery_enabled(self, options: RigidStressOptions, history: StressHistory | None = None) -> bool:
        return (
            gs.backend == gs.cuda
            and options.fused_pipeline
            and self.surface_inverse is not None
            and options.packed_surface_loads
            and options.cooperative_solve
            and self.recovery_capacity > 0
            and history is None
        )

    def recover(
        self,
        omega: qd.Tensor,
        state: StressState,
        options: RigidStressOptions | None = None,
        history: StressHistory | None = None,
        surface_load: bool = False,
    ) -> None:
        if options is None:
            options = self.options
        if surface_load and self.is_fused_recovery_enabled(options, history):
            coalesced = options.coalesced_residual
            if coalesced is None:
                coalesced = state.active.shape[0] >= 8192
            block_size = options.packed_block_size
            if block_size is None:
                block_size = 512 if state.active.shape[0] >= 8192 else 0
            kernel_recover(
                options.young,
                options.poisson,
                options.shell_thickness,
                options.tolerance,
                options.absolute_tolerance,
                omega,
                state,
                self.info,
                self.shell,
                self.surface_inverse.info,
                self.inverse.info,
                self.factor.info,
                options.inverse_corrections,
                self.recovery_capacity,
                block_size,
                coalesced,
                options.cooperative_balance,
            )
        else:
            super().recover(omega, state, options, history, surface_load)
