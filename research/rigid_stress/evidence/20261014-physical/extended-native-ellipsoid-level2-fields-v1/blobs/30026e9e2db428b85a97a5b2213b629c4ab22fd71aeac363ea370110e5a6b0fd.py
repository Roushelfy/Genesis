"""Runnable CPU integration layer over the previously validated P2 algorithms.

Only the reference mesh, material and mass distribution are cached. Contact
locations, directions, counts and radii are inputs at every call. This module
does not impose physical supports or contact symmetry.
"""

from dataclasses import dataclass

import numpy as np

from .history_cpu import BatchRecovery
from .mechanics import P2Shell, SurfaceGeometry, shell_mesh
from .peak_cpu import P2Peak
from .wrench import FinitePatchMapper, WrenchDiagnostics, WrenchPatch


@dataclass(frozen=True)
class EggConfig:
    level: int = 2
    layers: int = 2
    thickness_m: float = 0.0005
    young_pa: float = 1.0e10
    poisson: float = 0.3
    density_kg_m3: float = 2000.0
    surface_quadrature: int = 10
    ordering: str = "mmd"
    factor_backend: str = "superlu"


@dataclass(frozen=True)
class ContactBatch:
    """All arrays are CPU NumPy arrays in the authored egg reference frame.

    position_m/force_n: [B,C,3]; radius_m: [B,C]; valid: [B,C]. Forces
    already include normal and tangential components. A finite footprint is
    an explicit load model; rigid contacts alone do not identify its radius.
    """

    position_m: np.ndarray
    force_n: np.ndarray
    radius_m: np.ndarray
    valid: np.ndarray
    friction: np.ndarray
    inward_normal: np.ndarray | None = None
    source_epsilon: float = 0.0


@dataclass(frozen=True)
class MappedLoads:
    nodal_force_n: np.ndarray
    resultant_force_n: np.ndarray
    resultant_moment_nm: np.ndarray
    input_moment_nm: np.ndarray
    contact_diagnostics: tuple[tuple[WrenchDiagnostics, ...], ...]
    footprint_center_m: np.ndarray


@dataclass(frozen=True)
class RecoveryResult:
    peak_pa: np.ndarray
    displacement_m: np.ndarray
    relative_residual: np.ndarray
    corrected_environments: int


class EggRecoveryCPU:
    def __init__(self, environments=1, config=None, history=8, rtol=1e-6, direct=False, anchor_to_surface=False):
        if environments < 1 or history < 0 or rtol <= 0:
            raise ValueError("Positive environment count/tolerance and nonnegative history required")
        if config is None:
            config = EggConfig()
        self.config = config
        self.environments = environments
        xyz, tetrahedra, outer, self.mesh_metadata = shell_mesh(
            config.level, config.layers, config.thickness_m
        )
        self.fem = P2Shell(
            xyz, tetrahedra, outer, config.young_pa, config.poisson, config.density_kg_m3,
            config.layers, config.ordering, config.factor_backend,
        )
        self.surface = SurfaceGeometry(self.fem, config.surface_quadrature)
        self.mapper = FinitePatchMapper(self.surface, anchor_to_surface=anchor_to_surface)
        self.peak = P2Peak(self.fem.glambda, self.fem.elements, config.young_pa / (2 * (1 + config.poisson)))
        self.peak.warmup(np.zeros(self.fem.ndof))
        self.solver = BatchRecovery(
            self.fem, self.peak, environments, history, rtol, direct=direct
        )

    def map_contacts(self, contacts: ContactBatch) -> MappedLoads:
        b, c = contacts.valid.shape
        if b != self.environments or contacts.position_m.shape != (b, c, 3) or contacts.force_n.shape != (b, c, 3):
            raise ValueError("Contact batch must match environment count and [B,C,3] vector shapes")
        if contacts.radius_m.shape != (b, c):
            raise ValueError("radius_m must have shape [B,C]")
        if contacts.friction.shape != (b, c) or contacts.valid.dtype != np.bool_:
            raise ValueError("friction must have shape [B,C] and valid must be boolean")
        if contacts.inward_normal is not None and contacts.inward_normal.shape != (b, c, 3):
            raise ValueError("Supplied inward pad normals must have shape [B,C,3]")
        if (
            not np.isfinite(contacts.position_m[contacts.valid]).all()
            or not np.isfinite(contacts.force_n[contacts.valid]).all()
        ):
            raise ValueError("Active contact positions/forces must be finite")
        raw = np.zeros((self.fem.ndof, b), order="F")
        input_moment = np.zeros((b, 3))
        footprint_centers = contacts.position_m.astype(np.float64).copy()
        diagnostics = []
        for env in range(b):
            environment_diagnostics = []
            for contact in np.flatnonzero(contacts.valid[env]):
                radius = contacts.radius_m[env, contact]
                patch = WrenchPatch(
                    contacts.position_m[env, contact].astype(np.float64),
                    contacts.force_n[env, contact].astype(np.float64),
                    float(radius),
                    float(contacts.friction[env, contact]),
                    np.zeros(3),
                    None if contacts.inward_normal is None else contacts.inward_normal[env, contact],
                    contacts.source_epsilon,
                )
                mapped = self.mapper.map(patch)
                raw[:, env] += mapped.nodal_force_n
                footprint_centers[env, contact] = mapped.footprint_center_m
                environment_diagnostics.append(mapped.diagnostics)
            diagnostics.append(tuple(environment_diagnostics))
            input_moment[env] = np.cross(
                contacts.position_m[env, contacts.valid[env]].astype(np.float64),
                contacts.force_n[env, contacts.valid[env]].astype(np.float64),
            ).sum(axis=0)
        nodal = raw.reshape(-1, 3, b).transpose(2, 0, 1)
        return MappedLoads(
            raw, nodal.sum(axis=1), np.cross(self.fem.xyz[None], nodal).sum(axis=1), input_moment, tuple(diagnostics),
            footprint_centers,
        )

    def compatible_rhs(self, raw_force_n, omega_rad_s=None):
        """Inferred-acceleration inertia relief, including centrifugal inertia.

        raw_force_n [ndof,B] must include ALL external loads. This is not the
        measured-acceleration variant: see docs/ALGORITHM.md before integration.
        """
        if raw_force_n.shape != (self.fem.ndof, self.environments):
            raise ValueError("Complete external nodal loads must have shape [ndof,B]")
        raw = raw_force_n.copy(order="F")
        if omega_rad_s is not None:
            if omega_rad_s.shape != (self.environments, 3):
                raise ValueError("omega_rad_s must have shape [B,3]")
            relative = self.fem.xyz - self.fem.com
            for env in range(self.environments):
                omega = omega_rad_s[env]
                centrifugal = np.cross(omega, np.cross(omega, relative)).ravel()
                raw[:, env] -= self.fem.m @ centrifugal
        return self.fem.balance(raw)

    def recover(self, compatible_rhs_n, dt=1.0) -> RecoveryResult:
        u, peaks, statistics = self.solver.step(compatible_rhs_n, dt)
        residual = np.linalg.norm(self.fem.k @ u - compatible_rhs_n, axis=0)
        norm = np.linalg.norm(compatible_rhs_n, axis=0)
        if not np.isfinite(u).all() or not np.isfinite(peaks).all():
            raise FloatingPointError("Nonfinite displacement or stress")
        allowed = np.maximum(1e-11, self.solver.rtol * norm)
        if np.any(residual > allowed):
            raise ArithmeticError("Complete equilibrium residual failed, including gauge rows")
        relative = np.divide(residual, norm, out=np.zeros_like(residual), where=norm > 1e-12)
        return RecoveryResult(peaks, u, relative, statistics.failed)

    def reset(self, environments=None):
        ids = np.arange(self.environments) if environments is None else np.asarray(environments)
        self.solver.reset(ids)
