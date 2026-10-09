"""Runnable CPU integration layer over the previously validated P2 algorithms.

Only the reference mesh, material and mass distribution are cached. Contact
locations, directions, counts and radii are inputs at every call. This module
does not impose physical supports or contact symmetry.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REFERENCE = Path(__file__).resolve().parent / "reference"
sys.path[:0] = [
    str(REFERENCE / "output"),
    str(REFERENCE / "output/arbitrary_contact_cpu"),
    str(REFERENCE / "output/smooth_friction_cpu"),
    str(REFERENCE / "tmp/batch_history_validation"),
]
import arbitrary_contact_test as reference
from benchmark import BatchRecovery, CachedHistory
from general_peak import P2Peak
from indexed_contacts import IndexedContacts


@dataclass(frozen=True)
class EggConfig:
    level: int = 2
    layers: int = 2
    thickness_m: float = 0.0005
    young_pa: float = 1.0e10
    poisson: float = 0.3
    density_kg_m3: float = 2000.0
    surface_quadrature: int = 10


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


@dataclass(frozen=True)
class MappedLoads:
    nodal_force_n: np.ndarray
    resultant_force_n: np.ndarray
    resultant_moment_nm: np.ndarray
    input_moment_nm: np.ndarray


@dataclass(frozen=True)
class RecoveryResult:
    peak_pa: np.ndarray
    displacement_m: np.ndarray
    relative_residual: np.ndarray
    corrected_environments: int


class EggRecoveryCPU:
    def __init__(self, environments=1, config=EggConfig(), history=8, rtol=1e-6, direct=False):
        if environments < 1 or history < 0 or rtol <= 0:
            raise ValueError("Positive environment count/tolerance and nonnegative history required")
        self.config = config
        self.environments = environments
        xyz, tetrahedra, outer, self.mesh_metadata = reference.egg.egg_mesh(
            config.level, config.layers, config.thickness_m
        )
        self.fem = reference.conv.FEM(
            xyz, tetrahedra, outer, order=2, quarter=False,
            young=config.young_pa, poisson=config.poisson,
            density=config.density_kg_m3, label="full-egg-recovery",
        )
        self.surface = reference.SurfaceGeometry(self.fem, config.surface_quadrature)
        self.mapper = IndexedContacts(self.surface)
        self.peak = P2Peak(self.fem.glambda, self.fem.elements,
                           config.young_pa / (2 * (1 + config.poisson)))
        self.peak.warmup(np.zeros(self.fem.ndof))
        self.solver = BatchRecovery(self.fem, self.peak, environments, history, rtol,
                                    history="cached", strategy="compact", direct=direct)

    def map_contacts(self, contacts: ContactBatch) -> MappedLoads:
        b, c = contacts.valid.shape
        if b != self.environments or contacts.position_m.shape != (b, c, 3) or contacts.force_n.shape != (b, c, 3):
            raise ValueError("Contact batch must match environment count and [B,C,3] vector shapes")
        if contacts.radius_m.shape != (b, c):
            raise ValueError("radius_m must have shape [B,C]")
        if not np.isfinite(contacts.position_m[contacts.valid]).all() or not np.isfinite(contacts.force_n[contacts.valid]).all():
            raise ValueError("Active contact positions/forces must be finite")
        raw = np.zeros((self.fem.ndof, b), order="F")
        input_moment = np.zeros((b, 3))
        # This CPU adapter reuses the validated compact-support mapper. It
        # conserves resultant force, but reports (rather than hides) the
        # change of moment caused by replacing a point with a finite patch.
        for env in range(b):
            patches = []
            for contact in np.flatnonzero(contacts.valid[env]):
                radius = contacts.radius_m[env, contact]
                patches.append({"center_m": contacts.position_m[env, contact],
                                "force_N": contacts.force_n[env, contact],
                                "radius_m": radius, "sigma_m": 0.45 * radius})
            raw[:, env] = self.mapper.load(patches)
            input_moment[env] = np.cross(contacts.position_m[env, contacts.valid[env]],
                                         contacts.force_n[env, contacts.valid[env]]).sum(axis=0)
        nodal = raw.reshape(-1, 3, b).transpose(2, 0, 1)
        return MappedLoads(raw, nodal.sum(axis=1),
                           np.cross(self.fem.xyz[None], nodal).sum(axis=1), input_moment)

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

    def recover(self, compatible_rhs_n) -> RecoveryResult:
        u, peaks, statistics = self.solver.step(compatible_rhs_n)
        residual = np.linalg.norm(self.fem.k @ u - compatible_rhs_n, axis=0)
        norm = np.linalg.norm(compatible_rhs_n, axis=0)
        if not np.isfinite(u).all() or not np.isfinite(peaks).all():
            raise FloatingPointError("Nonfinite displacement or stress")
        allowed = np.maximum(1e-11, np.maximum(5 * self.solver.rtol, 1e-8) * norm)
        if np.any(residual > allowed):
            raise ArithmeticError("Complete equilibrium residual failed, including gauge rows")
        relative = np.divide(residual, norm, out=np.zeros_like(residual), where=norm > 1e-12)
        return RecoveryResult(peaks, u, relative, statistics["failed"])

    def reset(self, environments=None):
        ids = np.arange(self.environments) if environments is None else np.asarray(environments)
        for env in ids:
            self.solver.history[env] = CachedHistory(self.solver.K, self.solver.capacity)
        if self.solver.last is not None:
            self.solver.last[:, ids] = 0
        if self.solver.before is not None:
            self.solver.before[:, ids] = 0
