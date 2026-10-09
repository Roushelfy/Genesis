#!/usr/bin/env python3
"""CPU reference for floating-body, small-strain elastic stress recovery.

This is a mathematical reference and an integration starting point, NOT a
Genesis plugin, a GPU implementation, or a performance benchmark.  Dependencies
are NumPy and SciPy.  The rigid simulator supplies pointwise external forces in
the body's coordinates; this module recovers an auxiliary elastic displacement
and per-tetrahedron Cauchy stress.  It does not feed elastic deformation back into
rigid collision geometry.

The supplied xyz and point_load positions use the same original body-local mesh
coordinates; xyz does not have to be COM-centered.  Force vectors use the body's
orientation.  Rigid modes and inertial offsets use xyz - COM internally.

The cached objects are the P1 tetrahedral stiffness K, consistent mass M, six
rigid-body modes R, G = R.T M R, element stress operators D B, and one factorization
of a gauge-fixed K.  For each right-hand side, in body coordinates about the COM:

    c_i = omega x (omega x r_i)
    eta = solve(G, R.T @ (f - M @ c))
    b = f - M @ c - M @ R @ eta
    K u = b
    sigma_e = D @ B_e @ u_e

The six scalar pinned DOFs are only a gauge: R at these rows is nonsingular, so
any displacement admits a rigid shift that makes them zero.  They are NOT six
physical supports.  A compatible load has R.T @ b = 0, and all rows of the full
unconstrained residual must vanish, including pinned rows.  `eta` is inferred
from the supplied loads.  Prescribed rigid acceleration can instead be used in
the inertia field, but nonzero force/torque imbalance then must be diagnosed;
silently projecting it away changes the input physics.

Strain uses engineering Voigt order [xx, yy, zz, xy, yz, xz]; stress uses the
same order with true shear stress.  M uses rho*V/20 * (ones(4,4)+eye(4)) tensor I3,
which exactly integrates linear coordinates and their quadratic products.
Single-RHS stress has shape (n_element, 6); batch stress has shape
(n_rhs, n_element, 6).  Force, compatible_load and displacement retain shape
(n_dof,) for one RHS and (n_dof, n_rhs) for a batch.  Diagnostic residual norms
aggregate an entire batch.  The six-entry wrench norm combines force and torque
in the chosen units, so its normalization is only an internal mathematical
check, not a physically meaningful relative error across units or bodies.

Limits: linear elasticity, small elastic strain, fixed material and volume mesh,
quasi-static elastic response in a moving rigid frame, complete external loads,
and finite-sized contact traction supplied or approximated separately.  Mapping
a rigid point force to triangle nodes preserves resultant and torque, but does
not determine a physical contact area or a reliable near-contact peak stress.
No stress waves, elastic history, plasticity, fracture, or deformation feedback
are simulated.  Stiffness and mass must represent the same material body.  Do
not interpret a rigid collision impulse divided by dt as a resolved impact peak.
Inertia is evaluated on the reference body: terms involving elastic u, u_dot,
and u_ddot, including Coriolis and spin-dependent elastic stiffness, are omitted.

Run the self-contained validation:
    python rigid_stress_reference.py --validation-json rigid_stress_validation.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy import linalg, sparse
from scipy.sparse.linalg import splu


VOIGT_ORDER = ["xx", "yy", "zz", "xy", "yz", "xz"]


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def isotropic_elasticity(young: float, poisson: float) -> np.ndarray:
    if young <= 0.0 or not (-1.0 < poisson < 0.5):
        raise ValueError("Require E > 0 and -1 < nu < 0.5.")
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[np.arange(3), np.arange(3)] += 2.0 * mu
    d[3:, 3:] = mu * np.eye(3)
    return d


def structured_cube(subdivisions: int = 3, side: float = 1.0):
    """Conforming Freudenthal tetrahedra: six permutation paths per cube."""
    if subdivisions < 1 or side <= 0.0:
        raise ValueError("Need subdivisions >= 1 and side > 0.")
    n = subdivisions + 1
    grid = np.linspace(0.0, side, n)
    xyz = np.array(list(itertools.product(grid, repeat=3)), dtype=float)

    def node(index):
        i, j, k = index
        return (i * n + j) * n + k

    tetrahedra = []
    for cell in itertools.product(range(subdivisions), repeat=3):
        for axes in itertools.permutations(range(3)):
            p = np.array(cell, dtype=int)
            path = [node(p)]
            for axis in axes:
                p = p.copy()
                p[axis] += 1
                path.append(node(p))
            tetrahedra.append(path)
    return xyz, np.asarray(tetrahedra, dtype=np.int64)


def boundary_triangles(tetrahedra: np.ndarray) -> np.ndarray:
    counts = {}
    for tet in tetrahedra:
        for omitted in range(4):
            key = tuple(sorted(np.delete(tet, omitted).tolist()))
            counts[key] = counts.get(key, 0) + 1
    if any(count > 2 for count in counts.values()):
        raise ValueError("Nonmanifold tetrahedral mesh.")
    return np.array([face for face, count in counts.items() if count == 1],
                    dtype=np.int64)


def strain_matrix(gradient: np.ndarray) -> np.ndarray:
    b = np.zeros((6, 12))
    for a, (gx, gy, gz) in enumerate(gradient):
        j = 3 * a
        b[0, j] = gx
        b[1, j + 1] = gy
        b[2, j + 2] = gz
        b[3, j:j + 3] = [gy, gx, 0.0]
        b[4, j:j + 3] = [0.0, gz, gy]
        b[5, j:j + 3] = [gz, 0.0, gx]
    return b


def independent_gauge_rows(r: np.ndarray, candidate_order=None) -> np.ndarray:
    """Column-pivoted QR of R.T selects six independent scalar displacement rows."""
    order = (np.arange(r.shape[0]) if candidate_order is None
             else np.asarray(candidate_order, dtype=int))
    if not np.array_equal(np.sort(order), np.arange(r.shape[0])):
        raise ValueError("candidate_order must be a permutation of all DOFs.")
    _, upper, pivots = linalg.qr(r.T[:, order], mode="economic", pivoting=True)
    if abs(upper[5, 5]) <= 1e-12 * abs(upper[0, 0]):
        raise ValueError("The mesh does not have six independent rigid modes.")
    return order[pivots[:6]]


@dataclass
class Recovery:
    """Result shape conventions are intentionally different for fields and RHS.

    displacement / compatible_load: (ndof,) or (ndof, nrhs).
    stress: (nelement, 6) or (nrhs, nelement, 6), Voigt on the final axis.
    rigid_acceleration: (6,) or (6, nrhs).
    residual norms aggregate all RHS.  wrench_relative mixes force/torque units
    and is an internal fixed-unit algebra check, not a physical error measure.
    """
    displacement: np.ndarray
    stress: np.ndarray
    compatible_load: np.ndarray
    rigid_acceleration: np.ndarray
    residual_relative: float
    pinned_residual_relative: float
    wrench_relative: float


class CachedElasticRecovery:
    """One fixed body, one cached sparse LU, arbitrary single or multiple RHS.

    The sparse LU is a portable CPU implementation of the cached-factor idea;
    a production SPD implementation can use Cholesky.  A solve reuses factors,
    and multi-RHS solves reuse them without assembling an environment block
    diagonal matrix.  Factorization counts below are explicit, not timings.
    """

    def __init__(self, xyz, tetrahedra, young=1e6, poisson=0.3, density=2.0,
                 gauge_order=None):
        self.xyz = np.asarray(xyz, dtype=float).copy()
        self.tetrahedra = np.asarray(tetrahedra, dtype=np.int64).copy()
        if density <= 0.0:
            raise ValueError("Density must be positive.")
        self.young, self.poisson, self.density = young, poisson, density
        self.d = isotropic_elasticity(young, poisson)
        self.ndof = 3 * len(self.xyz)
        self.element_dofs = (3 * self.tetrahedra[:, :, None]
                             + np.arange(3)[None, None, :]).reshape(-1, 12)
        k_data, m_data, rows, columns = [], [], [], []
        self.b = np.empty((len(self.tetrahedra), 6, 12))
        self.volumes = np.empty(len(self.tetrahedra))
        for e, tet in enumerate(self.tetrahedra):
            x = self.xyz[tet]
            volume = abs(np.linalg.det((x[1:] - x[0]).T)) / 6.0
            if volume <= 0.0:
                raise ValueError("Degenerate tetrahedron.")
            coefficient = np.linalg.inv(np.column_stack([np.ones(4), x]))
            self.b[e] = strain_matrix(coefficient[1:, :].T)
            self.volumes[e] = volume
            ke = volume * (self.b[e].T @ self.d @ self.b[e])
            me = density * volume / 20.0 * np.kron(
                np.ones((4, 4)) + np.eye(4), np.eye(3))
            dofs = self.element_dofs[e]
            rows.extend(np.repeat(dofs, 12))
            columns.extend(np.tile(dofs, 12))
            k_data.extend(ke.ravel())
            m_data.extend(me.ravel())
        self.k = sparse.coo_matrix((k_data, (rows, columns)),
                                   shape=(self.ndof, self.ndof)).tocsc()
        self.m = sparse.coo_matrix((m_data, (rows, columns)),
                                   shape=(self.ndof, self.ndof)).tocsc()
        self.k.eliminate_zeros()
        self.m.eliminate_zeros()
        # Row sums are nodal integration weights, not a lumped M used by solves.
        nodal_mass = np.asarray(self.m.sum(axis=1)).ravel()[::3]
        self.mass = float(nodal_mass.sum())
        self.com = nodal_mass @ self.xyz / self.mass
        self.offsets = self.xyz - self.com
        self.r = np.empty((self.ndof, 6))
        for a, position in enumerate(self.offsets):
            self.r[3 * a:3 * a + 3] = np.column_stack([np.eye(3), -skew(position)])
        self.mr = self.m @ self.r
        self.g = self.r.T @ self.mr
        self.g_factor = linalg.cho_factor(self.g, lower=True)
        self.inertia = self.g[3:, 3:].copy()
        self.stress_operator = np.einsum("ij,ejk->eik", self.d, self.b)
        self.surface_faces = boundary_triangles(self.tetrahedra)
        self.pins = independent_gauge_rows(self.r, gauge_order)
        self.free = np.setdiff1d(np.arange(self.ndof), self.pins)
        self.factor = splu(self.k[self.free][:, self.free].tocsc())
        self.factorization_count = 1
        self.solve_call_count = 0
        self.rhs_solved = 0

    def point_load(self, face_index, point, force) -> np.ndarray:
        """Triangle barycentric nodal force; preserves force and moment exactly.

        `point` uses the SAME coordinates as the input xyz; neither has to be
        COM-centered.  `force` uses the body's orientation.  Transform world
        coordinates into these body-local mesh coordinates before calling.
        This interpolation says nothing about physical contact area.  Refining a
        point-loaded mesh does not generally converge the peak contact stress.
        """
        triangle = self.surface_faces[face_index]
        x = self.xyz[triangle]
        edge = (x[1:] - x[0]).T
        uv, _, _, _ = np.linalg.lstsq(edge, np.asarray(point) - x[0], rcond=None)
        weights = np.r_[1.0 - uv.sum(), uv]
        scale = max(np.linalg.norm(edge), 1.0)
        if np.linalg.norm(x[0] + edge @ uv - point) > 1e-10 * scale:
            raise ValueError("Point is not in the surface triangle's plane.")
        if weights.min() < -1e-10:
            raise ValueError("Point is outside the surface triangle.")
        load = np.zeros((len(self.xyz), 3))
        load[triangle] = weights[:, None] * np.asarray(force)[None, :]
        return load.ravel()

    def consistent_body_force(self, acceleration) -> np.ndarray:
        return self.m @ np.tile(np.asarray(acceleration), len(self.xyz))

    def remove_rigid_displacement(self, displacement) -> np.ndarray:
        """Mass-orthogonal canonical displacement for comparisons/visualization."""
        u = np.asarray(displacement)
        coefficients = linalg.cho_solve(self.g_factor, self.r.T @ (self.m @ u))
        return u - self.r @ coefficients

    def inertia_relief(self, external_force, omega=None):
        f = np.asarray(external_force, dtype=float)
        single = f.ndim == 1
        if single:
            f = f[:, None]
        if f.ndim != 2 or f.shape[0] != self.ndof:
            raise ValueError("Force must have shape (ndof,) or (ndof, nrhs).")
        nrhs = f.shape[1]
        w = np.zeros((nrhs, 3)) if omega is None else np.asarray(omega, dtype=float)
        if w.shape == (3,):
            w = np.broadcast_to(w[None, :], (nrhs, 3))
        if w.shape != (nrhs, 3):
            raise ValueError("omega must have shape (3,) or (nrhs, 3).")
        # The same formula includes the quadratic angular-velocity term.
        c = np.cross(w[:, None, :], np.cross(w[:, None, :],
                                            self.offsets[None, :, :]))
        h = f - self.m @ c.reshape(nrhs, self.ndof).T
        eta = linalg.cho_solve(self.g_factor, self.r.T @ h)
        compatible = h - self.mr @ eta
        if single:
            return compatible[:, 0], eta[:, 0]
        return compatible, eta

    def solve(self, external_force, omega=None) -> Recovery:
        b, eta = self.inertia_relief(external_force, omega)
        single = b.ndim == 1
        bm = b[:, None] if single else b
        u = np.zeros_like(bm)
        u[self.free] = self.factor.solve(np.asfortranarray(bm[self.free]))
        self.solve_call_count += 1
        self.rhs_solved += bm.shape[1]
        element_u = u[self.element_dofs]
        stress = np.einsum("eij,ejk->eik", self.stress_operator, element_u)
        residual = self.k @ u - bm
        scale = max(np.linalg.norm(bm), 1.0)
        wrench_scale = max(np.linalg.norm(self.r) * np.linalg.norm(bm), 1.0)
        result = Recovery(
            displacement=u[:, 0] if single else u,
            stress=stress[:, :, 0] if single else np.transpose(stress, (2, 0, 1)),
            compatible_load=b,
            rigid_acceleration=eta,
            residual_relative=float(np.linalg.norm(residual) / scale),
            pinned_residual_relative=float(np.linalg.norm(residual[self.pins]) / scale),
            wrench_relative=float(np.linalg.norm(self.r.T @ bm) / wrench_scale),
        )
        return result


def stress_tensor(voigt: np.ndarray) -> np.ndarray:
    """Convert stress with trailing Voigt axis 6 into trailing matrix axes 3x3."""
    s = np.asarray(voigt)
    if s.ndim < 1 or s.shape[-1] != 6:
        raise ValueError("Stress must have final axis length 6 in Voigt order.")
    result = np.empty(s.shape[:-1] + (3, 3))
    result[..., 0, 0], result[..., 1, 1], result[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    result[..., 0, 1] = result[..., 1, 0] = s[..., 3]
    result[..., 1, 2] = result[..., 2, 1] = s[..., 4]
    result[..., 0, 2] = result[..., 2, 0] = s[..., 5]
    return result


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Compute von Mises stress for single or batched fields, final axis 6."""
    s = np.asarray(stress)
    if s.ndim < 1 or s.shape[-1] != 6:
        raise ValueError("Stress must have final axis length 6 in Voigt order.")
    return np.sqrt(0.5 * ((s[..., 0] - s[..., 1]) ** 2
                          + (s[..., 1] - s[..., 2]) ** 2
                          + (s[..., 2] - s[..., 0]) ** 2)
                   + 3.0 * np.sum(s[..., 3:] ** 2, axis=-1))


def relative_error(value, expected, floor=1.0):
    return float(np.linalg.norm(np.asarray(value) - expected)
                 / max(np.linalg.norm(expected), floor))


def run_validation(subdivisions=3):
    xyz, tets = structured_cube(subdivisions)
    model = CachedElasticRecovery(xyz, tets)
    tests = []

    def check(name, metrics, conditions):
        passed = all(bool(condition) for condition in conditions)
        tests.append({"name": name, "passed": passed, "metrics": metrics})
        if not passed:
            raise AssertionError(f"Validation failed: {name}: {metrics}")

    # Consistent mass gives exact continuum mass, COM and inertia of the cube.
    expected_mass = model.density
    expected_inertia = expected_mass / 6.0 * np.eye(3)
    null_error = np.linalg.norm(model.k @ model.r) / (
        np.linalg.norm(model.k.data) * np.linalg.norm(model.r))
    symmetry_error = np.linalg.norm((model.k - model.k.T).data) / np.linalg.norm(model.k.data)
    mass_error = abs(model.mass - expected_mass) / expected_mass
    com_error = np.linalg.norm(model.com - np.full(3, 0.5))
    inertia_error = relative_error(model.inertia, expected_inertia, 1e-15)
    check("exact_mass_com_inertia_and_rigid_nullspace", {
        "volume": float(model.volumes.sum()), "mass": model.mass,
        "COM": model.com.tolist(), "inertia": model.inertia.tolist(),
        "mass_relative_error": mass_error, "COM_absolute_error": float(com_error),
        "inertia_relative_error": inertia_error,
        "translation_rotation_mass_cross_norm": float(np.linalg.norm(model.g[:3, 3:])),
        "K_R_relative_error": float(null_error),
        "K_symmetry_relative_error": float(symmetry_error),
        "gauge_rigid_mode_condition_number": float(np.linalg.cond(model.r[model.pins])),
    }, [mass_error < 1e-12, com_error < 1e-12, inertia_error < 1e-12,
        null_error < 1e-12, symmetry_error < 1e-12])

    # Off-node surface forces: preservation holds before any inertia relief.
    surface_loads = []
    point_metrics = []
    for face_index, weights, force in [
        (0, [0.2, 0.3, 0.5], [40.0, -15.0, 20.0]),
        (len(model.surface_faces) - 1, [0.45, 0.35, 0.2], [-10.0, 8.0, -25.0]),
    ]:
        point = np.asarray(weights) @ xyz[model.surface_faces[face_index]]
        f = model.point_load(face_index, point, force)
        nodal_force = f.reshape(-1, 3)
        force_error = relative_error(nodal_force.sum(axis=0), np.asarray(force))
        moment_error = relative_error(np.cross(model.offsets, nodal_force).sum(axis=0),
                                      np.cross(point - model.com, force))
        surface_loads.append(f)
        point_metrics.append({"point": point.tolist(), "force": force,
                              "force_relative_error": force_error,
                              "moment_relative_error": moment_error})
    check("barycentric_surface_load_preserves_force_and_moment", {
        "loads": point_metrics,
        "contact_area_is_defined": False,
    }, [item["force_relative_error"] < 1e-12 and item["moment_relative_error"] < 1e-12
        for item in point_metrics])
    contact_force = sum(surface_loads)
    contact = model.solve(contact_force)
    check("inertia_relief_balance_and_full_residual_including_pins", {
        "eta_translation_and_angular_acceleration": contact.rigid_acceleration.tolist(),
        "balanced_wrench": (model.r.T @ contact.compatible_load).tolist(),
        "full_residual_relative": contact.residual_relative,
        "pinned_residual_relative": contact.pinned_residual_relative,
        "wrench_relative": contact.wrench_relative,
    }, [contact.residual_relative < 1e-10,
        contact.pinned_residual_relative < 1e-10, contact.wrench_relative < 1e-12])

    gravity = np.array([0.0, 0.0, -9.81])
    freefall = model.solve(model.consistent_body_force(gravity))
    check("freefall_has_zero_elastic_stress", {
        "maximum_absolute_stress": float(np.abs(freefall.stress).max()),
        "compatible_load_norm": float(np.linalg.norm(freefall.compatible_load)),
        "translation_acceleration": freefall.rigid_acceleration[:3].tolist(),
    }, [np.abs(freefall.stress).max() < 1e-9,
        relative_error(freefall.rigid_acceleration[:3], gravity) < 1e-12])

    # Exact affine strain patch, including three shear components.
    eps = np.array([[0.012, 0.003, -0.002],
                    [0.003, -0.004, 0.0015],
                    [-0.002, 0.0015, 0.005]])
    affine_u = (model.offsets @ eps.T).ravel()
    affine_load = model.k @ affine_u
    affine = model.solve(affine_load)
    mu = model.young / (2.0 * (1.0 + model.poisson))
    lam = model.young * model.poisson / ((1.0 + model.poisson) * (1.0 - 2.0 * model.poisson))
    analytic_tensor = 2.0 * mu * eps + lam * np.trace(eps) * np.eye(3)
    analytic_tensors = np.broadcast_to(analytic_tensor, (len(tets), 3, 3))
    patch_error = relative_error(stress_tensor(affine.stress), analytic_tensors)
    check("affine_strain_patch_matches_analytic_stress_tensor", {
        "strain_tensor": eps.tolist(), "analytic_stress_tensor": analytic_tensor.tolist(),
        "recovered_first_element_stress_tensor": stress_tensor(affine.stress)[0].tolist(),
        "stress_relative_error": patch_error,
        "full_residual_relative": affine.residual_relative,
    }, [patch_error < 1e-10, affine.residual_relative < 1e-10])

    # Two independent pin gauges must give identical stress, despite different u.
    alt_order = np.arange(model.ndof)[::-1]
    alternative = CachedElasticRecovery(xyz, tets, gauge_order=alt_order)
    if set(alternative.pins) == set(model.pins):
        alt_order = np.random.default_rng(123).permutation(model.ndof)
        alternative = CachedElasticRecovery(xyz, tets, gauge_order=alt_order)
    alt_result = alternative.solve(contact_force)
    gauge_stress_error = relative_error(alt_result.stress, contact.stress)
    canonical_displacement_error = relative_error(
        alternative.remove_rigid_displacement(alt_result.displacement),
        model.remove_rigid_displacement(contact.displacement), 1e-15)
    check("gauge_choice_does_not_change_stress", {
        "first_pins": model.pins.tolist(), "second_pins": alternative.pins.tolist(),
        "stress_relative_difference": gauge_stress_error,
        "mass_orthogonal_displacement_relative_difference": canonical_displacement_error,
        "second_full_residual_relative": alt_result.residual_relative,
        "second_pinned_residual_relative": alt_result.pinned_residual_relative,
    }, [set(model.pins) != set(alternative.pins), gauge_stress_error < 1e-10,
        canonical_displacement_error < 1e-10, alt_result.residual_relative < 1e-10,
        alt_result.pinned_residual_relative < 1e-10])

    # Multi-RHS and superposition share exactly one factor for this body.
    rhs = np.column_stack([surface_loads[0], surface_loads[1],
                           surface_loads[0] + surface_loads[1], affine_load])
    batched = model.solve(rhs)
    separate = [model.solve(rhs[:, j]) for j in range(rhs.shape[1])]
    multi_error = relative_error(batched.stress,
                                 np.stack([item.stress for item in separate], axis=0))
    superposition_error = relative_error(batched.stress[2],
                                         batched.stress[0] + batched.stress[1])
    check("superposition_and_batched_rhs_reuse_one_factorization", {
        "number_of_rhs_in_batched_call": rhs.shape[1],
        "batched_vs_separate_stress_relative_error": multi_error,
        "superposition_stress_relative_error": superposition_error,
        "factorization_count": model.factorization_count,
        "solve_call_count_so_far": model.solve_call_count,
        "rhs_solved_so_far": model.rhs_solved,
        "full_residual_relative": batched.residual_relative,
    }, [multi_error < 1e-10, superposition_error < 1e-10,
        model.factorization_count == 1, batched.residual_relative < 1e-10])

    stiffer = CachedElasticRecovery(xyz, tets, young=10.0 * model.young)
    stiff_result = stiffer.solve(contact_force)
    stiffness_stress_error = relative_error(stiff_result.stress, contact.stress)
    stiffness_u_error = relative_error(10.0 * stiff_result.displacement,
                                      contact.displacement, 1e-15)
    check("uniform_E_times_ten_scales_u_but_not_force_driven_stress", {
        "young_modulus_multiplier": 10.0,
        "stress_relative_difference": stiffness_stress_error,
        "ten_times_stiffer_displacement_relative_difference": stiffness_u_error,
    }, [stiffness_stress_error < 1e-10, stiffness_u_error < 1e-10])

    # A spinning, free cube has balanced centrifugal body load and nonzero stress.
    spin = np.array([0.0, 0.0, 7.0])
    spinning = model.solve(np.zeros(model.ndof), omega=spin)
    c = np.cross(spin, np.cross(spin, model.offsets)).ravel()
    centrifugal_load_error = relative_error(spinning.compatible_load, -model.m @ c)
    maximum_vm = float(von_mises(spinning.stress).max())
    check("rotation_centrifugal_stress_is_nonzero_and_balanced", {
        "angular_velocity": spin.tolist(),
        "maximum_von_mises_stress": maximum_vm,
        "compatible_load_norm": float(np.linalg.norm(spinning.compatible_load)),
        "balanced_force_and_torque": (model.r.T @ spinning.compatible_load).tolist(),
        "eta": spinning.rigid_acceleration.tolist(),
        "minus_M_c_relative_error_for_isotropic_inertia": centrifugal_load_error,
        "full_residual_relative": spinning.residual_relative,
        "pinned_residual_relative": spinning.pinned_residual_relative,
        "wrench_relative": spinning.wrench_relative,
    }, [maximum_vm > 1.0, centrifugal_load_error < 1e-12,
        spinning.residual_relative < 1e-10,
        spinning.pinned_residual_relative < 1e-10,
        spinning.wrench_relative < 1e-12])

    # An anisotropic body exposes the gyroscopic torque hidden by a cube's I.
    cuboid = CachedElasticRecovery(xyz * np.array([1.0, 2.0, 3.0]), tets)
    cuboid_spin = np.array([2.0, 3.0, 4.0])
    cuboid_result = cuboid.solve(np.zeros(cuboid.ndof), omega=cuboid_spin)
    expected_cuboid_inertia = np.diag([13.0, 10.0, 5.0])
    gyro_torque = np.cross(cuboid_spin, cuboid.inertia @ cuboid_spin)
    expected_alpha = -np.linalg.solve(cuboid.inertia, gyro_torque)
    alpha_error = relative_error(cuboid_result.rigid_acceleration[3:], expected_alpha)
    check("anisotropic_spin_obeys_newton_euler_gyroscopic_balance", {
        "cuboid_sides": [1.0, 2.0, 3.0], "mass": cuboid.mass,
        "inertia": cuboid.inertia.tolist(), "angular_velocity": cuboid_spin.tolist(),
        "gyroscopic_torque_omega_cross_Iomega": gyro_torque.tolist(),
        "expected_angular_acceleration": expected_alpha.tolist(),
        "recovered_angular_acceleration": cuboid_result.rigid_acceleration[3:].tolist(),
        "angular_acceleration_relative_error": alpha_error,
        "balanced_force_and_torque": (cuboid.r.T @ cuboid_result.compatible_load).tolist(),
        "full_residual_relative": cuboid_result.residual_relative,
        "pinned_residual_relative": cuboid_result.pinned_residual_relative,
    }, [relative_error(cuboid.inertia, expected_cuboid_inertia) < 1e-12,
        alpha_error < 1e-12, cuboid_result.wrench_relative < 1e-12,
        cuboid_result.residual_relative < 1e-10,
        cuboid_result.pinned_residual_relative < 1e-10])

    # nrhs == 6 must not be mistaken for the Voigt axis by stress helpers.
    six_rhs = np.column_stack([surface_loads[0], surface_loads[1], contact_force,
                               affine_load, model.consistent_body_force(gravity),
                               surface_loads[0] - surface_loads[1]])
    six_batch = model.solve(six_rhs)
    six_separate = [model.solve(six_rhs[:, j]) for j in range(6)]
    expected_six_stress = np.stack([item.stress for item in six_separate], axis=0)
    expected_six_tensors = np.stack([stress_tensor(item.stress)
                                     for item in six_separate], axis=0)
    expected_six_vm = np.stack([von_mises(item.stress)
                                for item in six_separate], axis=0)
    six_field_error = relative_error(six_batch.stress, expected_six_stress)
    six_tensor_error = relative_error(stress_tensor(six_batch.stress),
                                      expected_six_tensors)
    six_vm_error = relative_error(von_mises(six_batch.stress), expected_six_vm)
    check("exactly_six_rhs_stress_tensor_and_von_mises_are_consistent", {
        "batch_stress_shape": list(six_batch.stress.shape),
        "batch_tensor_shape": list(stress_tensor(six_batch.stress).shape),
        "batch_von_mises_shape": list(von_mises(six_batch.stress).shape),
        "batched_vs_individual_stress_relative_error": six_field_error,
        "batched_vs_individual_tensor_relative_error": six_tensor_error,
        "batched_vs_individual_von_mises_relative_error": six_vm_error,
    }, [six_batch.stress.shape == (6, len(tets), 6),
        stress_tensor(six_batch.stress).shape == (6, len(tets), 3, 3),
        von_mises(six_batch.stress).shape == (6, len(tets)),
        six_field_error < 1e-10, six_tensor_error < 1e-10, six_vm_error < 1e-10])

    return {
        "all_tests_passed": all(test["passed"] for test in tests),
        "execution": {
            "backend": "CPU NumPy/SciPy", "genesis_integration_tested": False,
            "gpu_implementation_tested": False, "gpu_performance_measured": False,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "model": {
            "type": "fixed P1 tetrahedral isotropic small-strain elasticity",
            "cube_side": 1.0, "subdivisions_per_axis": subdivisions,
            "vertices": len(xyz), "tetrahedra": len(tets), "scalar_DOFs": model.ndof,
            "young_modulus": model.young, "poisson_ratio": model.poisson,
            "density": model.density, "consistent_mass": True,
            "stress_voigt_order": VOIGT_ORDER,
            "single_stress_shape": "(n_element, 6)",
            "batched_stress_shape": "(n_rhs, n_element, 6)",
            "single_displacement_and_load_shape": "(n_dof,)",
            "batched_displacement_and_load_shape": "(n_dof, n_rhs)",
            "point_load_coordinates": "same original body-local coordinates as xyz; not necessarily COM-centered",
            "engineering_shear_strain": True,
            "cached_stiffness_factorization": "SciPy SuperLU on gauge-fixed K",
            "factorization_count_for_primary_body": model.factorization_count,
            "total_solve_calls_for_primary_body": model.solve_call_count,
            "total_rhs_for_primary_body": model.rhs_solved,
        },
        "tests": tests,
        "diagnostics": {
            "batch_norms": "Aggregate the entire batch, not individual RHS error bounds.",
            "wrench_norm": "Combines force and torque in chosen units; internal algebra check only, not a physical relative error.",
        },
        "limitations": [
            "Mathematical CPU reference; no Genesis or GPU integration/performance validation.",
            "Single floating material body: six null modes assumed; disconnected meshes need more gauges.",
            "All external forces must use body-local axes; point_load positions use the original xyz coordinates, not necessarily COM-centered.",
            "Quasi-static elastic perturbation ignores stress waves, elastic history and contact deformation feedback.",
            "Rigid-reference inertia omits elastic acceleration, Coriolis terms and spin-dependent elastic stiffness.",
            "Barycentric rigid point loads do not supply a physical contact patch or convergent peak stress.",
            "Fixed linear material and mesh; near-incompressibility can cause P1 volumetric locking.",
            "Use physical units consistently; stiffness and mass must represent the same body.",
            "Force-driven stress is independent of a uniform E scale at fixed nu in this uncoupled model.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation-json", type=Path,
                        default=Path(__file__).with_name("rigid_stress_validation.json"))
    parser.add_argument("--subdivisions", type=int, default=3)
    args = parser.parse_args()
    report = run_validation(args.subdivisions)
    args.validation_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_tests_passed": report["all_tests_passed"],
                      "tests": len(report["tests"]), "model": report["model"],
                      "validation_json": str(args.validation_json)}, indent=2))


if __name__ == "__main__":
    main()
