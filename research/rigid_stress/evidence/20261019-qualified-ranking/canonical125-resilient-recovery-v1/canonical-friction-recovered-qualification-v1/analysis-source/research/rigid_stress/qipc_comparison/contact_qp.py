"""Independent primal-dual certificates for condensed quadratic plate contact."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

from .p2_elastic import quadratic_triangle_minimum, quadratic_triangle_weights


@dataclass
class ContactQPCertificate:
    interpolation: np.ndarray
    multipliers: np.ndarray
    coordinates: np.ndarray
    nodal_force: np.ndarray
    constraint_gap: np.ndarray
    rounds: list
    status: str
    iterations: int
    solve_seconds: float
    force_error: float
    stationarity: float
    complementarity: float
    minimum_gap: float
    fixed_passed: bool
    solver_version: str
    solver_module: str

    @property
    def candidate_passed(self):
        return self.fixed_passed and self.minimum_gap >= -1e-8


def solve_contact_qp(
    compliance,
    gap,
    rigid,
    target,
    triangles,
    package_path,
    interpolation=None,
    round_limit=20,
    form="whitened-primal",
    journal=None,
    constraint_compliance=None,
):
    """Certify hard contact or a declared positive series compliance at fixed quadrature points."""
    assert round_limit > 0 and form in ("dual", "whitened-primal")
    assert target.shape == (4,) and rigid.shape == (len(gap), 4)
    sys.path.insert(0, str(package_path))
    import clarabel

    assert clarabel.__version__ == "0.11.1"
    if interpolation is None:
        interpolation = np.eye(len(gap))
    if constraint_compliance is not None:
        constraint_compliance = np.asarray(constraint_compliance)
        assert constraint_compliance.shape == (len(interpolation),)
        assert np.isfinite(constraint_compliance).all() and (constraint_compliance > 0).all()
        # Negative body gaps are physical pad compression. The hard-contact
        # gap oracle cannot be applied to this different constitutive model.
        assert not len(triangles) and round_limit == 1
    pad_root = np.sqrt(constraint_compliance) if constraint_compliance is not None else np.empty(0)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = 1e-12
    settings.tol_gap_rel = 1e-12
    settings.tol_feas = 1e-12
    settings.iterative_refinement_abstol = 1e-14
    settings.iterative_refinement_reltol = 1e-14
    factor = np.linalg.cholesky(compliance) if form == "whitened-primal" else None
    rounds = []
    for iteration in range(round_limit):
        linear = interpolation @ gap
        equality = interpolation @ rigid
        if factor is None:
            matrix = interpolation @ compliance @ interpolation.T
            if constraint_compliance is not None:
                matrix[np.diag_indices_from(matrix)] += constraint_compliance
            solver = clarabel.DefaultSolver(
                sparse.triu(sparse.csc_matrix(matrix), format="csc"),
                linear,
                sparse.vstack((sparse.csc_matrix(equality.T), -sparse.eye(len(linear))), format="csc"),
                np.r_[target, np.zeros(len(linear))],
                [clarabel.ZeroConeT(4), clarabel.NonnegativeConeT(len(linear))],
                settings,
            )
        else:
            # v=L y with C=L L^T gives elastic energy y^T y/2.
            # Minimize that energy + target^T q subject to R(v+g+Qq)>=0.
            # Its nonnegative dual variables are the same compressive
            # point forces, without a rank-deficient R C R^T Hessian.
            constraint = sparse.csc_matrix(-interpolation @ np.column_stack((factor, rigid)))
            if len(pad_root):
                constraint = sparse.hstack((constraint, -sparse.diags(pad_root)), format="csc")
            solver = clarabel.DefaultSolver(
                sparse.diags(np.r_[np.ones(len(gap)), np.zeros(4), np.ones(len(pad_root))], format="csc"),
                np.r_[np.zeros(len(gap)), target, np.zeros(len(pad_root))],
                constraint,
                linear,
                [clarabel.NonnegativeConeT(len(linear))],
                settings,
            )
        solution = solver.solve()
        multipliers = np.array(solution.x if factor is None else solution.z)
        coordinates = np.array(solution.z[:4] if factor is None else solution.x[len(gap) : len(gap) + 4])
        alpha = interpolation.T @ multipliers
        nodal_gap = compliance @ alpha + gap + rigid @ coordinates
        constraint_gap = interpolation @ nodal_gap
        if constraint_compliance is not None:
            constraint_gap += constraint_compliance * multipliers
        wrench = float(np.linalg.norm(equality.T @ multipliers - target))
        if factor is None:
            stationarity = float(np.linalg.norm(constraint_gap - np.asarray(solution.z[4:])))
        else:
            elastic_coordinate = np.array(solution.x[: len(gap)])
            stationarity = float(
                max(
                    np.linalg.norm(elastic_coordinate - factor.T @ alpha),
                    np.linalg.norm(constraint_gap - np.array(solution.s)),
                )
            )
            if len(pad_root):
                stationarity = max(
                    stationarity,
                    float(np.linalg.norm(np.array(solution.x[len(gap) + 4 :]) - pad_root * multipliers)),
                )
        complementarity = float(np.max(np.abs(multipliers * constraint_gap)))
        if len(triangles):
            values, bary = quadratic_triangle_minimum(nodal_gap[triangles], return_points=True)
            minimum_gap = float(values.min())
        else:
            minimum_gap = float(constraint_gap.min())
        fixed_passed = bool(
            solution.status == clarabel.SolverStatus.Solved
            and max(wrench, stationarity, complementarity) <= 1e-8
            and multipliers.min() >= -1e-10
            and constraint_gap.min() >= -1e-10
        )
        rounds.append(
            {
                "round": iteration + 1,
                "constraints": len(interpolation),
                "status": str(solution.status),
                "fixed_qp_certificate_passed": fixed_passed,
                "minimum_candidate_face_gap_scaled": minimum_gap,
                "solve_seconds": solution.solve_time,
            }
        )
        if journal is not None:
            journal.write(json.dumps(rounds[-1], allow_nan=False) + "\n")
        if not fixed_passed or minimum_gap >= -1e-8 or iteration + 1 == round_limit:
            break
        selected = np.flatnonzero(values < -1e-8)
        points = bary[selected]
        weights = quadratic_triangle_weights(points)
        added = np.zeros((len(selected), len(gap)))
        added[np.arange(len(selected))[:, None], triangles[selected]] = weights
        unique = []
        for row in added:
            if np.max(np.abs(interpolation - row), axis=1).min() <= 1e-12:
                continue
            if not unique or np.max(np.abs(np.array(unique) - row), axis=1).min() > 1e-12:
                unique.append(row)
        if not unique:
            break
        interpolation = np.concatenate((interpolation, unique))
    return ContactQPCertificate(
        interpolation,
        multipliers,
        coordinates,
        alpha,
        constraint_gap,
        rounds,
        str(solution.status),
        solution.iterations,
        solution.solve_time,
        wrench,
        stationarity,
        complementarity,
        minimum_gap,
        fixed_passed,
        clarabel.__version__,
        str(Path(clarabel.__file__).resolve()),
    )
