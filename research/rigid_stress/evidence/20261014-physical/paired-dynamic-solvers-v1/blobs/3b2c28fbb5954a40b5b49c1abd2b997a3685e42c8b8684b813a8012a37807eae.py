"""Use a cached LU only as a preconditioner for the current exact Newton matrix."""

import atexit
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .bordered_metis_reference_factor import factorize

RECORDS = []


def save_records():
    path = os.environ.get("REFERENCE_LINEAR_SOLVER_STATS_PATH")
    if path:
        Path(path).write_text(json.dumps({"linear_solves": RECORDS}, indent=2) + "\n")


atexit.register(save_records)


def cached_newton_solve(system, matrix, rhs, phase):
    start = perf_counter()
    tag = (phase.rstrip("0123456789"), matrix.shape)
    cached = getattr(system, "_reference_newton_factor", None)
    record = {
        "phase": phase,
        "dimension": len(rhs),
        "fresh_factorization": False,
        "gmres_iterations": 0,
        "gmres_info": None,
    }
    budget = max(1e-12, 1e-11 * float(np.linalg.norm(rhs)))
    if cached is None or cached[0] != tag:
        factor = factorize(matrix.tocsc(), phase)
        system._reference_newton_factor = (tag, factor)
        result = factor.solve(rhs)
        record["fresh_factorization"] = True
    else:
        factor = cached[1]
        preconditioner = LinearOperator(matrix.shape, matvec=factor.solve, dtype=float)
        history = []
        result, info = gmres(
            matrix,
            rhs,
            M=preconditioner,
            restart=40,
            maxiter=2,
            rtol=1e-11,
            atol=1e-12,
            callback=history.append,
            callback_type="pr_norm",
        )
        residual = float(np.linalg.norm(matrix @ result - rhs))
        record.update(gmres_iterations=len(history), gmres_info=int(info), gmres_true_residual=residual)
        if info or not np.isfinite(residual) or residual > budget:
            factor = factorize(matrix.tocsc(), phase)
            system._reference_newton_factor = (tag, factor)
            result = factor.solve(rhs)
            record["fresh_factorization"] = True
            record["gmres_fallback"] = True
    record["linear_true_residual"] = float(np.linalg.norm(matrix @ result - rhs))
    record["gmres_acceptance_budget"] = budget
    record["elapsed_s"] = perf_counter() - start
    RECORDS.append(record)
    return result
