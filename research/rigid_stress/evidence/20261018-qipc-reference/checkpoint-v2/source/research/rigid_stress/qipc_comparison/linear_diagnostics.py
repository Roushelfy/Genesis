"""Bounded sparse-direct checks of the original solver's recorded linear residuals."""

import numpy as np

from qipc.diag import ReferencePCG


def linear_diagnostics(scene, output):
    diagnostic = scene.solver.extras.pcg_diagnostics
    diagnostic.save(output / "linear-system.npz")
    rows = []
    for iteration in range(diagnostic.n_systems):
        indices = diagnostic.newton_records(iteration)
        system = diagnostic.system(iteration)
        if not len(indices) or system.truncated:
            rows.append({"newton": iteration, "system_truncated": system.truncated, "records": len(indices)})
            continue
        reference = ReferencePCG(system)
        initial, final = diagnostic.r[indices[[0, -1]]]
        initial_norm = np.linalg.norm(initial)
        final_norm = np.linalg.norm(final)
        correction = reference.x_star
        error_squared = reference.true_error(final[None])["total"][0]
        rows.append(
            {
                "newton": iteration,
                "final_pcg_iteration": int(diagnostic.meta[indices[-1], 1]),
                "initial_residual_incremental_l2": float(initial_norm),
                "final_recursive_residual_incremental_l2": float(final_norm),
                "relative_recursive_residual": float(final_norm / initial_norm) if initial_norm else None,
                "direct_correction_max_m": float(np.abs(correction).max()),
                "recursive_residual_direct_error_l2_m": float(np.sqrt(error_squared)),
                "direct_system_relative_residual": float(
                    np.linalg.norm(reference.A @ correction - reference.b) / np.linalg.norm(reference.b)
                )
                if np.linalg.norm(reference.b)
                else None,
                "system_truncated": False,
            }
        )
    return {"records_dropped": diagnostic.n_dropped, "systems": rows}
