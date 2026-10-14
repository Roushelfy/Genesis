"""Qualify reference solve backends on an existing full3D pressure-shell mesh."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyamg
from scipy.sparse.linalg import cg, splu
from skfem import FacetBasis, asm

from .skfem_reference import SkfemElasticReference
from .spherical_shell_validation import analytic_shell, pressure_load


class AMGPinFactor:
    def __init__(self, operator, modes):
        self.operator = operator
        self.hierarchy = pyamg.smoothed_aggregation_solver(
            operator, B=modes, symmetry="symmetric", max_coarse=300, coarse_solver="splu"
        )
        self.preconditioner = self.hierarchy.aspreconditioner()
        self.residuals = []

    def solve(self, rhs):
        self.residuals = []

        def record(value):
            self.residuals.append(float(np.linalg.norm(self.operator @ value - rhs)))

        solved, code = cg(self.operator, rhs, M=self.preconditioner, rtol=1e-12, atol=0, maxiter=5000, callback=record)
        if code != 0:
            raise RuntimeError(f"Independent AMG-CG did not converge: {code}; last residual={self.residuals[-1]}")
        return solved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--solver", choices=("amg-cg", "superlu-symmetric"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    with np.load(args.mesh) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    reference = SkfemElasticReference(vertices, tets, 1e10, 0.3, 2000)
    free = reference.free_dofs
    operator = (reference.stiffness[free][:, free] / reference.young).tocsr()
    build_started = perf_counter()
    solver_record = {"method": args.solver}
    if args.solver == "amg-cg":
        np.random.seed(2026101403)
        modes = reference.rigid_modes[free]
        modes = modes / np.linalg.norm(modes, axis=0)
        factor = AMGPinFactor(operator, modes)
        reference.factor = factor
        solver_record.update(
            {
                "levels": len(factor.hierarchy.levels),
                "operator_complexity": factor.hierarchy.operator_complexity(),
                "pyamg_version": importlib.metadata.version("pyamg"),
            }
        )
    else:
        reference.factor = splu(
            operator.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0, options={"SymmetricMode": True}
        )
        solver_record["factor_nonzeros"] = reference.factor.L.nnz + reference.factor.U.nnz
    solver_record["construction_seconds"] = perf_counter() - build_started
    boundary = reference.mesh.boundary_facets()
    radii = np.linalg.norm(reference.mesh.p[:, reference.mesh.facets[:, boundary]], axis=0)
    inner = boundary[np.max(radii, axis=0) < 0.010]
    force = asm(pressure_load, FacetBasis(reference.mesh, reference.element, facets=inner, intorder=4), pressure=1e4)
    solved = reference.solve(force)
    if solved.residual_budget_fraction > 1:
        raise RuntimeError(f"Complete equilibrium failed: {solved.residual_budget_fraction}")
    directions = np.random.default_rng(2026101402).normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    probes = 0.0096 * np.concatenate((directions, np.eye(3), -np.eye(3)))
    tensor, vm = reference.stress_at_points(solved.displacement, probes)
    expected_u, expected_tensor, expected_vm = analytic_shell(probes, 0.008, 0.012, 1e4, 1e10, 0.3)
    actual_u = (reference.basis.probes(probes.T) @ solved.displacement).reshape(3, -1).T
    boundary_u, _, peak = analytic_shell(np.array([[0.008, 0, 0]]), 0.008, 0.012, 1e4, 1e10, 0.3)
    exact_energy = 0.5 * 4 * np.pi * 0.008**2 * 1e4 * boundary_u[0, 0]
    if args.solver == "amg-cg":
        solver_record["free_scaled_residual_history"] = factor.residuals
        solver_record["iterations"] = len(factor.residuals)
    np.savez_compressed(
        args.output / "field.npz",
        displacement=solved.displacement,
        tensor_Pa=solved.tensor_pa,
        von_mises_Pa=solved.von_mises_pa,
        probes_m=probes,
        probe_tensor_Pa=tensor,
        probe_vm_Pa=vm,
        exact_probe_tensor_Pa=expected_tensor,
    )
    report = {
        "scope": "Independent scikit-fem pressure-shell reference solver scaling qualification",
        "mesh": str(args.mesh.resolve()),
        "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "tetrahedra": len(tets),
        "dofs": reference.basis.N,
        "solver": solver_record,
        "peak_Pa": float(solved.peak_pa),
        "exact_peak_Pa": float(peak[0]),
        "peak_position_m": solved.peak_position_m.tolist(),
        "peak_relative_error": float(abs(solved.peak_pa / peak[0] - 1)),
        "energy_J": float(solved.energy_j),
        "energy_relative_error": float(abs(solved.energy_j / exact_energy - 1)),
        "common_probe_tensor_relative_error": float(
            np.linalg.norm(tensor - expected_tensor) / np.linalg.norm(expected_tensor)
        ),
        "common_probe_vm_relative_error": float(np.linalg.norm(vm - expected_vm) / np.linalg.norm(expected_vm)),
        "common_probe_displacement_relative_error": float(
            np.linalg.norm(actual_u - expected_u) / np.linalg.norm(expected_u)
        ),
        "complete_residual_N": float(solved.residual_n),
        "complete_residual_budget_fraction": float(solved.residual_budget_fraction),
        "elapsed_seconds": perf_counter() - started,
        "field_sha256": hashlib.sha256((args.output / "field.npz").read_bytes()).hexdigest(),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
