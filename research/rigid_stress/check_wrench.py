"""Verify finite-footprint wrench conservation, positivity and friction constraints on the full shell.

Prescribed loads include changing centers, tangential forces, radii and pure moments. Rejected tensile and unresolved
loads are explicit failures of the selected physical footprint law, rather than silently projected rigid wrenches.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU, reference
from .wrench import FinitePatchMapper, PadPressureFit, WrenchFit, WrenchPatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(config=EggConfig(level=args.level), history=0, direct=True)
        mapper = FinitePatchMapper(model.surface)
        rng = np.random.default_rng(91391)
        for i_case in range(32):
            face = rng.integers(len(model.fem.outer_faces))
            center, normal, tangent, _ = reference.surface_frame(model.fem, face, rng.dirichlet([2, 2, 2]))
            radius = rng.uniform(0.004, 0.009)
            force = -normal * rng.uniform(0.5, 3) + tangent * rng.uniform(-0.15, 0.15)
            patch = WrenchPatch(center, force, radius, 0.6, np.zeros(3))
            result = mapper.map(patch)
            nodal = result.nodal_force_n.reshape(-1, 3)
            actual_force = nodal.sum(axis=0)
            actual_moment = np.cross(model.fem.xyz, nodal).sum(axis=0)
            expected_moment = np.cross(center, force)
            force_error = np.linalg.norm(actual_force - force) / np.linalg.norm(force)
            moment_error = np.linalg.norm(actual_moment - expected_moment) / (radius * np.linalg.norm(force))
            assert force_error <= 1e-8
            assert moment_error <= 1e-8
            assert result.diagnostics.minimum_normal_force_n >= -1e-14
            assert result.diagnostics.maximum_cone_excess_n <= 1e-14
            rows.append(
                {
                    "case": i_case,
                    "center_m": center.tolist(),
                    "force_n": force.tolist(),
                    "radius_m": radius,
                    "force_error_normalized": float(force_error),
                    "moment_error_normalized": float(moment_error),
                    **asdict(result.diagnostics),
                }
            )
        # A known admissible distribution supplies an independent six-component target
        ids = result.quadrature_idx
        force_samples = result.quadrature_force_n.copy()
        perturbation = (0.5 + 0.5 * np.sin(np.arange(len(ids))))[:, None] * mapper.normals[ids] * -0.001
        force_samples += perturbation
        force = force_samples.sum(axis=0)
        moment = np.cross(mapper.positions[ids] - center, force_samples).sum(axis=0)
        patch = WrenchPatch(center, force, radius, 0.6, moment)
        fitted = mapper.map(patch)
        nodal = fitted.nodal_force_n.reshape(-1, 3)
        np.testing.assert_allclose(nodal.sum(axis=0), force, atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(np.cross(model.fem.xyz - center, nodal).sum(axis=0), moment, atol=1e-10, rtol=1e-8)
        fit = WrenchFit(mapper.positions[ids], np.ones(len(ids)), mapper.normals[ids], patch)
        multiplier = rng.normal(scale=0.4, size=6)
        numerical = np.column_stack(
            [
                (fit.residual(multiplier + np.eye(6)[i_d] * 1e-6) - fit.residual(multiplier - np.eye(6)[i_d] * 1e-6))
                / 2e-6
                for i_d in range(6)
            ]
        )
        np.testing.assert_allclose(fit.jacobian(multiplier), numerical, atol=1e-8, rtol=1e-6)
        positions = np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]) * 0.003
        normals = np.tile([0, 0, 1], (4, 1))
        force = np.array([0.3, 0, 2])
        moment = np.cross(np.array([0.0027, 0.0027, 0]), force)
        constrained_patch = WrenchPatch(np.zeros(3), force, 0.005, 0.3, moment)
        constrained_force, constrained_diagnostics = WrenchFit(
            positions, np.ones(4), normals, constrained_patch
        ).solve()
        assert constrained_diagnostics.constrained_evaluations > 0
        assert constrained_diagnostics.minimum_normal_force_n >= -1e-14
        assert constrained_diagnostics.maximum_cone_excess_n <= 1e-14
        np.testing.assert_allclose(constrained_force.sum(axis=0), force, atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(np.cross(positions, constrained_force).sum(axis=0), moment, atol=1e-10, rtol=1e-8)
        pad_center = np.array([0.0027, 0.0027, 0.0])
        pad_force = np.array([0.6, 0.0, 2.0])
        pad_patch = WrenchPatch(pad_center, pad_force, 0.009, 0.3, np.zeros(3), np.array([0, 0, 1]))
        pad_forces, pad_diagnostics = PadPressureFit(positions, np.ones(4), pad_patch).solve()
        assert pad_diagnostics.constrained_evaluations > 1
        assert pad_diagnostics.minimum_normal_force_n >= -1e-14
        assert pad_diagnostics.maximum_cone_excess_n <= 1e-14
        np.testing.assert_allclose(pad_forces.sum(axis=0), pad_force, atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(np.cross(positions - pad_center, pad_forces).sum(axis=0), 0, atol=1e-10)
        for i_case in range(16):
            face = rng.integers(len(model.fem.outer_faces))
            center, normal, tangent, _ = reference.surface_frame(model.fem, face, rng.dirichlet([2, 2, 2]))
            radius = rng.uniform(0.004, 0.009)
            force = -normal * 2 + tangent * 1.2
            pad = mapper.map(WrenchPatch(center, force, radius, 0.6, np.zeros(3), -normal))
            nodal = pad.nodal_force_n.reshape(-1, 3)
            np.testing.assert_allclose(nodal.sum(axis=0), force, atol=1e-9, rtol=1e-8)
            np.testing.assert_allclose(np.cross(model.fem.xyz - center, nodal).sum(axis=0), 0, atol=1e-10)
            assert pad.diagnostics.minimum_normal_force_n >= -1e-14
            assert pad.diagnostics.maximum_cone_excess_n <= 1e-14
        is_tensile_rejected = False
        try:
            mapper.map(WrenchPatch(center, normal, radius, 0.6, np.zeros(3)))
        except ValueError:
            is_tensile_rejected = True
        assert is_tensile_rejected
    report = {
        "passed": True,
        "scope": "CPU full-shell finite patch wrench tests; no live Genesis contacts or physical mesh convergence",
        "dofs": model.fem.ndof,
        "physical_mesh_converged": False,
        "gpu_mapping_tested": False,
        "known_distribution_moment_nm": moment.tolist(),
        "known_distribution_diagnostics": asdict(fitted.diagnostics),
        "constrained_fit_diagnostics": asdict(constrained_diagnostics),
        "constant_ratio_pad_diagnostics": asdict(pad_diagnostics),
        "pad_shell_cases": 16,
        "tensile_wrench_rejected": is_tensile_rejected,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
