"""Cross-check clean P2 operators and reordered factors against preserved independent CPU math."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU
from .oracle import reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with threadpool_limits(limits=1):
        xyz, tets, outer, _ = reference.egg.egg_mesh(args.level, 2, 0.0005)
        oracle = reference.conv.FEM(
            xyz, tets, outer, order=2, young=1e10, poisson=0.3, density=2000, quarter=False,
            label="preserved-independent-reference",
        )
        rng = np.random.default_rng(28281)
        raw = rng.normal(size=(oracle.ndof, 12))
        rhs = oracle.balance(raw)
        expected = np.zeros_like(rhs)
        expected[oracle.free] = oracle.factor.solve(rhs[oracle.free])
        expected_peak = np.array([reference.peak(oracle, expected[:, i]) for i in range(12)])
        for ordering in ("mmd", "column-nd"):
            model = EggRecoveryCPU(12, EggConfig(level=args.level, ordering=ordering), history=0, direct=True)
            fem = model.fem
            np.testing.assert_array_equal(fem.xyz, oracle.xyz)
            np.testing.assert_array_equal(fem.elements, oracle.elements)
            stiffness_difference = fem.k - oracle.k
            mass_difference = fem.m - oracle.m
            stiffness_error = float(np.max(abs(stiffness_difference.data), initial=0) / np.max(abs(oracle.k.data)))
            mass_error = float(np.max(abs(mass_difference.data), initial=0) / np.max(abs(oracle.m.data)))
            assert stiffness_error < 1e-14 and mass_error < 1e-14
            np.testing.assert_allclose(fem.r, oracle.r, atol=1e-17, rtol=1e-14)
            np.testing.assert_allclose(fem.gram, oracle.gram, atol=1e-18, rtol=1e-13)
            np.testing.assert_allclose(fem.balance(raw), rhs, atol=1e-12, rtol=1e-10)
            recovered = model.recover(rhs)
            peak_error = abs(recovered.peak_pa - expected_peak) / np.maximum(expected_peak, 1)
            assert peak_error.max() < 1e-8
            strain_difference = np.linalg.norm(fem.k @ (recovered.displacement_m - expected), axis=0)
            assert np.max(strain_difference / np.linalg.norm(rhs, axis=0)) < 1e-8
            surface = reference.SurfaceGeometry(oracle, model.config.surface_quadrature)
            np.testing.assert_allclose(model.surface.coords, surface.coords, atol=1e-17, rtol=1e-14)
            np.testing.assert_array_equal(model.surface.nodes, surface.nodes)
            rows.append({
                "ordering": ordering,
                "stiffness_entry_scaled_error": stiffness_error,
                "mass_entry_scaled_error": mass_error,
                "peak_relative_error_max": float(peak_error.max()),
                "full_residual_relative_max": float(recovered.relative_residual.max()),
                "assembly_seconds": fem.assembly_s,
                "ordering_seconds": fem.ordering_s,
                "factor_seconds": fem.factor_s,
                "factor_nonzeros": fem.factor.L.nnz + fem.factor.U.nnz,
            })
            print(json.dumps(rows[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "level": args.level, "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
