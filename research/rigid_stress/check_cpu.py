"""Numerical checks for imported CPU recovery and the contact-facing adapter.

Run from repo root: python -m research.rigid_stress.check_cpu
Optional: --torch to compare the CUDA-capable prototype on CPU, not GPU.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from general_peak import baseline_numpy
from threadpoolctl import threadpool_limits

from .cpu import ContactBatch, EggConfig, EggRecoveryCPU
from .oracle import reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(4, EggConfig(level=1, layers=2), rtol=1e-6)
        f = model.fem
        faces = [0, 13, 41, 75]
        frames = [reference.surface_frame(f, i, [0.2, 0.3, 0.5]) for i in faces]
        points = np.array([frame[0] for frame in frames])
        forces = np.array([-frame[1] * (1 + i) + frame[2] * 0.15 for i, frame in enumerate(frames)])
        contacts = ContactBatch(
            points[:, None],
            forces[:, None],
            np.full((4, 1), 0.009),
            np.array([[True], [True], [True], [False]]),
            np.full((4, 1), 0.6),
        )
        loads = model.map_contacts(contacts)
        expected_force = (contacts.force_n * contacts.valid[..., None]).sum(axis=1)
        np.testing.assert_allclose(loads.resultant_force_n, expected_force, atol=1e-12)
        np.testing.assert_allclose(loads.resultant_moment_nm, loads.input_moment_nm, atol=1e-12)
        # The same gravity vector is broadcast to each independent environment.
        raw = loads.nodal_force_n + (f.m @ np.tile([0.0, 0.0, -9.81], len(f.xyz)))[:, None]
        rhs = model.compatible_rhs(raw, np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 4.0], [2.0, -1.0, 3.0], [0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(f.r.T @ rhs, 0, atol=1e-11)
        expected = np.zeros_like(rhs)
        expected[f.free] = f.factor.solve(np.asfortranarray(rhs[f.free]))
        recovered = model.recover(rhs)
        np.testing.assert_allclose(f.k @ (recovered.displacement_m - expected), 0, atol=1e-8)
        mu = f.young / (2 * (1 + f.poisson))
        expected_peak = np.array([baseline_numpy(f.glambda, f.elements, expected[:, e], mu)[0] for e in range(4)])
        np.testing.assert_allclose(recovered.peak_pa, expected_peak, rtol=1e-8, atol=1e-6)
        # Unknown future contacts never enter the predictor; reuse only current
        # complete RHS plus each environment's own history.
        maximum_error = 0.0
        for step in range(12):
            moving = rhs * (1 + 0.03 * np.sin(0.2 * step + np.arange(4)))
            out = model.recover(moving)
            truth = expected_peak * (1 + 0.03 * np.sin(0.2 * step + np.arange(4)))
            maximum_error = max(maximum_error, np.max(abs(out.peak_pa - truth) / np.maximum(truth, 1.0)))
        model.reset([1, 3])
        np.testing.assert_array_equal(model.solver.last[:, [1, 3]], 0)
        assert not model.solver.history[1].order
        zero = model.recover(np.zeros_like(rhs))
        np.testing.assert_array_equal(zero.peak_pa, 0)
        # Scalar pins are a gauge, not contact supports: independent gauge
        # changes cannot alter strains or the peak on the complete domain.
        alternate, _, _ = f.solve(raw[:, 0], alternate_gauge=True)
        original, _, _ = f.solve(raw[:, 0])
        np.testing.assert_allclose(model.peak(alternate)[0], model.peak(original)[0], rtol=1e-8)
        result = {
            "passed": True,
            "GPU_tested": False,
            "genesis_scene_tested": False,
            "ndof": f.ndof,
            "tets": f.ne,
            "full_model": True,
            "stress_mesh_converged": False,
            "max_temporal_peak_relative_error": float(maximum_error),
            "max_first_full_relative_residual": float(recovered.relative_residual.max()),
            "finite_patch_point_moment_difference_Nm": np.linalg.norm(
                loads.resultant_moment_nm - loads.input_moment_nm, axis=1
            ).tolist(),
        }
        if args.torch:
            # Optional Torch dependency, isolated from the required CPU stack.
            import torch

            from .gpu_prototype import DenseDebugFactor, P2PeakTorch, compact_correct

            factor = DenseDebugFactor.build(f.k[f.free][:, f.free], device="cpu")
            torch_rhs = torch.tensor(rhs[f.free], dtype=torch.float64)
            x = factor.solve(torch_rhs)
            np.testing.assert_allclose(x.numpy(), expected[f.free], rtol=1e-6, atol=1e-12)
            peak = P2PeakTorch(f.glambda, f.elements, mu, device="cpu", tile=37)
            actual_peak = peak(torch.tensor(expected.T.reshape(4, -1, 3), dtype=torch.float64))
            np.testing.assert_allclose(actual_peak.numpy(), expected_peak, rtol=1e-10, atol=1e-6)
            x0 = torch.zeros_like(x)
            failed = torch.tensor([True, False, True, False])
            compact_correct(factor, torch_rhs, failed, x0)
            np.testing.assert_allclose(x0[:, [0, 2]].numpy(), expected[f.free][:, [0, 2]], rtol=1e-6, atol=1e-12)
            np.testing.assert_array_equal(x0[:, [1, 3]].numpy(), 0)
            compact_correct(factor, torch_rhs, torch.zeros(4, dtype=torch.bool), x0)
            result["torch_prototype_CPU_tested"] = True
        encoded = json.dumps(result, indent=2)
        print(encoded)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
