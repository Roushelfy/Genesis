"""Diagnose large GPU eigenvalue batches with finite known tensors and verify chunked physical stress."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .artifacts import runtime_identity, write_json
from .elastic import evaluate, snh_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=61440)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    angle = 0.2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
    stretch = np.broadcast_to([1.001, 0.9995, 1.0002], (args.elements, 3)).copy()
    stretch[:, 0] += np.arange(args.elements) % 7 * 1e-5
    deformation = rotation[None] * stretch[:, None]
    mu, lame = snh_parameters(1e10, 0.3)
    squared = (stretch * stretch).sum(axis=1)
    J = stretch.prod(axis=1)
    principal = mu * (1 - 1 / (squared[:, None] + 1)) * stretch**2 / J[:, None] + lame * (
        J[:, None] - 1 - 3 * mu / (4 * lame)
    )
    expected = (rotation[None] * principal[:, None]) @ rotation.T
    tensor = torch.tensor(expected, device="cuda")
    assert torch.isfinite(tensor).all().item()
    direct_error = None
    try:
        torch.linalg.eigvalsh(tensor)
        torch.cuda.synchronize()
    except RuntimeError as error:
        direct_error = str(error)
    F = torch.tensor(deformation, device="cuda")
    field = evaluate(F, F.new_tensor(mu), F.new_tensor(lame))
    actual = field.cauchy_pa.cpu().numpy()
    maximum = field.maximum_principal_pa.cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-4)
    np.testing.assert_allclose(maximum, principal.max(axis=1), rtol=2e-11, atol=1e-4)
    assert (
        torch.isfinite(field.first_piola_pa).all().item()
        and torch.isfinite(field.energy_density_relative_rest_j_per_m3).all().item()
    )
    write_json(
        args.output / "result.json",
        {
            "kind": "finite-large-batch-stress-observer-control",
            "identity": runtime_identity(),
            "elements": args.elements,
            "unchunked_gpu_eigenvalue_error_on_known_finite_tensor": direct_error,
            "maximum_cauchy_difference_pa": np.abs(actual - expected).max().item(),
            "maximum_principal_difference_pa": np.abs(maximum - principal.max(axis=1)).max().item(),
            "analytic_rotated_diagonal_SNH_control_passed": True,
            "scope": "Known finite double tensors establish whether the large eigensolver call itself fails. Independent NumPy scalar SNH formula validates pointwise physical Cauchy and principal stress after observer batching; no native energy, gradient, Hessian, contact or solve parameter is changed.",
        },
    )


if __name__ == "__main__":
    main()
