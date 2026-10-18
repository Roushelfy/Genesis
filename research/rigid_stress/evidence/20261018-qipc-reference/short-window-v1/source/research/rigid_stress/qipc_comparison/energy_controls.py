"""Compare the observer with the actual unmodified generated C++ energy and derivative."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .artifacts import file_id, write_json
from .elastic import evaluate, snh_energy, snh_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    probe = args.output / "energy-probe"
    command = [
        "/usr/bin/c++",
        "-O2",
        "-std=c++20",
        "-I" + str(args.source / "qipc/_src/native"),
        "-I" + str(args.source / "external/eigen"),
        str(Path(__file__).with_name("energy_probe.cpp")),
        "-o",
        str(probe),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    (args.output / "compile.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0
    rng = np.random.default_rng(623016)
    gradients = np.eye(3)[None] + rng.normal(0, 0.08, (32, 3, 3))
    assert (np.linalg.det(gradients) > 0).all()
    mu, lame = snh_parameters(1.0, 0.3)
    rows = [np.r_[mu, lame, F.reshape(-1)] for F in gradients]
    stream = "\n".join(" ".join(f"{x:.17g}" for x in row) for row in rows) + "\n"
    result = subprocess.run([str(probe)], input=stream, capture_output=True, text=True, check=True)
    native = np.array([[float(x) for x in line.split()] for line in result.stdout.splitlines()])
    F = torch.tensor(gradients, dtype=torch.float64, device="cuda", requires_grad=True)
    energy = snh_energy(F, mu, lame)
    derivative = torch.autograd.grad(energy.sum(), F)[0]
    physical = evaluate(F.detach(), F.new_tensor(mu), F.new_tensor(lame))
    native_gradient = native[:, 1:].reshape(-1, 3, 3)
    assert torch.allclose(physical.first_piola_pa, derivative, rtol=1e-12, atol=2e-14)
    np.testing.assert_allclose(energy.detach().cpu().numpy(), native[:, 0], rtol=1e-12, atol=2e-14)
    np.testing.assert_allclose(physical.first_piola_pa.cpu().numpy(), native_gradient, rtol=1e-12, atol=2e-14)
    transform = torch.tensor(Rotation.from_rotvec([0.3, -0.2, 0.6]).as_matrix(), dtype=F.dtype, device=F.device)
    rotated = evaluate(transform @ F.detach(), F.new_tensor(mu), F.new_tensor(lame))
    torch.testing.assert_close(rotated.cauchy_pa, transform @ physical.cauchy_pa @ transform.T, rtol=1e-12, atol=2e-14)
    linear = []
    for model in ("snh", "hencky"):
        for poisson in (0.0, 0.3):
            young = 1e10
            true_mu = young / (2 * (1 + poisson))
            true_lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
            parameters = snh_parameters(young, poisson) if model == "snh" else (true_mu, true_lame)
            for amplitude in (1e-4, 1e-6):
                strain = torch.tensor(
                    [[amplitude, amplitude / 3, 0], [amplitude / 3, -amplitude / 4, 0], [0, 0, amplitude / 7]],
                    dtype=F.dtype,
                    device=F.device,
                )
                deformation = torch.eye(3, dtype=F.dtype, device=F.device) + strain
                actual = evaluate(deformation, F.new_tensor(parameters[0]), F.new_tensor(parameters[1]), model)
                target = 2 * true_mu * strain + true_lame * torch.trace(strain) * torch.eye(
                    3, dtype=F.dtype, device=F.device
                )
                error = (torch.linalg.norm(actual.cauchy_pa - target) / torch.linalg.norm(target)).item()
                assert error < 3 * amplitude
                linear.append(
                    {
                        "energy": model,
                        "poisson": poisson,
                        "strain_scale": amplitude,
                        "relative_small_strain_tensor_difference": error,
                    }
                )
    write_json(
        args.output / "result.json",
        {
            "kind": "unmodified-QIPC-energy-stress-control",
            "source_energy": file_id(args.source / "qipc/_src/native/solver/generated/stable_neo_hookean.h"),
            "probe_source": file_id(Path(__file__).with_name("energy_probe.cpp")),
            "compile_command": command,
            "maximum_native_vs_observer_piola_difference_at_E1": float(
                np.abs(native_gradient - physical.first_piola_pa.cpu().numpy()).max()
            ),
            "rotation_covariance": "passed",
            "small_strain": linear,
            "scope": "C++ original unprojected energy/gradient versus CUDA observer and autograd; no contact or dynamic reference qualification yet",
        },
    )


if __name__ == "__main__":
    main()
