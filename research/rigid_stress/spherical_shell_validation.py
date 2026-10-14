"""Opt-in independent spherical-shell convergence against the three-dimensional Lame solution."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from skfem import FacetBasis, LinearForm, asm
from skfem.helpers import dot

from .skfem_reference import SkfemElasticReference
from .validation_meshes import generate_mesh


@LinearForm
def pressure_load(v, w):
    return -w.pressure * dot(w.n, v)


def analytic_shell(points, inner, outer, pressure, young, poisson):
    radius = np.linalg.norm(points, axis=1)
    direction = points / radius[:, None]
    a = pressure * inner**3 / (outer**3 - inner**3)
    b = pressure * inner**3 * outer**3 / (outer**3 - inner**3)
    tangential = a + b / (2 * radius**3)
    tensor = tangential[:, None, None] * np.eye(3) - 1.5 * b / radius[:, None, None] ** 3 * (
        direction[:, :, None] * direction[:, None, :]
    )
    displacement = direction * (((1 - 2 * poisson) * a * radius + (1 + poisson) * b / (2 * radius**2)) / young)[:, None]
    return displacement, tensor, 1.5 * abs(b) / radius**3


def validate_shell(size, layers, output, probes):
    started = perf_counter()
    mesh = output / "mesh.npz"
    metadata = generate_mesh("cavity", size, layers, mesh)
    with np.load(mesh) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    reference = SkfemElasticReference(vertices, tets, 1e10, 0.3, 2000)
    boundary = reference.mesh.boundary_facets()
    boundary_radius = np.linalg.norm(reference.mesh.p[:, reference.mesh.facets[:, boundary]], axis=0)
    inner_faces = boundary[np.max(boundary_radius, axis=0) < 0.010]
    assert len(inner_faces) > 0
    facet_basis = FacetBasis(reference.mesh, reference.element, facets=inner_faces, intorder=4)
    pressure = 1e4
    force = asm(pressure_load, facet_basis, pressure=pressure)
    solved = reference.solve(force)
    actual_tensor, actual_vm = reference.stress_at_points(solved.displacement, probes)
    expected_displacement, expected_tensor, expected_vm = analytic_shell(probes, 0.008, 0.012, pressure, 1e10, 0.3)
    probe_displacement = (reference.basis.probes(probes.T) @ solved.displacement).reshape(3, -1).T
    inner_displacement = analytic_shell(np.array([[0.008, 0, 0]]), 0.008, 0.012, pressure, 1e10, 0.3)[0][0, 0]
    exact_energy = 0.5 * (4 * np.pi * 0.008**2) * pressure * inner_displacement
    exact_peak = analytic_shell(np.array([[0.008, 0, 0]]), 0.008, 0.012, pressure, 1e10, 0.3)[2][0]
    tensor_error = np.linalg.norm(actual_tensor - expected_tensor) / np.linalg.norm(expected_tensor)
    report = {
        "mesh": metadata,
        "young_Pa": 1e10,
        "poisson": 0.3,
        "density_kg_m3": 2000,
        "inner_pressure_Pa": pressure,
        "outer_pressure_Pa": 0,
        "inner_radius_m": 0.008,
        "outer_radius_m": 0.012,
        "reference": "full3D independent scikit-fem P2, closed pressure shell, consistent-mass aligned free-body solution",
        "linear_gauge": reference.gauge,
        "factor_seconds": reference.factor_seconds,
        "factor_nonzeros": reference.factor_nonzeros,
        "peak_Pa": float(solved.peak_pa),
        "exact_peak_Pa": float(exact_peak),
        "peak_relative_error": float(abs(solved.peak_pa / exact_peak - 1)),
        "peak_position_m": solved.peak_position_m.tolist(),
        "energy_J": float(solved.energy_j),
        "exact_energy_J": float(exact_energy),
        "energy_relative_error": float(abs(solved.energy_j / exact_energy - 1)),
        "common_probe_tensor_relative_error": float(tensor_error),
        "common_probe_vm_relative_error": float(np.linalg.norm(actual_vm - expected_vm) / np.linalg.norm(expected_vm)),
        "common_probe_displacement_relative_error": float(
            np.linalg.norm(probe_displacement - expected_displacement) / np.linalg.norm(expected_displacement)
        ),
        "complete_residual_N": float(solved.residual_n),
        "complete_residual_budget_fraction": float(solved.residual_budget_fraction),
        "total_mass_kg": float(reference.total_mass),
        "elapsed_seconds": perf_counter() - started,
    }
    assert solved.residual_budget_fraction <= 1
    np.savez_compressed(
        output / "field.npz",
        displacement=solved.displacement,
        tensor_Pa=solved.tensor_pa,
        von_mises_Pa=solved.von_mises_pa,
        probes_m=probes,
        probe_tensor_Pa=actual_tensor,
        probe_vm_Pa=actual_vm,
        exact_probe_tensor_Pa=expected_tensor,
    )
    report["field_sha256"] = hashlib.sha256((output / "field.npz").read_bytes()).hexdigest()
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=float, nargs="+", default=[0.6, 0.4, 0.28, 0.2])
    parser.add_argument("--shell-layers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    directions = np.random.default_rng(2026101402).normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    directions = np.concatenate((directions, np.eye(3), -np.eye(3)))
    probes = 0.0096 * directions
    results = []
    for size in args.sizes:
        output = args.output / f"h{size:g}"
        output.mkdir()
        results.append(validate_shell(size, args.shell_layers, output, probes))
        (args.output / "summary.json").write_text(json.dumps({"results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
