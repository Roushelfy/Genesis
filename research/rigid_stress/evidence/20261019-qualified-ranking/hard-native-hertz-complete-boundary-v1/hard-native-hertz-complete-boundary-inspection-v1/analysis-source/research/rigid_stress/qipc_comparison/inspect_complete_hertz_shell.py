"""Compare native complete-boundary fields with the retained independent integral."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import discrete_particular, matrix
from .p2_elastic import FreeElasticP2


def read_arrays(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--native", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--original", type=Path, required=True)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = read_arrays(args.reference / "comparison.npz")
    reference_rows = json.loads((args.reference / "result.json").read_text())["rows"]
    original = read_arrays(args.original)
    mesh = read_arrays(args.mesh)
    cpu = FreeElasticP2(mesh["vertices"], mesh["tetrahedra"], 1e10, 0.3, 2000)
    reference_indices = [i for i, row in enumerate(reference_rows) if row["rule"] == "focused32"]
    assert [reference_rows[i]["env"] for i in reference_indices] == list(range(4))
    rows, payloads = [], []
    for group in [0, 1]:
        path = args.native / f"group{group}/shell-q16/fields.npz"
        data = read_arrays(path)
        payloads.append(file_id(path))
        assert np.all(data["status"] == 0) and np.all(data["errno"] == 0)
        assert len(data["material_surface_nodes"]) == len(cpu.reference.mesh.boundary_facets())
        for key in ["centers_m", "frames", "axes_m", "force_n", "pressure_pa", "material_nodes_m"]:
            np.testing.assert_allclose(data[key], original[key], rtol=1e-12, atol=1e-16)
        permutation = cpu.reference.native_dof_permutation(data["material_nodes_m"]).reshape(-1, 3)[:, 0] // 3
        for env, index in enumerate(reference_indices):
            applied = np.empty_like(data["regular_force_n"][env])
            applied[permutation] = data["regular_force_n"][env]
            expected_load = reference["nodal_force_n"][index]
            expected = reference["corners_pa"][index]
            native = matrix(data["corner_tensor_pa"][env])
            regular, _, _, _ = cpu.solve(expected_load)
            axes, frame = data["axes_m"][env], data["frames"][env]
            local = data["probe_offsets"] * np.r_[axes, axes[0]]
            particular = discrete_particular(local, axes, data["pressure_pa"][env], data["force_n"][env], True)
            expected_probe = np.einsum("qc,qcij->qij", data["probe_barycentric"][env],
                                       regular[data["probe_elements"][env]])
            expected_probe += frame @ particular @ frame.T
            rows.append({
                "backend": "cpu" if group == 0 else "gpu", "environment": env,
                "boundary_load_relative": float(np.linalg.norm(applied - expected_load) / np.linalg.norm(expected_load)),
                "corner_field_relative": float(np.linalg.norm(native - expected) / np.linalg.norm(expected)),
                "probe_field_relative": float(np.linalg.norm(matrix(data["probe_tensor_pa"][env]) - expected_probe)
                                                / np.linalg.norm(expected_probe)),
                "native_wrench_error": float(data["wrench_error"][env]),
            })
    report = {"passed": False, "identity": runtime_identity(require_gpu_inventory=False),
              "native_fields": payloads, "reference_fields": file_id(args.reference / "comparison.npz"),
              "original_selected_surface_fields": file_id(args.original), "mesh": file_id(args.mesh),
              "rows": rows, "targets": {"boundary_load": 1e-4, "complete_fields": 1e-7, "wrench": 1e-4},
              "scope": "All material boundaries, independent higher-order point-traction integral and P2 elasticity, prescribed normal Hertz load. The retained original particulars carry the previously measured same-load numerical error. Curved-contact pressure, moving frictional shells and Franka throughput remain unqualified.",
              "whole_goal_complete": False}
    write_json(args.output / "result.json", report)
    assert all(row["boundary_load_relative"] < 1e-4 and row["corner_field_relative"] < 1e-7
               and row["probe_field_relative"] < 1e-7 and row["native_wrench_error"] < 1e-4 for row in rows), rows
    report["passed"] = True
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
