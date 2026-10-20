"""Independently solve complete fields from the actual native pressure arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .compare_distributed_pad_transfer import field_norm, peaks
from .p2_elastic import FreeElasticP2


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inputs", type=Path, required=True)
    cli.add_argument("--gpu", type=Path, required=True)
    cli.add_argument("--cpu", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = json.loads((args.inputs / "result.json").read_text())
    start = json.loads((args.inputs / "start.json").read_text())
    assert prepared["passed"]
    mesh = start["inputs"][0]
    assert file_id(Path(mesh["path"])) == mesh
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [
                file_id(args.inputs / "result.json"),
                file_id(args.gpu / "result.json"),
                file_id(args.cpu / "result.json"),
            ],
            "mesh": mesh,
            "targets": {
                "complete_field_relative": 1e-6,
                "peak_relative": 1e-6,
                "elastic_equation": 1e-7,
                "balance": 1e-8,
                "backend_force_relative": 1e-10,
            },
        },
    )
    with np.load(mesh["path"]) as saved:
        vertices, tets, faces = (saved[k] for k in ("vertices", "tetrahedra", "mid_faces"))
    volumes = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) / 6
    model = FreeElasticP2(
        vertices, tets, 1e10, 0.3, 2000, column_faces=faces, through_wall_ordering=True, cvxopt_factor=True
    )
    gpu, cpu = (json.loads((directory / "result.json").read_text()) for directory in (args.gpu, args.cpu))
    assert gpu["passed"] and cpu["passed"]
    rows, backend_errors = [], []
    for order_index, record in enumerate(prepared["records"]):
        path = Path(record["inputs"]["path"])
        assert file_id(path) == record["inputs"]
        with np.load(path) as saved:
            nodes, sign, interpolation, loads = (saved[k] for k in ("nodes", "node_sign", "interpolation", "loads_n"))
            np.testing.assert_array_equal(saved["p2_coordinates_m"], model.vertices)
        normalized = []
        for name, result in (("gpu", gpu), ("cpu", cpu)):
            row = result["rows"][order_index]
            assert row["inputs"] == record["inputs"] and row["order"] == record["order"]
            path = Path(row["pressure"]["path"])
            assert file_id(path) == row["pressure"]
            with np.load(path) as saved:
                pressure_force = saved["normalized_force"]
                assert np.all(saved["status"] == 1)
            normalized.append(pressure_force)
            for case, reference in enumerate(record["references"]):
                force = np.zeros_like(model.vertices)
                force[nodes, 2] = sign * (interpolation.T @ (pressure_force[:, case] * loads[case]))
                stress, displacement, correction, residual = model.solve(force)
                equation_error = np.linalg.norm(residual) / np.linalg.norm(force)
                balance_error = np.linalg.norm(correction) / np.linalg.norm(force)
                assert equation_error <= 1e-7 and balance_error <= 1e-8
                assert file_id(Path(reference["path"])) == reference
                with np.load(reference["path"]) as original:
                    np.testing.assert_array_equal(original["vertices_m"], model.vertices)
                    field_error = field_norm(stress - original["stress_pa"], volumes) / field_norm(
                        original["stress_pa"], volumes
                    )
                    force_error = np.linalg.norm(force - original["forces_n"]) / np.linalg.norm(original["forces_n"])
                    actual_peaks, expected_peaks = peaks(stress), peaks(original["stress_pa"])
                    peak_errors = {
                        k: abs(actual_peaks[k] / expected_peaks[k] - 1) for k in ("raw_vm_pa", "raw_tensile_pa")
                    }
                assert max(field_error, force_error, *peak_errors.values()) <= 1e-6
                path = args.output / f"{name}-q{record['order']}-case{case}.npz"
                np.savez_compressed(
                    path,
                    vertices_m=model.vertices,
                    tetrahedra=tets,
                    stress_pa=stress,
                    forces_n=force,
                    displacement_at_10gpa_m=displacement,
                )
                rows.append(
                    {
                        "backend": name,
                        "order": record["order"],
                        "case": case,
                        "complete_stress_relative_error": field_error.item(),
                        "nodal_force_relative_error": force_error.item(),
                        "peak_relative_errors": peak_errors,
                        "equation_relative_error": equation_error.item(),
                        "balance_relative_error": balance_error.item(),
                        "field": file_id(path),
                        "reference": reference,
                        "native_pressure": row["pressure"],
                    }
                )
                write_json(args.output / "progress.json", {"rows": rows})
        backend_error = np.linalg.norm(normalized[0] - normalized[1]) / np.linalg.norm(normalized[1])
        assert backend_error <= 1e-10
        backend_errors.append({"order": record["order"], "pressure_force_relative": backend_error.item()})
    assert len(rows) == 2 * sum(len(record["references"]) for record in prepared["records"])
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "backend_errors": backend_errors,
            "production_automatic_model_admitted": False,
            "whole_goal_complete": False,
            "scope": "Actual CPU/GPU native pressure outputs drive independent complete P2 elastic fields. Preserved physical-force units, free-link mass projection, complete-volume norms and raw peaks are checked. This validates the numerical pressure stage and its physical response on the declared fixed normal pad cases. Collider geometry, native operator preparation, friction and production lifecycle integration remain separate work.",
        },
    )


if __name__ == "__main__":
    main()
