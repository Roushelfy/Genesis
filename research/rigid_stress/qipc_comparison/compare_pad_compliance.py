"""Measure contact quadrature sensitivity separately from pressure-model error."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .compare_distributed_pad_transfer import field_norm, peaks


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    study = json.loads((args.study / "result.json").read_text())
    start = json.loads((args.study / "start.json").read_text())
    assert study["passed"]
    mesh = Path(start["mesh"]["path"])
    assert file_id(mesh) == start["mesh"]
    with np.load(mesh) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    volumes = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) / 6
    assert volumes.min() > 0
    rows = []
    for case in range(len(study["cases"])):
        for mode in ("coupled", "rigid", "diagonal"):
            pair = []
            for order in (0, 6):
                record = next(r for r in study["records"] if r["case"] == case and r["mode"] == mode
                              and r["order"] == order and r["search_contains_actual_elastic_compression"])
                path = Path(record["field"]["path"])
                assert file_id(path) == record["field"]
                with np.load(path) as saved:
                    pair.append({key: saved[key] for key in ("points_m", "area_m2", "pressure_pa", "stress_pa")})
            first, second = pair
            count = len(first["area_m2"])
            assert len(second["area_m2"]) % count == 0
            np.testing.assert_allclose(second["points_m"].reshape(count, -1, 3).mean(axis=1), first["points_m"], rtol=0, atol=1e-14)
            total_area = second["area_m2"].reshape(count, -1).sum(axis=1)
            np.testing.assert_allclose(total_area, first["area_m2"], rtol=1e-12, atol=0)
            averaged = (second["pressure_pa"] * second["area_m2"]).reshape(count, -1).sum(axis=1) / total_area
            pressure_error = np.sqrt(np.sum(total_area * (first["pressure_pa"] - averaged)**2)
                                     / np.sum(total_area * averaged**2))
            field_error = field_norm(first["stress_pa"] - second["stress_pa"], volumes) / field_norm(second["stress_pa"], volumes)
            p0, p6 = peaks(first["stress_pa"]), peaks(second["stress_pa"])
            rows.append({"case": case, "mode": mode, "parameters": study["cases"][case],
                "cell_average_pressure_relative_l2": pressure_error.item(),
                "complete_stress_volume_relative_l2": field_error.item(),
                "raw_mises_peak_relative_change": abs(p0["raw_vm_pa"] / p6["raw_vm_pa"] - 1),
                "raw_tensile_peak_relative_change": abs(p0["raw_tensile_pa"] / p6["raw_tensile_pa"] - 1),
                "quadrature_points_per_cell": len(second["area_m2"]) // count})
    write_json(args.output / "result.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "passed": True, "study": file_id(args.study / "result.json"), "rows": rows,
        "physical_reference_qualified": False, "whole_goal_complete": False,
        "scope": "Explicit centroid versus degree6 contact quadrature on the same coarse mesh and physical model. Pressure compares integrated face averages, not unmatched point locations. Complete stress uses exact physical-volume integration of its affine tensor. Changes are sensitivity measurements, not continuum bounds or new acceptance thresholds."})


if __name__ == "__main__":
    main()
