"""Measure separate body/finger discrete force densities from unchanged independently audited contact fields."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, rows = {}, []
    for comparison in args.run:
        closed_source(comparison)
        assert int((comparison / "returncode.txt").read_text()) == 0
        path = comparison / "results/summary.json"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = json.loads(path.read_text())
        for index, original in enumerate(report["rows"]):
            field = comparison / f"results/case{index:02d}.npz"
            geometry = Path(original["run"]) / "results/contact_geometry.npz"
            for path in [field, geometry]:
                inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(geometry) as saved:
                groups = saved["groups"].copy()
            with np.load(field) as saved:
                normal, shear, area = [
                    saved[key].copy()
                    for key in ["normal_surface_force_N", "shear_surface_force_N", "surface_vertex_lumped_area_m2"]
                ]
            assert len(groups) == len(area) and np.all(np.isfinite(area)) and np.all(area > 0)
            for group in [0, 1, 2]:
                selected = groups == group
                n, s = np.linalg.norm(normal[selected], axis=1), np.linalg.norm(shear[selected], axis=1)
                a = area[selected]
                active = n > 1e-8
                norm = float(np.sqrt(np.sum(n**2 / a)))
                if group == 0:
                    np.testing.assert_allclose(
                        norm, original["lumped_body_normal_force_density_L2_N_per_m"], rtol=1e-12
                    )
                rows.append(
                    {
                        "comparison": str(comparison),
                        "producer": original["run"],
                        "body_contact_refinement": original["contact_refinement"],
                        "finger_contact_refinement": original.get("finger_refinement", 0),
                        "group": group,
                        "active_samples": int(active.sum()),
                        "active_lumped_area_m2": float(a[active].sum()),
                        "unsigned_normal_sample_force_N": float(n.sum()),
                        "unsigned_shear_sample_force_N": float(s.sum()),
                        "lumped_normal_force_density_L2_N_per_m": norm,
                        "lumped_shear_force_density_L2_N_per_m": float(np.sqrt(np.sum(s**2 / a))),
                        "maximum_lumped_normal_force_density_Pa": float((n / a).max()),
                    }
                )
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    with (args.output / "group_densities.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "These are discrete sample-force/lumped-area diagnostics, not continuous pressures, continuum contact areas or physical islands. Threshold1e-8N selects active samples for area/count reporting only; all forces contribute to density norms. No field, force law, raw global maximum or acceptance observable is changed, and no physical convergence follows from a fixed-bulk field tail.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
