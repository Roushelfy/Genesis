"""Diagnose exact-point stress traces and additional common interior probes on unchanged thickness fields."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from skfem import MeshTet

from .sample_load_fields import common_physical_probes, locate_probes
from .summarize_fine_references import CASES, closed_source, relative_change
from .validation_meshes import transform_shell


def interpolate(mesh, tensor, points):
    elements, selected, barycentric = locate_probes(mesh, points)
    assert selected.all()
    return np.einsum("nc,ncij->nij", barycentric, tensor[elements]), barycentric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-audit", type=Path, required=True)
    parser.add_argument("--run", type=Path, nargs=3, required=True)
    parser.add_argument("--mesh", type=Path, nargs=3, required=True)
    parser.add_argument("--original-probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.axis_audit, *args.run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    audit_path = args.axis_audit / "results/summary.json"
    audit = json.loads(audit_path.read_text())
    assert audit["fields_verified"] == 15
    hashes = {str(audit_path): hashlib.sha256(audit_path.read_bytes()).hexdigest()}
    original = common_physical_probes("egg")
    with np.load(args.original_probes) as saved:
        np.testing.assert_array_equal(original, saved["points_m"])
    hashes[str(args.original_probes)] = hashlib.sha256(args.original_probes.read_bytes()).hexdigest()
    directions = np.random.default_rng(2026101505).normal(size=(64, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    fractions = (np.polynomial.legendre.leggauss(4)[0] + 1) / 2
    off_points = transform_shell("egg", ((1 - 0.08 * fractions[:, None, None]) * directions[None]).reshape(-1, 3))
    perturbations = np.random.default_rng(2026101506).normal(size=(6, 3))
    perturbations /= np.linalg.norm(perturbations, axis=1)[:, None]
    perturbations = np.concatenate((perturbations, -perturbations))
    steps = [1e-8, 1e-9, 1e-10]
    args.output.mkdir(parents=True, exist_ok=False)
    rows, fields = [], {}
    for layers, run, path in zip([2, 3, 4], args.run, args.mesh, strict=True):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
        _elements, selected, original_bary = locate_probes(mesh, original)
        assert selected.all()
        _off_elements, off_selected, off_bary = locate_probes(mesh, off_points)
        assert off_selected.all()
        for case in CASES:
            field_path = run / f"results/{case}/reference_field.npz"
            expected = next(row for row in audit["rows"] if row["case"] == case and row["layers"] == layers)
            digest = hashlib.sha256(field_path.read_bytes()).hexdigest()
            assert digest == expected["field_sha256"]
            hashes[str(field_path)] = digest
            with np.load(field_path) as saved:
                tensor = saved["tensor_Pa"]
            central, _bary = interpolate(mesh, tensor, original)
            off, _off_bary = interpolate(mesh, tensor, off_points)
            fields[layers, case] = (central, off)
            neighborhoods = []
            for step in steps:
                points = (original[:, None] + step * perturbations[None]).reshape(-1, 3)
                traces, _trace_bary = interpolate(mesh, tensor, points)
                traces = traces.reshape(len(original), len(perturbations), 3, 3)
                differences = traces[:, :, None] - traces[:, None]
                diameter = np.sqrt(np.sum(differences**2, axis=(-2, -1))).max(axis=(1, 2))
                neighborhoods.append(
                    {
                        "step_m": step,
                        "per_probe_tensor_diameter_Pa": diameter.tolist(),
                        "tensor_diameter_L2_over_central_field_L2": float(
                            np.linalg.norm(diameter) / np.linalg.norm(central)
                        ),
                    }
                )
            row = {
                "case": case,
                "layers": layers,
                "original_probe_count": len(original),
                "original_minimum_barycentric_coordinate": float(original_bary.min()),
                "original_probe_indices_with_barycentric_margin_below_1e8": np.flatnonzero(
                    original_bary.min(axis=1) < 1e-8
                ).tolist(),
                "off_probe_count": len(off_points),
                "off_minimum_barycentric_coordinate": float(off_bary.min()),
                "neighborhoods": neighborhoods,
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"layers{layers}-{case}.npz",
                original_points_m=original,
                original_barycentric=original_bary,
                central_tensor_Pa=central,
                off_points_m=off_points,
                off_barycentric=off_bary,
                off_tensor_Pa=off,
                perturbation_directions=perturbations,
            )
    comparisons = []
    for case in CASES:
        for first, second in [(2, 3), (3, 4)]:
            central_change, _absolute = relative_change(fields[first, case][0], fields[second, case][0])
            off_change, off_absolute = relative_change(fields[first, case][1], fields[second, case][1])
            original_metric = next(
                row["tensor_Pa"]["relative_change"]
                for row in audit["adjacent_changes"]
                if row["case"] == case and row["from_layers"] == first and row["to_layers"] == second
            )
            np.testing.assert_allclose(central_change, original_metric, rtol=1e-12, atol=1e-14)
            comparisons.append(
                {
                    "case": case,
                    "from_layers": first,
                    "to_layers": second,
                    "original_tensor_relative_change": central_change,
                    "additional_off_probe_tensor_relative_change": off_change,
                    "additional_off_probe_absolute_change_Pa": off_absolute,
                }
            )
    assert hashes == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in hashes}
    result = {
        "scope": __doc__,
        "rows": rows,
        "comparisons": comparisons,
        "input_sha256": hashes,
        "diagnostic_definition": {
            "original_probes_unchanged": True,
            "perturbation_radii_m": steps,
            "signed_directions_per_probe": len(perturbations),
            "additional_directions_seed": 2026101505,
            "additional_depth_rule": "Four Gauss-Legendre interior depths on the declared analytic normal-offset egg",
            "additional_common_probe_count": len(off_points),
        },
        "physical_accuracy_accepted": False,
        "goal_completion": False,
        "limitations": "Original exact-point metrics and every raw global maximum remain unchanged. The neighborhood samples expose possible discontinuous element stress traces without averaging or selecting a preferred trace. Additional predeclared interior probes diagnose depth/interface sensitivity and do not replace the original acceptance observable or establish a global volume norm. Geometry, loads and all solved fields are immutable; no new equilibrium solve or physical convergence is claimed. Original geometry/contact/model/feedback/full-application and verified-cost gates remain.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fields": len(rows), "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
