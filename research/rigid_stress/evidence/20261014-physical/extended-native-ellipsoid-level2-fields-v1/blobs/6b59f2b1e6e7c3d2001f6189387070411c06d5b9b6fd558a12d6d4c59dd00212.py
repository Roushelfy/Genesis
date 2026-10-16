"""Separate body and pad spatial changes on the four completed U references."""

import argparse
import json
from pathlib import Path

import numpy as np

from .sample_load_fields import common_physical_probes, locate_probes, sample_field
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source, relative_change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = (
        "grasp-u-spatial-y-l1-v1",
        "grasp-u-pad-axis-l1-p7-v1",
        "grasp-u-body-axis-l2-p5-v1",
        "grasp-u-spatial-y-l2-v1",
    )
    probes = common_physical_probes("u")
    rows, fields, configurations = [], [], []
    for name in names:
        run = args.runtime_root / name
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "solution/configuration.json").read_text())
        result = json.loads((run / "solution/result.json").read_text())
        configurations.append(config)
        assert result["complete_constrained_residual_N"] <= 1e-9
        with np.load(config["mesh"]) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        with np.load(run / "solution/fields.npz") as saved:
            vector = np.empty(reference.basis.N)
            vector[saved["body_component_permutation"]] = saved["displacement"][: reference.basis.N]
            tensor = saved["body_tensor_Pa"]
        elastic = reference.align_rigid_gauge(vector)
        elements, valid, barycentric = locate_probes(reference.mesh, probes)
        stress, vm, displacement = sample_field(
            reference.basis, elements, valid, barycentric, probes, {"tensor_Pa": tensor, "displacement": elastic}
        )
        fields.append({"tensor": stress, "vm": vm, "displacement": displacement, "valid": valid})
        rows.append(
            {
                "run": name,
                "body_tetrahedra": reference.mesh.nelements,
                "pad_resolution": config["pad_resolution"],
                "raw_global_peak_Pa": result["raw_body_peak_Pa"],
                "energy_J": float(0.5 * elastic @ (reference.stiffness @ elastic)),
                "outside_probe_indices": np.flatnonzero(~valid).tolist(),
            }
        )
    for config in configurations[1:]:
        for key, value in configurations[0].items():
            if key not in ("mesh", "pad_resolution", "output"):
                assert config[key] == value, key
    comparisons = []
    for first, second, axis in (
        (0, 1, "pad_fixed_coarse_body"),
        (2, 3, "pad_fixed_finer_body"),
        (0, 2, "body_fixed_pad5"),
        (1, 3, "body_fixed_pad7"),
    ):
        shared = fields[first]["valid"] & fields[second]["valid"]
        row = {"axis": axis, "first": names[first], "second": names[second], "common_probes": int(shared.sum())}
        for observable in ("tensor", "vm", "displacement"):
            relative, absolute = relative_change(fields[first][observable][shared], fields[second][observable][shared])
            row[observable] = {"relative_change": relative, "absolute_change": absolute}
        for observable in ("raw_global_peak_Pa", "energy_J"):
            row[observable] = abs(rows[first][observable] / rows[second][observable] - 1)
        comparisons.append(row)
    report = {
        "scope": __doc__,
        "rows": rows,
        "comparisons": comparisons,
        "goal_completion": False,
        "limitations": "Two levels per separated axis diagnose the dominant spatial change; they do not establish a tail. Body mesh changes also approximate the continuum geometry, while pad refinement approximates its curved surface and contact integration. Linear body/part assumptions remain explicit and nonlinear references are separate physical-model checks. Whole raw peaks, original residuals and all fixed probes remain retained.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
