"""Independently reconstruct raw corner peaks and work identities before publishing FEM references."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .summarize_fine_references import CASES, GEOMETRIES, closed_source

SIZES = {
    "ellipsoid": ("0.00175", "0.00125", "0.000875"),
    "capsule": ("0.00175", "0.00125", "0.000875"),
    "thin_neck": ("0.00175", "0.00125", "0.000875"),
    "u": ("0.00125", "0.000875", "0.000625"),
    "narrow_gap": ("0.00125", "0.000875", "0.000625"),
    "cavity": ("0.11", "0.08", "0.06"),
    "egg": ("0.06", "0.04", "0.03"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for level, size in enumerate(SIZES[args.geometry]):
        run = args.runtime_root / f"fine-ref-{args.geometry}-l{level}-v1"
        for directory in (run, Path(str(run) + "-probes"), Path(str(run) + "-launcher")):
            closed_source(directory)
            assert int((directory / "returncode.txt").read_text()) == 0
        mesh_path = args.runtime_root / f"fine-matrix-mesh-{args.geometry}-v1/meshes/h{size}.npz"
        mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        with np.load(mesh_path) as saved:
            vertices, tetrahedra = saved["vertices"], saved["tetrahedra"]
        for case in CASES:
            snapshot = json.loads((run / f"results/{case}/snapshot.json").read_text())
            assert snapshot["mesh_sha256"] == mesh_hash
            assert snapshot["reference_complete_budget_fraction"] <= 1
            field_path = run / f"results/{case}/reference_field.npz"
            with np.load(field_path) as field:
                tensor, saved_vm = field["tensor_Pa"], field["vm_Pa"]
                displacement, rhs = field["displacement"], field["rhs_N"]
            assert tensor.shape == (len(tetrahedra), 4, 3, 3)
            assert all(np.isfinite(value).all() for value in (tensor, saved_vm, displacement, rhs))
            deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
            vm = np.sqrt(1.5 * np.sum(deviator * deviator, axis=(-2, -1)))
            np.testing.assert_allclose(saved_vm, vm, rtol=1e-11, atol=1e-8)
            element, corner = np.unravel_index(np.argmax(vm), vm.shape)
            peak = float(vm[element, corner])
            position = vertices[tetrahedra[element, corner]]
            np.testing.assert_allclose(peak, snapshot["reference_peak_Pa"], rtol=1e-12, atol=1e-8)
            np.testing.assert_array_equal(position, snapshot["reference_peak_position_m"])
            energy = float(0.5 * displacement @ rhs)
            np.testing.assert_allclose(energy, snapshot["reference_energy_J"], rtol=1e-10, atol=1e-24)
            rows.append(
                {
                    "geometry": args.geometry,
                    "case": case,
                    "level": level,
                    "tetrahedra": len(tetrahedra),
                    "raw_reconstructed_peak_Pa": peak,
                    "raw_peak_position_m": position.tolist(),
                    "reconstructed_work_J": energy,
                    "complete_budget_fraction_recorded_by_independent_FEM": snapshot[
                        "reference_complete_budget_fraction"
                    ],
                    "mesh_sha256": mesh_hash,
                    "field_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
                }
            )
    result = {
        "scope": "Source closure, full corner-stress/peak and work identity verification of independently assembled FEM fields",
        "fields": rows,
        "verified_fields": len(rows),
        "limitations": "This postprocessing audit does not reassemble stiffness or recompute the equilibrium residual; those are computed by the independent FEM driver and retained with its source. It does not certify continuum or coupled-contact accuracy.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"geometry": args.geometry, "verified_fields": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
