"""Verify retained physical-step global peak states and describe their discrete material locations without claiming continuum extrema."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import voigt_tensor


def invariants(stress):
    mean = np.trace(stress, axis1=-2, axis2=-1) / 3
    deviator = stress - mean[..., None, None] * np.eye(3)
    return np.sqrt(1.5 * np.sum(deviator**2, axis=(-2, -1))), np.linalg.eigvalsh(stress)[..., -1]


def audit(folder):
    receipt = json.loads((folder / "result.json").read_text())
    assert receipt["completed_requested_history"], "A peak of a partial history is not a complete-history maximum"
    journal = folder / "physical-step-peaks.jsonl"
    if not journal.exists():
        journal = folder / "steps.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    times = np.array([row["time_s"] for row in rows])
    np.testing.assert_allclose(times, np.arange(1, len(rows) + 1) * (times[1] - times[0]), atol=2e-12, rtol=0)
    assert max(np.diff(times)) <= 0.001 + 1e-12, "Use actual physical-step peak coverage, not last-substep-only peaks"
    with np.load(folder / "rest.npz") as saved:
        rest = {key: saved[key] for key in saved.files}
    results = []
    for kind, filename in (("VM", "raw-vm-peak-state.npz"), ("maximum-principal", "raw-principal-peak-state.npz")):
        path = folder / filename
        with np.load(path) as saved:
            frame = {key: saved[key] for key in saved.files}
        native = "stress_voigt_link_pa" in frame
        shell = "cauchy_world_pa" in frame
        stress = (
            voigt_tensor(frame["stress_voigt_link_pa"])
            if native
            else frame["cauchy_world_pa" if shell else "cauchy_pa"]
        )
        assert np.isfinite(stress).all()
        vm, principal = invariants(stress)
        if "vm_pa" in frame:
            np.testing.assert_allclose(vm, frame["vm_pa"], rtol=2e-12, atol=1e-6)
        value = vm if kind == "VM" else principal
        index = np.unravel_index(np.argmax(value), value.shape)
        if native:
            point = rest["vertices"][rest["tetrahedra"][index[0], index[1]]]
            location = "P2 tetrahedron corner in authored link/material coordinates"
        elif shell:
            point = rest["vertices_m"][rest["faces"][index[0]]].mean(axis=0)
            location = "Authored mid-surface face centroid; through-wall fiber index is retained separately"
        else:
            point = rest["authored_vertices_m"][rest["tetrahedra"][index[0]]].mean(axis=0)
            location = "Authored P1 tetrahedron centroid; the tensor is constant throughout this cell"
        key = (
            "raw_maximum_principal_pa"
            if kind != "VM"
            else ("raw_vm_peak_last_physical_substep_pa" if native else "raw_vm_peak_pa")
        )
        maximum = max(row[key] for row in rows)
        np.testing.assert_allclose(value[index], maximum, rtol=2e-12, atol=1e-6)
        time = float(frame["time_s"])
        at_time = next(row for row in rows if abs(row["time_s"] - time) < 1e-12)
        np.testing.assert_allclose(at_time[key], maximum, rtol=2e-12, atol=1e-6)
        tie = np.abs(value - maximum) <= 64 * np.finfo(float).eps * max(abs(maximum), 1)
        results.append(
            {
                "measure": kind,
                "value_pa": float(value[index]),
                "time_s": time,
                "array_index": [int(x) for x in index],
                "material_representative_point_m": point.tolist(),
                "location_contract": location,
                "same_field_entries_within_64epsilon_peak_tolerance": int(tie.sum()),
                "full_journal_and_saved_peak_agree": True,
                "state": file_id(path),
            }
        )
    return {
        "folder": str(folder),
        "receipt": file_id(folder / "result.json"),
        "journal": file_id(journal),
        "physical_observations": len(rows),
        "last_time_s": float(times[-1]),
        "peaks": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Independent analytic invariants, including a signed compressive principal value.
    tensor = np.array([np.diag([3.0, 3, 3]), np.diag([-3.0, -2, -1]), [[0, 2, 0], [2, 0, 0], [0, 0, 0]]])
    vm, principal = invariants(tensor)
    np.testing.assert_allclose(vm, [0, np.sqrt(3), np.sqrt(12)], rtol=2e-15)
    np.testing.assert_array_equal(principal, [3, -1, 2])
    write_json(
        args.output / "result.json",
        {
            "kind": "physical-step-retained-global-peak-state-audit",
            "identity": runtime_identity(),
            "analytic_hydrostatic_compressive_and_shear_controls_passed": True,
            "histories": [audit(folder) for folder in args.histories],
            "reference_qualified": False,
            "scope": "All recorded physical-step maxima are checked against retained complete fields; no display-grid maxima or phase cuts substitute for them. QIPC/native use full3D physical Cauchy, Shell its declared tangential/fiber field with transverse omissions. Their raw invariants are descriptive, not a common-observable accuracy ranking. Representative cell/corner/face locations and ties are explicit; no continuum extremum, unresolved-peak convergence or physical damage claim is made. Finite complete coverage alone does not establish numerical/contact/reference admission.",
        },
    )


if __name__ == "__main__":
    main()
