"""Recompute complete FEM residuals and raw native fields for a physical-load publication."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .skfem_reference import SkfemElasticReference


def source_closure(run):
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes(), run
    for line in (run / "source.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        candidate = run / "source" / name
        if not candidate.exists():
            candidate = run / "source" / Path(name).name
        assert hashlib.sha256(candidate.read_bytes()).hexdigest() == expected, (run, name)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.runtime_root
    closures = {name: source_closure(root / name) for name in args.run_names}
    tests = []
    for name in (
        "pressure-module-cpu-v1",
        "pressure-module-cuda-v1",
        "independent-transfer-cpu-v5",
        "independent-transfer-cuda-v5",
    ):
        suite = next(ET.parse(root / name / "report.xml").getroot().iter("testsuite"))
        assert suite.attrib["failures"] == suite.attrib["errors"] == "0"
        tests.append({"run": name, **suite.attrib})
        for line in (root / name / "source.sha256").read_text().splitlines():
            expected, source = line.split("  ", 1)
            if source.startswith("genesis/engine/solvers/rigid/stress/"):
                assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, (name, source)
    before, after = root / "load-u-smoke-v5/results", root / "load-u-after-v1/results"
    input_identity = {}
    for case in ("opposing", "bending"):
        first, second = before / case / "frozen_inputs.npz", after / case / "frozen_inputs.npz"
        assert first.read_bytes() == second.read_bytes()
        input_identity[case] = hashlib.sha256(first.read_bytes()).hexdigest()
    numeric = []
    for geometry in ("ellipsoid", "capsule", "u", "thin_neck", "cavity", "narrow_gap", "egg"):
        for level in range(1 if geometry == "egg" else 3):
            run = root / ("egg-pcg-diagnostic-v3" if geometry == "egg" else f"load-matrix-{geometry}-l{level}-v3")
            summary = json.loads((run / "results/summary.json").read_text())
            mesh_record = next(
                row
                for row in (root / f"matrix-mesh-{geometry}-v1/meshes").glob("*.json")
                if json.loads(row.read_text())["mesh_sha256"] == summary["cases"][0]["mesh_sha256"]
            )
            with np.load(mesh_record.with_suffix(".npz")) as mesh:
                vertices, tetrahedra = mesh["vertices"], mesh["tetrahedra"]
            reference = SkfemElasticReference(vertices, tetrahedra, 1e10, 0.3, 2000)
            edges = np.unique(
                np.sort(tetrahedra[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2), axis=1), axis=0
            )
            native_coordinates = np.concatenate((vertices, vertices[edges].mean(axis=1)))
            permutation = reference.native_dof_permutation(native_coordinates)
            for case in summary["cases"]:
                directory = run / "results" / case["case"]
                for model in case["models"]:
                    if not model["accepted"]:
                        continue
                    name = model["model"]
                    with np.load(directory / f"{name}_mapped_load.npz") as saved:
                        force = np.empty(reference.basis.N)
                        force[permutation] = saved["force_N"]
                    rhs = reference.balanced_rhs(force, np.array(case["omega_rad_s"]))
                    with np.load(directory / f"{name}_elastic_diagnostic.npz") as saved:
                        displacement = np.empty(reference.basis.N)
                        displacement[permutation] = saved["displacement_native_order"]
                        np.testing.assert_allclose(
                            saved["rhs_native_order_N"], rhs[permutation], rtol=1e-10, atol=1e-13
                        )
                        np.testing.assert_allclose(saved["expected_independent_rhs_N"], rhs, rtol=1e-10, atol=1e-13)
                    displacement = reference.align_rigid_gauge(displacement)
                    residual = np.linalg.norm(reference.stiffness @ displacement - rhs)
                    budget = max(1e-11, 1e-8 * np.linalg.norm(rhs))
                    assert residual <= budget, (geometry, level, case["case"], name, residual, budget)
                    tensor, vm = reference.corner_stress(displacement)
                    with np.load(directory / f"{name}_field.npz") as saved:
                        np.testing.assert_allclose(
                            saved["raw_native_tensor_Pa"],
                            tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
                            rtol=1e-4,
                            atol=1e-3,
                        )
                        np.testing.assert_allclose(saved["vm_Pa"], vm, rtol=1e-4, atol=1e-3)
                        np.testing.assert_allclose(saved["tensor_Pa"], tensor, rtol=1e-4, atol=1e-3)
                        np.testing.assert_allclose(saved["displacement"], displacement, rtol=1e-10, atol=1e-14)
                        raw_peak = float(saved["vm_Pa"].max())
                    np.testing.assert_allclose(raw_peak, model["peak_Pa"], rtol=1e-4, atol=1e-3)
                    energy = 0.5 * displacement @ (reference.stiffness @ displacement)
                    np.testing.assert_allclose(energy, model["energy_J"], rtol=1e-8, atol=1e-20)
                    wrench = reference.rigid_modes.T @ force
                    expected_force = np.sum([patch["force_N"] for patch in case["patches"]], axis=0)
                    expected_moment = np.sum(
                        [
                            np.cross(np.array(patch["point_m"]) - reference.com, patch["force_N"])
                            + patch["free_moment_Nm"]
                            for patch in case["patches"]
                        ],
                        axis=0,
                    )
                    numeric.append(
                        {
                            "geometry": geometry,
                            "level": level,
                            "case": case["case"],
                            "model": name,
                            "complete_residual_N": float(residual),
                            "complete_budget_fraction": float(residual / budget),
                            "global_raw_peak_Pa": raw_peak,
                            "energy_J": float(energy),
                            "aggregate_force_error_N": (wrench[:3] - expected_force).tolist(),
                            "aggregate_moment_error_Nm": (wrench[3:] - expected_moment).tolist(),
                            "sampled_coulomb_violation_Pa": model["maximum_coulomb_violation_Pa"],
                        }
                    )
    assert len(numeric) == 343
    solver_comparisons = []
    with np.load(root / "spherical-shell-pinned-v3/sweep/h0.08/field.npz") as direct:
        for name in ("shell-amg-h08-v2", "shell-symmetric-h08-v2"):
            with np.load(root / name / "solution/field.npz") as field:
                difference = np.linalg.norm(field["tensor_Pa"] - direct["tensor_Pa"]) / np.linalg.norm(
                    direct["tensor_Pa"]
                )
                assert difference <= 1e-8
                solver_comparisons.append({"run": name, "tensor_relative_difference_vs_direct": float(difference)})
    first = json.loads((root / "shell-amg-h08-v2/solution/result.json").read_text())
    second = json.loads((root / "shell-amg-h06-v1/solution/result.json").read_text())
    with (
        np.load(root / "shell-amg-h08-v2/solution/field.npz") as coarse,
        np.load(root / "shell-amg-h06-v1/solution/field.npz") as fine,
    ):
        np.testing.assert_array_equal(coarse["probes_m"], fine["probes_m"])
        shell_tail = {
            "tensor_relative_change": float(
                np.linalg.norm(coarse["probe_tensor_Pa"] - fine["probe_tensor_Pa"])
                / np.linalg.norm(fine["probe_tensor_Pa"])
            ),
            "vm_relative_change": float(
                np.linalg.norm(coarse["probe_vm_Pa"] - fine["probe_vm_Pa"]) / np.linalg.norm(fine["probe_vm_Pa"])
            ),
            "raw_peak_relative_change": abs(first["peak_Pa"] / second["peak_Pa"] - 1),
            "energy_relative_change": abs(first["energy_J"] / second["energy_J"] - 1),
        }
    report = {
        "scope": "Physical-load progress; complete convex/nonconvex physical and deformable-grasp acceptance remains unfinished",
        "source_closures": closures,
        "tests": tests,
        "matched_u_inputs": input_identity,
        "recomputed_native_fields": numeric,
        "maximum_complete_budget_fraction": max(row["complete_budget_fraction"] for row in numeric),
        "reference_solver_comparisons": solver_comparisons,
        "shell_h08_to_h06_tail": shell_tail,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "source_closures": len(closures),
                "fields": len(numeric),
                "maximum_complete_budget_fraction": report["maximum_complete_budget_fraction"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
