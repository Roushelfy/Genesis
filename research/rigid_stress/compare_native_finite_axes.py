"""Compare audited native distance/integration controls and model bias with exact volume tensor norms."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .summarize_fine_references import CASES, closed_source

COMPONENT_WEIGHTS = np.array([1.0, 1, 1, 2, 2, 2])
MODELS = ["pressure", "componentwise", "pair_union", "body_source_prior"]


def linear_tensor_squared_integral(corners, volumes):
    """Integrate squared symmetric-tensor Frobenius norm exactly on affine tetrahedra."""
    assert corners.shape == (len(volumes), 4, 6)
    diagonal = np.sum(corners * corners * COMPONENT_WEIGHTS, axis=(1, 2))
    summed = corners.sum(axis=1)
    cross = np.sum(summed * summed * COMPONENT_WEIGHTS, axis=1)
    return float(volumes @ (diagonal + cross) / 20)


def components(tensor):
    return tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]


def manufactured_norm_control():
    rng = np.random.default_rng(20261016)
    corners = rng.normal(size=(7, 4, 6))
    volumes = rng.uniform(0.01, 2, size=7)
    small = (5 - np.sqrt(5)) / 20
    barycentric = np.full((4, 4), small)
    np.fill_diagonal(barycentric, (5 + 3 * np.sqrt(5)) / 20)
    evaluated = np.einsum("qc,ecp->eqp", barycentric, corners)
    quadrature = float(volumes @ np.sum(evaluated * evaluated * COMPONENT_WEIGHTS, axis=(1, 2)) / 4)
    np.testing.assert_allclose(linear_tensor_squared_integral(corners, volumes), quadrature, rtol=1e-14, atol=0)
    constant = np.broadcast_to(corners[:, :1], corners.shape)
    exact = float(volumes @ np.sum(corners[:, 0] ** 2 * COMPONENT_WEIGHTS, axis=1))
    np.testing.assert_allclose(linear_tensor_squared_integral(constant, volumes), exact, rtol=1e-14, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manufactured_norm_control()
    p = args.runtime_root
    inputs, comparisons, biases, qualified_runs = {}, [], [], []

    def retain(path, expected=None):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert expected is None or digest == expected, path
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
        return digest

    def read_json(path):
        retain(path)
        return json.loads(path.read_text())

    def closed(run):
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0

    def load_run(name, expected_ports, expected_quadrature):
        run = p / name
        closed(run)
        auditor = p / name.replace("models", "independent-audit")
        closed(auditor)
        audit = read_json(auditor / "results/summary.json")
        assert audit["accepted_fields_freshly_audited"] == 19
        assert audit["unsupported_or_rejected_inputs_retained"] == 1 and len(audit["rows"]) == 20
        for row in audit["rows"]:
            if row["accepted"]:
                assert row["all_original_corner_and_independent_quadrature_gates_passed"]
                assert row["original_budget_fraction"] <= 1
                assert row["all_quadrature_tensor_components_checked"] == 99 * audit["tetrahedra"]
        summary_path = run / "results/summary.json"
        retain(summary_path, audit["input_sha256"][str(summary_path)])
        summary = json.loads(summary_path.read_text())
        assert [row["case"] for row in summary["cases"]] == list(CASES)
        for case in summary["cases"]:
            assert [row["model"] for row in case["models"]] == MODELS
            assert case["patch_radius_m"] == 0.003 and case["reference_quadrature_level"] == 3
            assert all(row["ports"] == expected_ports for row in case["models"])
            assert all(row["native_quadrature"] == expected_quadrature for row in case["models"])
        qualified_runs.append(
            {
                "run": str(run),
                "audit": str(auditor),
                "accepted": 19,
                "unsupported": 1,
                "quadrature_components_checked": 19 * 99 * audit["tetrahedra"],
            }
        )
        return run, audit, {row["case"]: row for row in summary["cases"]}

    def raw_field(record, case, model):
        run, audit, _cases = record
        path = run / f"results/{case}/{model}_field.npz"
        retain(path, audit["input_sha256"][str(path)])
        with np.load(path) as saved:
            tensor = saved["raw_native_tensor_Pa"].copy()
            peak = float(saved["vm_Pa"].max())
        return tensor, peak

    for geometry in ["ellipsoid", "capsule"]:
        reference_auditor = p / f"extended-finite-{geometry}-three-level-audit-v1"
        closed(reference_auditor)
        reference_audit = read_json(reference_auditor / "results/summary.json")
        assert reference_audit["all_original_and_new_fields_freshly_audited"]
        records = {}
        volumes_by_level = {}
        for level in range(3):
            key = f"spatial{level}"
            records[key] = load_run(f"extended-native-{geometry}-level{level}-models-v2", 8, 10)
            audited = next(row for row in reference_audit["rows"] if row["level"] == level)
            reference_run = Path(audited["run"])
            closed(reference_run)
            configuration = read_json(
                p / f"extended-finite-{geometry}-original-qualification-v1/results/configuration.json"
            )
            if level == 0:
                mesh_path = Path(configuration["baseline_mesh"])
            else:
                mesh_path = Path(read_json(reference_run / "results/configuration.json")["mesh"])
            mesh_hash = retain(mesh_path, audited["mesh_sha256"])
            with np.load(mesh_path) as saved:
                coordinates = saved["vertices"][saved["tetrahedra"]]
            volumes = abs(np.linalg.det(coordinates[:, 1:] - coordinates[:, :1])) / 6
            assert np.all(volumes > 0) and len(volumes) == audited["tetrahedra"]
            volumes_by_level[level] = volumes
            run, _audit, cases = records[key]
            for case in CASES:
                snapshot = cases[case]
                assert snapshot["mesh_sha256"] == mesh_hash
                reference_path = reference_run / f"results/{case}/reference_field.npz"
                retain(reference_path, reference_audit["input_sha256"][str(reference_path)])
                with np.load(reference_path) as saved:
                    reference_tensor = components(saved["tensor_Pa"])
                    reference_force = saved["force_N"].copy()
                    reference_peak = float(saved["vm_Pa"].max())
                native_reference = run / f"results/{case}/reference_field.npz"
                retain(native_reference)
                with np.load(native_reference) as saved:
                    np.testing.assert_allclose(saved["force_N"], reference_force, rtol=1e-11, atol=1e-13)
                denominator = linear_tensor_squared_integral(reference_tensor, volumes)
                assert denominator > 0
                for model in snapshot["models"]:
                    row = {
                        "geometry": geometry,
                        "case": case,
                        "model": model["model"],
                        "spatial_level": level,
                        "tetrahedra": len(volumes),
                        "accepted": model["accepted"],
                        "classification": model["classification"],
                    }
                    if model["accepted"]:
                        tensor, peak = raw_field(records[key], case, model["model"])
                        row.update(
                            {
                                "raw_native_volume_tensor_relative_L2_vs_audited_continuous_load": float(
                                    np.sqrt(
                                        linear_tensor_squared_integral(tensor - reference_tensor, volumes) / denominator
                                    )
                                ),
                                "raw_native_global_peak_Pa": peak,
                                "audited_continuous_load_raw_peak_Pa": reference_peak,
                                "raw_peak_relative_difference_vs_continuous_load": abs(peak - reference_peak)
                                / reference_peak,
                                "original_unweighted_corner_statistic": model[
                                    "tensor_relative_error_vs_continuous_load"
                                ],
                            }
                        )
                    biases.append(row)
        for label, ports, quadrature in [("p4q10", 4, 10), ("p16q10", 16, 10), ("p8q14", 8, 14)]:
            records[label] = load_run(
                f"native-{geometry}-fixed-original-p{ports}-q{quadrature}-models-v1", ports, quadrature
            )
        baseline = records["spatial0"]
        for label in ["p4q10", "p16q10", "p8q14"]:
            candidate = records[label]
            for case in CASES:
                old = baseline[2][case]
                new = candidate[2][case]
                for key in ["mesh_sha256", "material", "pose", "omega_rad_s", "patch_radius_m", "patches"]:
                    assert old[key] == new[key], (geometry, label, case, key)
                frozen_hashes = []
                for record in [baseline, candidate]:
                    path = record[0] / f"results/{case}/frozen_inputs.npz"
                    frozen_hashes.append(retain(path, record[1]["input_sha256"][str(path)]))
                assert frozen_hashes[0] == frozen_hashes[1]
                source, target = (candidate, baseline) if label == "p4q10" else (baseline, candidate)
                for index, model in enumerate(MODELS):
                    left, right = source[2][case]["models"][index], target[2][case]["models"][index]
                    assert left["accepted"] == right["accepted"]
                    row = {
                        "geometry": geometry,
                        "case": case,
                        "model": model,
                        "control": label,
                        "from_ports": left["ports"],
                        "to_ports": right["ports"],
                        "from_quadrature": left["native_quadrature"],
                        "to_quadrature": right["native_quadrature"],
                        "accepted": right["accepted"],
                        "classification": right["classification"],
                        "identical_frozen_inputs_sha256": frozen_hashes[0],
                    }
                    if right["accepted"]:
                        a, peak_a = raw_field(source, case, model)
                        b, peak_b = raw_field(target, case, model)
                        volumes = volumes_by_level[0]
                        denominator = linear_tensor_squared_integral(b, volumes)
                        assert denominator > 0
                        islands_a, islands_b = left["islands"], right["islands"]
                        assert len(islands_a) == len(islands_b) == len(old["patches"])
                        island_force_delta = np.array([r["resultant_N"] for r in islands_b]) - np.array(
                            [r["resultant_N"] for r in islands_a]
                        )
                        row.update(
                            {
                                "raw_native_volume_tensor_relative_L2_change": float(
                                    np.sqrt(linear_tensor_squared_integral(b - a, volumes) / denominator)
                                ),
                                "raw_native_unweighted_corner_relative_Frobenius_change": float(
                                    np.sqrt(
                                        np.sum((b - a) ** 2 * COMPONENT_WEIGHTS) / np.sum(b * b * COMPONENT_WEIGHTS)
                                    )
                                ),
                                "raw_peak_from_Pa": peak_a,
                                "raw_peak_to_Pa": peak_b,
                                "raw_peak_relative_change": abs(peak_b - peak_a) / peak_b,
                                "energy_relative_change": abs(right["energy_J"] - left["energy_J"])
                                / abs(right["energy_J"]),
                                "destination_disk_resultant_change_N": island_force_delta.tolist(),
                                "maximum_destination_disk_force_change_N": float(
                                    np.linalg.norm(island_force_delta, axis=1).max()
                                ),
                            }
                        )
                    comparisons.append(row)
                    print(json.dumps(row), flush=True)
    assert len(biases) == 120 and len(comparisons) == 120 and len(qualified_runs) == 12
    assert sum(row["accepted"] for row in comparisons) == sum(row["accepted"] for row in biases) == 114
    for filename, rows in [("numerical_axes.csv", comparisons), ("model_bias.csv", biases)]:
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected for name, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "same_mesh_numerical_axis_comparisons": comparisons,
        "three_spatial_level_model_bias_rows": biases,
        "qualified_native_runs": qualified_runs,
        "manufactured_exact_volume_norm_control_passed": True,
        "independently_audited_native_fields": 228,
        "unsupported_inputs_retained": 12,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Exact affine-tetrahedron integrals use all raw native corner components with symmetric shear multiplicity. Same-mesh ports4/8/16 at quadrature10 and quadrature10/14 at ports8 isolate declared numerical axes while preserving every frozen input byte and all four model rows. Three spatial-level model-bias rows compare each actual native field to its separately audited prescribed finite-traction FEM reference on that same mesh; they are not cross-mesh field-convergence estimates. The prescribed Euclidean traction reference and native intrinsic profiles differ, so these differences are conditional model/distance/integration effects, not pure model error or resolved-contact truth. Destination disk identities are not inferred continuum islands. Original unweighted statistics, every raw maximum and unsupported pressure row remain. No physical convergence, application/contact-feedback or warmed-cost completion is asserted.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
