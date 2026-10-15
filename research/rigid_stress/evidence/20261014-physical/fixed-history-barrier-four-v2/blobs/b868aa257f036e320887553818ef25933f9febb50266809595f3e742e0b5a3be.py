"""Compare all744 audited U stages across a fixed-history contact-barrier distance sequence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_objective_frame import finite_frame
from .neohookean_reference import NeoHookeanBulk
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=4, required=True)
    parser.add_argument("--audit", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    configs, audits, records, original_geometry = [], [], [], None
    for index, (run, audit) in enumerate(zip(args.run, args.audit, strict=True)):
        for path in [run, audit]:
            closed_source(path)
            assert int((path / "returncode.txt").read_text()) == 0
        config_path = run / "solution/configuration.json"
        audit_path = audit / "results/summary.json"
        retain(config_path)
        retain(audit_path)
        config = json.loads(config_path.read_text())
        result = json.loads(audit_path.read_text())
        assert Path(result["run"]).resolve() == run.resolve()
        assert result["stages_checked"] == len(result["rows"]) == 744
        assert result["all_retained_stages_independent_current_friction_and_original_force_gate"]
        assert result["all_retained_stages_full_corner_quadrature_and_raw_bounds"]
        assert all(row["independent_complete_constrained_residual_N"] <= 1e-9 for row in result["rows"])
        assert config["dhat"] == [2.5e-6, 1.25e-6, 6.25e-7, 3.125e-7][index]
        assert config["force_steps"] == 64 and config["pad_resolution"] == 5
        excluded = {"dhat", "output", "contact_initialization_steps"}
        if configs:
            assert {k: v for k, v in config.items() if k not in excluded} == {
                k: v for k, v in configs[0].items() if k not in excluded
            }
            assert config["contact_initialization_steps"] == 604
        for path, digest in result["input_sha256"].items():
            assert retain(Path(path)) == digest
        manifest_path = run / "accepted-contact-history/manifest.json"
        retain(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["accepted_stages"] == 744
        assert [(call["phase"], call["load_steps"]) for call in manifest["calls"]] == [
            ("collision_free_approach", 12),
            ("contact_initialization", 604),
            ("held_body_force_control", 64),
            ("free_body_force_control", 64),
        ]
        stages_path = run / "accepted-contact-history/stages.jsonl"
        assert retain(stages_path) == manifest["stages_sha256"]
        stages = [json.loads(line) for line in stages_path.read_text().splitlines()]
        assert len(stages) == 744
        field_path = run / "solution/fields.npz"
        retain(field_path)
        with np.load(field_path) as saved:
            geometry = {
                key: saved[key].copy() for key in ["rest_positions", "surface_triangles", "body_component_permutation"]
            }
        if original_geometry is None:
            original_geometry = geometry
        else:
            for key, value in geometry.items():
                np.testing.assert_array_equal(value, original_geometry[key])
        for name in ["neohookean_reference", "skfem_reference"]:
            assert (
                Path(__file__).with_name(name + ".py").read_bytes()
                == (run / "source/research/rigid_stress" / (name + ".py")).read_bytes()
            )
        configs.append(config)
        audits.append(result)
        records.append(stages)
    mesh_path = Path(configs[0]["mesh"])
    retain(mesh_path)
    with np.load(mesh_path) as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    bulk = NeoHookeanBulk(body)
    assert body.mesh.nelements == 845
    weights = bulk.basis.dx
    assert np.all(weights > 0)
    # A declared absolute stress scale identifies near-zero denominators only;
    # all absolute differences and raw maxima remain, with no new acceptance gate.
    tensor_denominator_floor = 1e-3 * np.sqrt(weights.sum())
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    for ordinal in range(1, 745):
        states, level_rows = [], []
        for index, (run, audit) in enumerate(zip(args.run, args.audit, strict=True)):
            record = records[index][ordinal - 1]
            other = records[0][ordinal - 1]
            assert (record["ordinal"], record["phase"], record["step"]) == (ordinal, other["phase"], other["step"])
            audited = audits[index]["rows"][ordinal - 1]
            assert (
                audited["ordinal"] == ordinal
                and audited["phase"] == record["phase"]
                and audited["step"] == record["step"]
            )
            source = run / "accepted-contact-history" / record["file"]
            assert retain(source) == record["sha256"]
            with np.load(source) as saved:
                displacement = saved["displacement"]
            vector = np.empty(body.basis.N)
            vector[original_geometry["body_component_permutation"]] = displacement[: body.basis.N]
            rotation, _translation, local = finite_frame(body, vector)
            field = audit / f"results/stage{ordinal:06d}.npz"
            retain(field)
            with np.load(field) as saved:
                tensor = rotation.T @ saved["body_quadrature_tensor_Pa"] @ rotation
            assert tensor.shape == (*weights.shape, 3, 3)
            deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
            vm = np.sqrt(1.5 * np.sum(deviator**2, axis=(-2, -1)))
            energy, _, _ = bulk.evaluate(vector, False)
            norm = float(np.sqrt(np.sum(weights[..., None, None] * tensor**2)))
            source_rows = []
            for source_row in audited["sources"]:
                source_rows.append(
                    {
                        "source": source_row["source"],
                        "normal_force_material_N": (rotation.T @ source_row["normal_force_on_body_N"]).tolist(),
                        "shear_force_material_N": (rotation.T @ source_row["shear_force_on_body_N"]).tolist(),
                        "moment_about_current_com_material_Nm": (
                            rotation.T @ source_row["total_moment_on_body_about_current_com_Nm"]
                        ).tolist(),
                    }
                )
            level_rows.append(
                {
                    "barrier_distance_m": configs[index]["dhat"],
                    "volume_weighted_tensor_L2_Pa_m1p5": norm,
                    "raw_global_peak_lower_Pa": audited["raw_global_peak_lower_Pa"],
                    "raw_global_peak_upper_Pa": audited["raw_global_peak_upper_Pa"],
                    "body_nonlinear_energy_J": float(energy),
                    "sources": source_rows,
                }
            )
            states.append((tensor, vm, local))
        pairs = []
        for first, second in [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)]:
            a, b = states[first], states[second]
            absolute = float(np.sqrt(np.sum(weights[..., None, None] * (a[0] - b[0]) ** 2)))
            denominator = level_rows[second]["volume_weighted_tensor_L2_Pa_m1p5"]
            vm_absolute = float(np.sqrt(np.sum(weights * (a[1] - b[1]) ** 2)))
            delta_u = a[2] - b[2]
            displacement_absolute = float(np.sqrt(max(0, float(delta_u @ (body.mass @ delta_u) / body.total_mass))))
            pairs.append(
                {
                    "from_barrier_m": configs[first]["dhat"],
                    "to_barrier_m": configs[second]["dhat"],
                    "objective_tensor_absolute_L2_Pa_m1p5": absolute,
                    "finer_tensor_reference_L2_Pa_m1p5": denominator,
                    "objective_tensor_relative_L2": absolute / denominator
                    if denominator > tensor_denominator_floor
                    else None,
                    "near_zero_reference_denominator": bool(denominator <= tensor_denominator_floor),
                    "von_mises_absolute_L2_Pa_m1p5": vm_absolute,
                    "objective_displacement_mass_RMS_difference_m": displacement_absolute,
                }
            )
        rows.append(
            {
                "ordinal": ordinal,
                "phase": records[0][ordinal - 1]["phase"],
                "step": records[0][ordinal - 1]["step"],
                "is_physical_time_sample": False,
                "levels": level_rows,
                "pairs": pairs,
            }
        )
        if ordinal % 50 == 0:
            print(json.dumps({"all_level_stages_compared": ordinal}), flush=True)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "stages_compared_per_level": 744,
        "levels": 4,
        "rows": rows,
        "absolute_stress_scale_for_near_zero_relative_denominators_Pa": 1e-3,
        "tensor_denominator_floor_Pa_m1p5": tensor_denominator_floor,
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "goal_completion": False,
        "limitations": "All744 original quasi-static stages have independently audited current normal/friction forces, original1e-9N balances and raw fields at each barrier distance. Geometry, mass, material, pad5, original initialization coordinates,12/604/64/64 stage counts, force, gravity, friction and1e-8m per-stage tangential mollifier remain fixed. Proper finite material frames and the unchanged reference quadrature define volume-weighted tensor comparisons; these are not exact nonlinear continuum integrals. Near-zero reference norms retain every absolute error and raw peak; only the ill-conditioned relative quotient is null under the declared1e-3Pa scale. This is one coarse regularization-sensitivity family, not spatial/contact-pressure/history convergence, a hard-contact continuum limit, full Franka/model/feedback acceptance or physically verified warmed native cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
