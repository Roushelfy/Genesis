"""Audit resolved-contact mesh changes and reference-only gauge/cache candidates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .sample_load_fields import common_physical_probes, locate_probes, sample_field
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source, relative_change


def read_run(run):
    closed_source(run)
    assert int((run / "returncode.txt").read_text()) == 0
    result = json.loads((run / "solution/result.json").read_text())
    config = json.loads((run / "solution/configuration.json").read_text())
    with np.load(run / "solution/fields.npz") as saved:
        field = {key: saved[key].copy() for key in saved.files}
    assert result["complete_constrained_residual_N"] <= 1e-9
    assert result["complete_body_residual_N"] <= 1e-9
    return result, config, field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--finer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, baseline_config, baseline_field = read_run(args.baseline)
    candidates = []
    for run in args.candidate:
        result, config, field = read_run(run)
        for key, value in baseline_config.items():
            if key != "output":
                assert config[key] == value, (run, key)
        for key in ("rest_positions", "surface_triangles", "body_component_permutation"):
            np.testing.assert_array_equal(field[key], baseline_field[key])
        delta = (field["displacement"] - baseline_field["displacement"]).reshape(-1, 3)
        common_x_shift = delta[:, 0].mean()
        delta[:, 0] -= common_x_shift
        tensor_relative, tensor_absolute = relative_change(baseline_field["body_tensor_Pa"], field["body_tensor_Pa"])
        np.testing.assert_allclose(field["body_tensor_Pa"], baseline_field["body_tensor_Pa"], rtol=1e-6, atol=0.01)
        assert np.linalg.norm(delta, axis=1).max() < 1e-8
        force_relative, force_absolute = relative_change(baseline_field["contact_force_N"], field["contact_force_N"])
        candidates.append(
            {
                "run": str(run),
                "full_body_tensor_relative_difference": tensor_relative,
                "full_body_tensor_absolute_difference_Pa": tensor_absolute,
                "full_contact_force_relative_difference": force_relative,
                "full_contact_force_absolute_difference_N": force_absolute,
                "common_x_translation_m": float(common_x_shift),
                "maximum_all_part_displacement_difference_after_translation_m": float(
                    np.linalg.norm(delta, axis=1).max()
                ),
                "complete_original_coordinate_residual_N": result["complete_constrained_residual_N"],
                "empty_factor_reuses": result["empty_factor_reuses"],
                "supported_factorizations": result["supported_factorizations"],
            }
        )
    finer, finer_config, finer_field = read_run(args.finer)
    for key, value in baseline_config.items():
        if key not in ("output", "mesh", "pad_resolution"):
            assert finer_config[key] == value, key
    probes = common_physical_probes(baseline_config["geometry"])
    levels, observed, input_hashes = [], [], {}
    for run, result, config, field in (
        (args.baseline, baseline, baseline_config, baseline_field),
        (args.finer, finer, finer_config, finer_field),
    ):
        mesh_path = Path(config["mesh"])
        assert hashlib.sha256(mesh_path.read_bytes()).hexdigest() == result["mesh_sha256"]
        with np.load(mesh_path) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        displacement = np.empty(reference.basis.N)
        displacement[field["body_component_permutation"]] = field["displacement"][: reference.basis.N]
        elastic = reference.align_rigid_gauge(displacement)
        elements, valid, barycentric = locate_probes(reference.mesh, probes)
        tensor, vm, u = sample_field(
            reference.basis,
            elements,
            valid,
            barycentric,
            probes,
            {"tensor_Pa": field["body_tensor_Pa"], "displacement": elastic},
        )
        observations = {"valid": valid, "tensor_Pa": tensor, "vm_Pa": vm, "displacement_m": u}
        observed.append(observations)
        np.savez_compressed(args.output / f"level{len(levels)}_probes.npz", points_m=probes, **observations)
        levels.append(
            {
                "run": str(run),
                "tetrahedra": reference.mesh.nelements,
                "pad_resolution": config["pad_resolution"],
                "raw_global_peak_Pa": result["raw_body_peak_Pa"],
                "raw_global_peak_position_m": result["raw_body_peak_position_m"],
                "body_elastic_energy_J": float(0.5 * elastic @ (reference.stiffness @ elastic)),
                "outside_probe_indices": np.flatnonzero(~valid).tolist(),
                "complete_constrained_residual_N": result["complete_constrained_residual_N"],
            }
        )
        for name in ("result.json", "configuration.json", "fields.npz"):
            path = run / "solution" / name
            input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    common = observed[0]["valid"] & observed[1]["valid"]
    assert common.any()
    changes = {"common_valid_probes": int(common.sum())}
    for key in ("tensor_Pa", "vm_Pa", "displacement_m"):
        relative, absolute = relative_change(observed[0][key][common], observed[1][key][common])
        changes[key] = {"relative_change": relative, "absolute_change": absolute}
    for key in ("raw_global_peak_Pa", "body_elastic_energy_J"):
        changes[key] = abs(levels[0][key] / levels[1][key] - 1)
    report = {
        "scope": __doc__,
        "candidates": candidates,
        "levels": levels,
        "adjacent_changes": changes,
        "probe_coordinate_sha256": hashlib.sha256(probes.tobytes()).hexdigest(),
        "input_sha256": input_hashes,
        "goal_completion": False,
        "limitations": "Two useful contact levels do not establish a tail. Body geometry and pad spatial discretization change together; separate axes, contact regularization, nonlinear pad compliance and full egg/Franka comparisons remain required. Cache and centre fixing are reference-only changes under exactly zero total x force. No wall-time speedup is inferred from factor reuse counts.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidates": candidates, "adjacent_changes": changes}), flush=True)


if __name__ == "__main__":
    main()
