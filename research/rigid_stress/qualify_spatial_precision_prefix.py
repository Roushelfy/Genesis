"""Compare all saved original spatial states with an immutable read-only prefix observation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .neohookean_reference import NeoHookeanBulk
from .qualify_articulated_precision import peak, tensor_from_f
from .summarize_fine_references import closed_source
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.baseline)
    observed = json.loads((args.observation / "observation.json").read_text())
    assert observed["complete_producer_history"] is False
    assert (
        hashlib.sha256((args.observation / "producer-source.sha256").read_bytes()).hexdigest()
        == observed["source_sha256"]
    )
    for row in observed["files"]:
        assert hashlib.sha256(Path(row["observed"]).read_bytes()).hexdigest() == row["sha256"]
    configs = [
        json.loads((args.baseline / "results/configuration.json").read_text()),
        json.loads((args.observation / "configuration.json").read_text()),
    ]
    assert {k: v for k, v in configs[0].items() if k != "output"} == {
        k: v for k, v in configs[1].items() if k != "output"
    }
    for name in ["reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz", "closure_state.npz"]:
        assert (
            hashlib.sha256((args.baseline / "results" / name).read_bytes()).hexdigest()
            == hashlib.sha256((args.observation / name).read_bytes()).hexdigest()
        )
    static, _ = restore_static(args.baseline)
    body, permutation = static.body, static.parts[0].permutation
    bulk = NeoHookeanBulk(body)
    original = sorted((args.baseline / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
    assert len(original) == observed["observed_states"]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for step, path in enumerate(original, 1):
        assert path.name == f"step{step:04d}.npz"
        with np.load(path) as a, np.load(args.observation / path.name) as b:
            assert a.files == b.files
            errors = {}
            for key in a.files:
                assert a[key].shape == b[key].shape
                delta = np.asarray(a[key], dtype=np.longdouble) - b[key]
                error, scale = float(np.linalg.norm(delta)), float(np.linalg.norm(b[key]))
                assert error <= 1e-5 * scale + 1e-12, (step, key, error, scale)
                errors[key] = {
                    "maximum_absolute_difference": float(abs(delta).max(initial=0)),
                    "L2_difference": error,
                    "relative_L2_difference": error / max(scale, 1e-12),
                }
            for key in ["body_component_permutation", "applied_force_N"]:
                np.testing.assert_array_equal(a[key], b[key])
            vectors = []
            for saved in [a, b]:
                vector = np.empty(body.basis.N, dtype=np.longdouble)
                vector[permutation] = saved["displacement"][: body.basis.N]
                vectors.append(vector)
        corners = [tensor_from_f(bulk.corner_deformation_gradients(u), body.lame) for u in vectors]
        quads = [tensor_from_f(bulk.deformation_gradients(u).transpose(2, 3, 0, 1), body.lame) for u in vectors]
        corner = float(np.linalg.norm(corners[0] - corners[1]) / np.linalg.norm(corners[1]))
        quad = float(np.linalg.norm(quads[0] - quads[1]) / np.linalg.norm(quads[1]))
        raw_peak = abs(peak(corners[0]) / peak(corners[1]) - 1)
        assert max(corner, quad, raw_peak) <= 1e-6
        rows.append(
            {
                "step": step,
                "all_saved_array_changes": errors,
                "all_corner_tensor_relative_change": corner,
                "all_quadrature_tensor_relative_change": quad,
                "raw_corner_peak_relative_change": raw_peak,
            }
        )
    result = {
        "scope": __doc__,
        "baseline": str(args.baseline),
        "observation": str(args.observation),
        "all_original_saved_states_checked": len(rows),
        "complete_candidate_history": False,
        "declared_numerical_comparison_budgets": {
            "state_relative": 1e-5,
            "state_absolute": 1e-12,
            "tensor_and_raw_peak_relative": 1e-6,
        },
        "rows": rows,
        "goal_completion": False,
        "limitations": "All original25 accepted saved states and complete arrays/corner/quadrature fields are compared against immutable copies captured while the producer was live. This uses the existing cached-reference comparison budgets, independently of unchanged1e-9N force acceptance. No producer completion or full80-state audit is inferred; those remain separate. Physical contact/geometry/spatial convergence and complete application/model/feedback/accuracy-cost acceptance remain open.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "states_checked": len(rows),
                "max_corner_tensor_relative_change": max(r["all_corner_tensor_relative_change"] for r in rows),
                "max_quadrature_tensor_relative_change": max(r["all_quadrature_tensor_relative_change"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
