"""Compare sealed common states from original-history continuations with identical physical acceptance gates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .neohookean_reference import NeoHookeanBulk
from .qualify_articulated_precision import peak, tensor_from_f
from .skfem_reference import SkfemElasticReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observed = json.loads((args.observation / "observation.json").read_text())
    assert not observed["complete_producer_history"]
    for row in observed["files"]:
        assert hashlib.sha256(Path(row["observed"]).read_bytes()).hexdigest() == row["sha256"]
    dirs = [args.observation / name for name in ["baseline", "candidate"]]
    configs = [json.loads((root / "configuration.json").read_text()) for root in dirs]
    assert {k: v for k, v in configs[0].items() if k not in {"output", "mjcf"}} == {
        k: v for k, v in configs[1].items() if k not in {"output", "mjcf"}
    }
    assert (dirs[0] / "panda.xml").read_bytes() == (dirs[1] / "panda.xml").read_bytes()
    with np.load(dirs[0] / "reference_geometry.npz") as a, np.load(dirs[1] / "reference_geometry.npz") as b:
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        body = SkfemElasticReference(a["vertices"], a["tetrahedra"], 1e10, 0.3, 2000)
        permutation = a["body_component_permutation"].copy()
    bulk = NeoHookeanBulk(body)
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    for step in observed["observed_steps"]:
        with np.load(dirs[0] / f"step{step:05d}.npz") as a, np.load(dirs[1] / f"step{step:05d}.npz") as b:
            assert a.files == b.files
            changes = {}
            for key in a.files:
                assert a[key].shape == b[key].shape
                difference = np.asarray(a[key], dtype=np.longdouble) - b[key]
                delta, norm = float(np.linalg.norm(difference)), float(np.linalg.norm(b[key]))
                assert delta <= 1e-5 * norm + 1e-12, (step, key, delta, norm)
                changes[key] = {"absolute_L2": delta, "relative_L2": delta / max(norm, 1e-12)}
            for key in ["applied_force_N", "controller_target", "controller_caps"]:
                np.testing.assert_array_equal(a[key], b[key])
            vectors = []
            for state in [a, b]:
                vector = np.empty(body.basis.N, dtype=np.longdouble)
                vector[permutation] = state["displacement"][: body.basis.N]
                vectors.append(vector)
        corners = [tensor_from_f(bulk.corner_deformation_gradients(u), body.lame) for u in vectors]
        quads = [tensor_from_f(bulk.deformation_gradients(u).transpose(2, 3, 0, 1), body.lame) for u in vectors]
        corner = float(np.linalg.norm(corners[0] - corners[1]) / np.linalg.norm(corners[1]))
        quad = float(np.linalg.norm(quads[0] - quads[1]) / np.linalg.norm(quads[1]))
        peak_change = abs(peak(corners[0]) / peak(corners[1]) - 1)
        assert max(corner, quad, peak_change) <= 1e-6
        rows.append(
            {
                "step": step,
                "all_saved_array_changes": changes,
                "all_corner_tensor_relative_change": corner,
                "all_quadrature_tensor_relative_change": quad,
                "raw_global_corner_peak_relative_change": peak_change,
            }
        )
    result = {
        "scope": __doc__,
        "observation": str(args.observation),
        "rows": rows,
        "all1784_original_prefix_bytes_identical_between_runs": observed[
            "all1784_original_prefix_bytes_identical_between_runs"
        ],
        "declared_existing_comparison_budgets": {
            "array_relative": 1e-5,
            "array_absolute": 1e-12,
            "tensor_and_peak_relative": 1e-6,
        },
        "complete_producer_history": False,
        "goal_completion": False,
        "limitations": "Sealed live observations compare only the stated already-saved common physical states. Every1784 previously audited state is copied byte-identically by both producers. No producer completion or complete future-history field equivalence is inferred. Original1e-9 physical force/robot gates remain unchanged and independent all-state audits remain separate. Physical convergence, model/feedback and verified accuracy/cost are not established.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "states_compared": len(rows),
                "max_corner_tensor_relative_change": max(r["all_corner_tensor_relative_change"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
