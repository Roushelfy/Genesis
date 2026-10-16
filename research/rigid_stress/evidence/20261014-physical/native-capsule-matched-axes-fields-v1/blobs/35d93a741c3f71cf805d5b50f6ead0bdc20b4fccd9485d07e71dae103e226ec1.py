"""Identify the coupled grasp's free common-x translation in retained diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schur-run", type=Path, required=True)
    parser.add_argument("--replay", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    closed_source(args.schur_run)
    schur_path = args.schur_run / "results/rigid_schur.npz"
    with np.load(schur_path) as saved:
        matrix = saved["matrix"]
    common = np.array([1, 0, 0, 0, 0, 0, 1, 1]) / np.sqrt(3)
    _, values, right = np.linalg.svd(matrix)
    result = {
        "scope": __doc__,
        "schur_sha256": hashlib.sha256(schur_path.read_bytes()).hexdigest(),
        "common_translation_definition": "Body x translation and equal x translations of both backplates; elastic pad coordinates follow their rigid translations",
        "common_mode_relative_matrix_residual": float(np.linalg.norm(matrix @ common) / np.linalg.norm(matrix)),
        "weakest_mode_alignment_with_common_translation": float(abs(right[-1] @ common)),
        "singular_values_N_per_m": values.tolist(),
        "rows": [],
    }
    for run in args.replay:
        closed_source(run)
        summary = json.loads((run / "results/summary.json").read_text())
        reference = Path(summary["reference"])
        coordinate_map = sparse.load_npz(reference / "solution/coordinate_map.npz")
        left = coordinate_map[:, -2].nonzero()[0]
        right_plate = coordinate_map[:, -1].nonzero()[0]
        assert np.all(left % 3 == 0) and np.all(right_plate % 3 == 0)
        with np.load(run / "results/contact_hold.npz") as saved:
            increment = (saved["displacement"] - saved["previous_displacement"]).reshape(-1, 3)
            external = saved["applied_force_N"].reshape(-1, 3)
            assert abs(external[:, 0].sum()) <= 1e-13
            centre_change = (increment[left // 3, 0].mean() + increment[right_plate // 3, 0].mean()) / 2
            gauged = increment.copy()
            gauged[:, 0] -= centre_change
        result["rows"].append(
            {
                "replay": str(run),
                "zero_total_applied_x_force_N": float(external[:, 0].sum()),
                "common_centre_increment_m": float(centre_change),
                "raw_maximum_nodal_increment_m": float(np.linalg.norm(increment, axis=1).max()),
                "maximum_increment_after_common_translation_m": float(np.linalg.norm(gauged, axis=1).max()),
                "tensor_relative_hold_change": summary["constant_load_hold_tensor_relative_change"],
            }
        )
    assert result["weakest_mode_alignment_with_common_translation"] > 0.99
    assert result["common_mode_relative_matrix_residual"] < 1e-8
    result["limitations"] = (
        "This identifies a gauge freedom, not missing physical contact support. The other weak mode and fixed-stencil wrench failures remain separate. Bulk/force controls are unchanged. Pose comparisons must remove this common translation; imposing a centre support under nonzero net x force would change the physical problem."
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
