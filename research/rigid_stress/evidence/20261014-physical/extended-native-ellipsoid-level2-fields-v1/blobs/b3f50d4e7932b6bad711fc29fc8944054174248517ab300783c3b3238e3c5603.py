"""Recover the actual temporary body initialization coordinates from saved U coordinate maps."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import qr
from scipy.spatial import cKDTree

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, inputs, baseline, coordinates = [], {}, None, []
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        path = run / "solution/original_coordinate_map.npz"
        mapping = sparse.load_npz(path).tocsc()
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        path = run / "solution/fields.npz"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            n = len(saved["body_component_permutation"])
            rest = saved["rest_positions"].copy()
        coordinates.append(rest[: n // 3])
        singles = np.flatnonzero(np.diff(mapping.indptr) == 1)
        nonzero = mapping.indptr[singles]
        np.testing.assert_allclose(mapping.data[nonzero], 1, rtol=0, atol=1e-15)
        independent = mapping.indices[nonzero]
        constrained = np.setdiff1d(np.arange(n), independent[independent < n])
        assert len(constrained) == 6
        points, components = rest[constrained // 3], constrained % 3
        if baseline is None:
            baseline = points.copy(), components.copy()
        distances = np.linalg.norm(points[:, None] - baseline[0][None], axis=-1)
        distances[components[:, None] != baseline[1][None]] = np.inf
        closest = distances.min(axis=1)
        rows.append(
            {
                "run": str(run),
                "body_dofs": n,
                "initialization_coordinate_rows": constrained.tolist(),
                "initialization_points_m": points.tolist(),
                "initialization_components": components.tolist(),
                "component_matched_nearest_baseline_distance_m": closest.tolist(),
                "all_six_physical_coordinate_constraints_match_baseline": bool(np.all(closest <= 1e-12)),
            }
        )
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    shared = coordinates[0].copy()
    for points in coordinates[1:]:
        distances = cKDTree(points).query(shared)[0]
        shared = shared[distances <= 1e-12]
    assert len(shared) >= 3
    relative = shared - shared.mean(axis=0)
    modes = np.zeros((3 * len(shared), 6))
    modes[:, :3] = np.tile(np.eye(3), (len(shared), 1))
    for axis in range(3):
        modes[:, axis + 3] = np.cross(np.eye(3)[axis], relative).reshape(-1)
    scaled = modes / np.linalg.norm(modes, axis=0)
    _, triangular, pivot = qr(scaled.T, pivoting=True, mode="economic")
    selected = pivot[:6]
    assert np.min(abs(triangular.diagonal())) > 1e-12
    anchors, components = shared[selected // 3], selected % 3
    args.output.mkdir(parents=True, exist_ok=False)
    anchor_path = args.output / "common_initialization_anchors.npz"
    np.savez_compressed(anchor_path, positions_m=anchors, components=components, shared_candidate_positions_m=shared)
    result = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": inputs,
        "common_candidate_points": len(shared),
        "common_anchor_file": str(anchor_path),
        "common_anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "common_anchor_positions_m": anchors.tolist(),
        "common_anchor_components": components.tolist(),
        "common_anchor_scaled_rigid_mode_condition": float(np.linalg.cond(scaled[selected])),
        "goal_completion": False,
        "limitations": "The six omitted independent body entries define the held numerical rigid-coordinate origin in initialization; the complete original coordinate map restores six free body modes after release. This audit reports actual geometric coordinate/component identities. Different initialization constraints do not by themselves measure their effect on final friction history or stress, and matching coordinates do not establish physical convergence. The distinction remains explicit when interpreting coupled body/geometry refinements.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
