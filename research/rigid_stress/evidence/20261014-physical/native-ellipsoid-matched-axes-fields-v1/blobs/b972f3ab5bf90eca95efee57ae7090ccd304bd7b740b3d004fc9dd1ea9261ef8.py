"""Build2/4/8-layer volume controls with unchanged original covered-egg boundaries and canonical P2 contact trace."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .audit_franka_reference import rotation
from .refine_egg_thickness import canonical_rows, prism_tetrahedra
from .refine_fixed_geometry import geometric_integrals
from .summarize_fine_references import closed_source


def mesh_metrics(vertices, tetrahedra):
    determinant = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
    assert np.all(abs(determinant) > 0)
    oriented = tetrahedra.copy()
    negative = determinant < 0
    oriented[negative, 2:4] = oriented[negative, 3:1:-1]
    sides = oriented[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    sorted_sides, inverse, counts = np.unique(np.sort(sides, axis=1), axis=0, return_inverse=True, return_counts=True)
    assert counts.max() == 2
    inversions = (sides[:, 0] > sides[:, 1]).astype(int) + (sides[:, 0] > sides[:, 2]) + (sides[:, 1] > sides[:, 2])
    signs = 1 - 2 * (inversions % 2)
    orientation_sum = np.bincount(inverse, weights=signs)
    assert np.all(orientation_sum[counts == 2] == 0)
    boundary = sorted_sides[counts == 1]
    volumes, areas, com = geometric_integrals(vertices, oriented, boundary)
    corners = vertices[oriented]
    summed = corners.sum(axis=1)
    second = np.einsum("e,ecj,eck->jk", volumes / 20, corners, corners)
    second += np.einsum("e,ej,ek->jk", volumes / 20, summed, summed)
    second -= volumes.sum() * np.outer(com, com)
    inertia = 2000 * (np.trace(second) * np.eye(3) - second)
    return (
        oriented,
        boundary,
        {
            "volume_m3": float(volumes.sum()),
            "mass_kg": float(2000 * volumes.sum()),
            "com_m": com.tolist(),
            "surface_area_m2": float(areas.sum()),
            "inertia_about_com_kg_m2": inertia.tolist(),
            "minimum_tetrahedron_volume_m3": float(volumes.min()),
            "opposite_oriented_internal_faces_verified": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.source_run)
    assert int((args.source_run / "returncode.txt").read_text()) == 0
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    config_path = args.source_run / "results/configuration.json"
    retain(config_path)
    config = json.loads(config_path.read_text())
    assert config["mode"] == "cad" and config["cold_closure"]
    parent = Path(config["geometry_run"])
    closed_source(parent)
    assert int((parent / "returncode.txt").read_text()) == 0
    metadata = parent / "results/summary.json"
    seed = parent / f"results/cad-level{config['level']}.npz"
    for path in [metadata, seed]:
        retain(path)
    old_row = next(
        row
        for row in json.loads(metadata.read_text())["rows"]
        if row["mode"].lower() == "cad" and row["level"] == config["level"]
    )
    assert inputs[str(seed)] == old_row["mesh_sha256"]
    with np.load(seed) as saved:
        original, old_tets, surface = [saved[key].copy() for key in ["vertices", "tetrahedra", "surface_triangles"]]
    assert len(original) % 3 == 0
    count = len(original) // 3
    assert surface.max() < count
    np.testing.assert_array_equal(
        canonical_rows(old_tets), canonical_rows(prism_tetrahedra(np.sort(surface, axis=1), count, 2))
    )
    _, old_boundary, old_metrics = mesh_metrics(original, old_tets)
    closure_path = Path(config["closure_run"]) / "results/configuration.json"
    retain(closure_path)
    capture = Path(json.loads(closure_path.read_text())["capture"])
    capture_path = capture / "results/summary.json"
    reference_path = args.source_run / "results/reference_geometry.npz"
    for path in [capture_path, reference_path]:
        retain(path)
    frame = [row for row in json.loads(capture_path.read_text())["captures"] if row["mode"] == "stationary_lift_hold"][
        -1
    ]
    transform = rotation(frame["egg_quaternion_wxyz"])
    shift = np.array(frame["egg_position_m"]) - frame["egg_world_com_m"]
    with np.load(reference_path) as saved:
        np.testing.assert_allclose(original @ transform.T + shift, saved["vertices"], rtol=0, atol=1e-14)
        fixed = saved["fixed_initializer_points_m"].copy()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for level, layers in enumerate([2, 4, 8], start=1):
        blocks = [original.copy()]
        ids = []
        for index in range(layers + 1):
            fraction = index / layers
            if fraction in [0, 0.5, 1]:
                ids.append(np.arange(count) + int(2 * fraction) * count)
                continue
            half = int(fraction > 0.5)
            mix = 2 * fraction - half
            block = (1 - mix) * original[half * count : (half + 1) * count] + mix * original[
                (half + 1) * count : (half + 2) * count
            ]
            ids.append(np.arange(count) + sum(len(item) for item in blocks))
            blocks.append(block)
        vertices = np.concatenate(blocks)
        ids = np.array(ids)
        tets = ids.reshape(-1)[prism_tetrahedra(np.sort(surface, axis=1), count, layers)]
        tets, boundary, metrics = mesh_metrics(vertices, tets)
        np.testing.assert_array_equal(vertices[: len(original)], original)
        np.testing.assert_array_equal(canonical_rows(boundary), canonical_rows(old_boundary))
        np.testing.assert_array_equal(vertices[old_boundary], original[old_boundary])
        for key in ["volume_m3", "mass_kg", "surface_area_m2", "inertia_about_com_kg_m2"]:
            np.testing.assert_allclose(metrics[key], old_metrics[key], rtol=1e-11, atol=1e-18)
        np.testing.assert_allclose(metrics["com_m"], old_metrics["com_m"], rtol=0, atol=1e-14)
        all_edges = np.unique(np.sort(tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)), axis=0)
        points = np.concatenate((vertices, vertices[all_edges].mean(axis=1))) @ transform.T + shift
        distances = cKDTree(points).query(fixed)[0]
        assert distances.max() < 1e-12
        edges = np.unique(np.sort(boundary[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
        canonical_trace = np.concatenate((vertices[np.unique(boundary)], vertices[edges].mean(axis=1)))
        baseline_trace = np.concatenate((original[np.unique(old_boundary)], original[edges].mean(axis=1)))
        np.testing.assert_array_equal(canonical_trace, baseline_trace)
        path = args.output / f"cad-level{level}.npz"
        if layers == 2:
            shutil.copy2(seed, path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == inputs[str(seed)]
        else:
            np.savez_compressed(path, vertices=vertices, tetrahedra=tets, surface_triangles=surface)
        row = {
            **old_row,
            **metrics,
            "mode": "CAD",
            "level": level,
            "mesh": str(path),
            "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tetrahedra": len(tets),
            "vertices": len(vertices),
            "through_thickness_layers": layers,
            "refinement_axis": "Volume thickness only at fixed original polyhedral boundaries and canonical P2 surface trace",
            "all_original_vertices_byte_preserved": True,
            "both_original_boundary_triangle_sets_identical": True,
            "canonical_boundary_P2_coordinates_byte_identical": True,
            "original_initializer_all_anchors_retained": True,
            "original_initializer_nearest_P2_node_distances_m": distances.tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "source_run": str(args.source_run),
        "rows": rows,
        "input_sha256": inputs,
        "new_equilibrium_states_solved": 0,
        "goal_completion": False,
        "limitations": "These assets preserve all original vertices including the mid-layer, exact inner/outer triangle identities, canonical surface P2 coordinates, mass/COM/inertia, and original six physical initializer anchors. The2-layer asset is the original NPZ byte-for-byte. Only internal layer count/basis changes; no new CAD projection, rounding, contact law or load radius is introduced. Internal meshes are not claimed per-parent nested simplicial partitions. Canonical trace identity is not yet an actual reconstructed collision-mesh equivalence or accepted contact state: subsequent producer/source/force/full-field qualification must verify those scopes. Existing64 initializer controls, five dynamic steps and all physical budgets remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
