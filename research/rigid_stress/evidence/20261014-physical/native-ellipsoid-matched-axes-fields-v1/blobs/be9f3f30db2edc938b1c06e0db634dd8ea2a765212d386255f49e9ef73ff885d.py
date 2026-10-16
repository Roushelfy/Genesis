"""Add fixed shared1–50um material probes near original contact samples without replacing any original norm or raw maximum."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .audit_objective_frame import finite_frame
from .compare_covered_cad_fields import point_fields
from .compare_spatial_franka import qualify, weighted_change
from .summarize_fine_references import closed_source
from .wide_spatial_franka import restore_static


def shared_probes(body, rest, groups, normal):
    facets = body.mesh.boundary_facets()
    faces = body.mesh.facets[:, facets].T
    vertices = body.mesh.p.T
    triangles = vertices[faces]
    edges = triangles[:, 1:] - triangles[:, :1]
    gram_inverse = np.linalg.inv(np.einsum("fai,fbi->fab", edges, edges))
    tree = cKDTree(triangles.mean(axis=1))
    points, records = [], []
    for source, force in enumerate(normal, start=1):
        anchors = np.flatnonzero((groups == 0) & np.any(force != 0, axis=1))
        assert len(anchors) > 0
        for node in anchors:
            anchor = rest[node]
            candidates = np.atleast_1d(tree.query(anchor, k=min(64, len(facets)))[1])
            delta = anchor - triangles[candidates, 0]
            local = np.einsum("kab,kbj,kj->ka", gram_inverse[candidates], edges[candidates], delta)
            projected = triangles[candidates, 0] + np.einsum("ka,kaj->kj", local, edges[candidates])
            plane_distance = np.linalg.norm(projected - anchor, axis=1)
            valid = (local >= -1e-10).all(axis=1) & (local.sum(axis=1) <= 1 + 1e-10) & (plane_distance <= 1e-13)
            assert valid.any(), (source, int(node))
            # A deterministic physical boundary owner; no point is discarded.
            candidate = int(candidates[valid][np.argmin(facets[candidates[valid]])])
            face, triangle = faces[candidate], triangles[candidate]
            cell = int(body.mesh.f2t[0, facets[candidate]])
            opposite = np.setdiff1d(body.mesh.t[:, cell], face)
            assert len(opposite) == 1
            outward = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            outward /= np.linalg.norm(outward)
            if outward @ (vertices[opposite[0]] - triangle[0]) > 0:
                outward *= -1
            assert outward @ (vertices[opposite[0]] - triangle[0]) < 0
            for mix in np.sqrt([2, 3, 5]) / 20:
                surface_point = (1 - mix) * anchor + mix * triangle.mean(axis=0)
                for depth_um in [1, 2, 5, 10, 20, 50]:
                    depth = depth_um * 1e-6
                    points.append(surface_point - depth * outward)
                    records.append(
                        {
                            "source": source,
                            "original_surface_vertex": int(node),
                            "original_anchor_m": anchor.tolist(),
                            "baseline_boundary_facet": int(facets[candidate]),
                            "outward_facet_normal": outward.tolist(),
                            "centroid_mix_fraction": float(mix),
                            "declared_facet_normal_depth_m": depth,
                            "original_anchor_boundary_plane_distance_m": float(
                                np.linalg.norm(projected[candidates == candidate][0] - anchor)
                            ),
                        }
                    )
    points = np.array(points)
    assert len(points) == len(records) and len(np.unique(points, axis=0)) == len(points)
    return points, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest

    for run in [args.comparison, args.source_audit]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        retain(run / "results/summary.json")
    original = json.loads((args.comparison / "results/summary.json").read_text())
    sources = json.loads((args.source_audit / "results/summary.json").read_text())
    assert original["common_physical_probes"] == 158340 and len(original["rows"]) == 3
    assert sources["selector_mesh_reconstruction_qualified"] and len(sources["rows"]) == 15
    args.output.mkdir(parents=True, exist_ok=False)
    original_probe = None
    previous, rows, changes, original_changes = None, [], [], []
    for index, row in enumerate(original["rows"]):
        run = Path(row["run"])
        static, config = restore_static(run)
        closure_path = Path(config["closure_run"]) / "results/configuration.json"
        retain(closure_path)
        capture = Path(json.loads(closure_path.read_text())["capture"])
        retain(capture / "results/summary.json")
        assert int((run / "returncode.txt").read_text()) == 0
        assert row["step"] == 5 and row["time_s"] == config["dt"] * row["step"] == 0.00625
        source_row = [item for item in sources["rows"] if item["run"] == str(run) and item["step"] == row["step"]]
        assert len(source_row) == 1
        source_field = args.source_audit / f"results/case{index:02d}-step{row['step']:04d}.npz"
        field = args.comparison / f"results/case{index:02d}.npz"
        state = run / f"results/step{row['step']:04d}.npz"
        for path in [
            source_field,
            field,
            state,
            run / "results/reference_geometry.npz",
            run / "results/contact_geometry.npz",
            run / "results/contact_node_map.npz",
            run / "results/configuration.json",
        ]:
            retain(path)
        with np.load(field) as saved:
            old_points, weights, original_tensor = [
                saved[key].copy() for key in ["common_points_m", "common_weights_m3", "objective_probe_tensor_Pa"]
            ]
        assert len(old_points) == len(weights) == original["common_physical_probes"]
        if original_probe is not None:
            np.testing.assert_array_equal(old_points, original_probe[0])
            np.testing.assert_array_equal(weights, original_probe[1])
            change = weighted_change(original_probe[2][None], original_tensor[None], weights)
            for key in change:
                np.testing.assert_allclose(
                    change[key], original["comparisons"][index - 1]["tensor"][key], rtol=1e-12, atol=0
                )
            original_changes.append(change)
        original_probe = old_points, weights, original_tensor
        if index == 0:
            with np.load(source_field) as saved:
                rest, groups, normal = [
                    saved[key].copy() for key in ["rest_surface_positions_m", "groups", "normal_force_by_source_N"]
                ]
            points, records = shared_probes(static.body, rest, groups, normal)
            depths = np.array([item["declared_facet_normal_depth_m"] for item in records])
            source_ids = np.array([item["source"] for item in records])
            anchors = rest[(groups == 0) & np.any(normal != 0, axis=(0, 2))]
            nearest_anchor_distance = cKDTree(anchors).query(points)[0]
            (args.output / "probe_protocol.json").write_text(
                json.dumps(
                    {
                        "records": records,
                        "shared_points_m": points.tolist(),
                        "source": "Every nonzero original baseline body normal-force sample, mixed toward a deterministic incident baseline boundary face centroid before moving inward along that facet normal. No stress peak search defines this set.",
                        "no_filtering_or_relocation_after_construction": True,
                    },
                    indent=2,
                )
                + "\n"
            )
        # Every point must be strictly interior on every mesh. A failed location
        # is a failed diagnostic run, never a silently dropped or moved probe.
        elements = static.body.mesh.element_finder()(*points.T)
        local = static.body.basis.mapping.invF(points.T[:, :, None], tind=elements)[:, :, 0]
        barycentric = np.column_stack((1 - local.sum(axis=0), local.T))
        assert barycentric.min() > 1e-8
        qualification = qualify(static.body, points)
        with np.load(state) as saved:
            vector = np.empty(static.body.basis.N, dtype=np.longdouble)
            vector[static.parts[0].permutation] = saved["displacement"][: static.body.basis.N]
        rotation, _, objective = finite_frame(static.body, np.asarray(vector, dtype=float))
        tensor, vm, _ = point_fields(static.body, vector, points)
        tensor = rotation.T @ tensor @ rotation
        _, _, displacement = point_fields(static.body, objective, points)
        assert float(vm.max()) <= row["raw_peak_upper_Pa"] * (1 + 1e-10) + 1e-5
        result = {
            "run": str(run),
            "tetrahedra": row["tetrahedra"],
            "time_s": row["time_s"],
            "supplemental_probes": len(points),
            "minimum_barycentric_coordinate": float(barycentric.min()),
            "point_evaluator_qualification": qualification,
            "supplemental_sample_vm_maximum_Pa": float(vm.max()),
            "original_raw_global_lower_Pa": row["raw_peak_lower_Pa"],
            "original_raw_global_upper_Pa": row["raw_peak_upper_Pa"],
        }
        rows.append(result)
        if previous is not None:
            for source in [0, 1, 2]:
                for depth in np.unique(depths):
                    selected = (depths == depth) & ((source_ids == source) if source else True)
                    if not selected.any():
                        continue
                    changes.append(
                        {
                            "coarse_tetrahedra": original["rows"][index - 1]["tetrahedra"],
                            "fine_tetrahedra": row["tetrahedra"],
                            "source_zero_means_both": source,
                            "facet_normal_depth_m": float(depth),
                            "probes": int(selected.sum()),
                            "tensor_equal_point_weight_diagnostic": weighted_change(
                                previous[selected][None], tensor[selected][None]
                            ),
                        }
                    )
        previous = tensor
        np.savez_compressed(
            args.output / f"case{index:02d}.npz",
            shared_points_m=points,
            source_ids=source_ids,
            facet_normal_depth_m=depths,
            nearest_baseline_normal_sample_distance_m=nearest_anchor_distance,
            containing_tetrahedra=elements,
            barycentric_coordinates=barycentric,
            objective_tensor_Pa=tensor,
            vm_Pa=vm,
            objective_displacement_m=displacement,
        )
        print(json.dumps(result), flush=True)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "rows": rows,
        "adjacent_depth_changes": changes,
        "original158340_probe_tensor_changes_reproduced_without_replacement": original_changes,
        "supplemental_probes": len(points),
        "supplemental_probes_within25um_of_baseline_normal_samples": int((nearest_anchor_distance < 25e-6).sum()),
        "all_shared_points_strictly_inside_every_mesh_and_evaluator_qualified": True,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Supplemental probes are fixed once using original baseline contact samples and baseline boundary-facet inward normals; all remain at identical material coordinates on every mesh. Nominal depths refer to that baseline polyhedral facet, not the continuum CAD or current deformed interface. Near-interface traces are never filtered or moved. Equal point weights define supplemental diagnostics only, with no volume-integral claim; all original158340 probes, seed-weighted norms and raw whole-volume bounds remain unchanged. Bulk/geometry/contact axes remain coupled. These probes address a measured sampling blind spot without proving physical convergence, singularity, pressure accuracy, full application feedback or warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
