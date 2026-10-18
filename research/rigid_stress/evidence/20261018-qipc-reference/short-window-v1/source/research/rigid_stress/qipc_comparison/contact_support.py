"""Measure discrete recorded contact-force supports without identifying them as continuum pressure footprints or local slip."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import complete_control_grid, requested_duration_s
from .reference_uncertainty import frame_index


def boundary_faces(tets):
    faces = np.concatenate([tets[:, ids] for ids in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))])
    faces, count = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    assert np.all((count == 1) | (count == 2))
    return faces[count == 1]


def nodal_areas(points, faces):
    triangle = points[faces]
    area = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1) / 2
    weights = np.bincount(faces.ravel(), weights=np.repeat(area / 3, 3), minlength=len(points))
    np.testing.assert_allclose(weights.sum(), area.sum(), rtol=2e-15, atol=0)
    return weights


def support(points, forces):
    magnitude = np.linalg.norm(forces, axis=1)
    assert np.all(magnitude > 0)
    center = magnitude @ points / magnitude.sum()
    rms = np.sqrt(magnitude @ np.sum((points - center) ** 2, axis=1) / magnitude.sum())
    return {
        "force_magnitude_weighted_center_material_m": center.tolist(),
        "force_magnitude_weighted_radius_rms_m": float(rms),
        "maximum_pairwise_node_distance_m": float(pdist(points).max()) if len(points) > 1 else 0.0,
        "force_resultant_material_n": forces.sum(axis=0).tolist(),
        "moment_about_material_origin_nm": np.cross(points, forces).sum(axis=0).tolist(),
        "sum_nodal_force_magnitudes_n": float(magnitude.sum()),
    }


def analytic_controls():
    points = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    forces = np.array([[0.0, 1, 0], [0, -1, 0]])
    result = support(points, forces)
    np.testing.assert_array_equal(result["force_resultant_material_n"], [0, 0, 0])
    np.testing.assert_array_equal(result["moment_about_material_origin_nm"], [0, 0, -2])
    np.testing.assert_array_equal(result["force_magnitude_weighted_center_material_m"], [0, 0, 0])
    assert result["force_magnitude_weighted_radius_rms_m"] == 1
    assert result["maximum_pairwise_node_distance_m"] == 2
    triangle = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(nodal_areas(triangle, np.array([[0, 1, 2]])), np.full(3, 1 / 6), rtol=1e-15)
    return {
        "opposing_forces_preserve_nonzero_moment": True,
        "known_center_radius_diameter": True,
        "lumped_triangle_area": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--force-floor", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.force_floor == 1e-12, "Keep the recorded packet's declared support threshold"
    args.output.mkdir(parents=True, exist_ok=False)
    controls = analytic_controls()
    reference = json.loads((args.reference / "result.json").read_text())
    packet_receipt = json.loads((args.packets / "result.json").read_text())
    assert reference["completed_requested_history"]
    assert packet_receipt["reference"] == file_id(args.reference / "result.json")
    frames = frame_index(args.reference)
    assert complete_control_grid(frames, requested_duration_s(reference))
    with np.load(args.reference / "rest.npz") as saved:
        authored, tets = saved["authored_vertices_m"], saved["tetrahedra"]
    faces = boundary_faces(tets)
    reference_areas = nodal_areas(authored, faces)
    pairs = np.concatenate([tets[:, [a, b]] for a in range(4) for b in range(a + 1, 4)])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    adjacency = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(authored),) * 2).tocsr()
    adjacency = adjacency + adjacency.T
    records, components, sources = [], [], []
    for tick, frame in sorted(frames.items()):
        packet = args.packets / "frames" / frame.name
        with np.load(packet) as saved:
            points, force = saved["physical_contact_positions_material_m"], saved["contact_forces_material_n"]
            time, radius = saved["time_s"].item(), saved["radius_m"].item()
            active = np.flatnonzero(np.linalg.norm(force, axis=1) > args.force_floor)
            count, labels = (
                connected_components(adjacency[active][:, active], directed=False)
                if len(active)
                else (0, np.empty(0, dtype=np.int64))
            )
            selected = saved["selected_nodes"]
            assert np.array_equal(labels[np.searchsorted(active, selected)], saved["support_components"])
        assert abs(time - tick * 0.01) < 1e-12
        assert np.all(reference_areas[active] > 0), "Recorded force on a non-boundary node"
        current_areas = nodal_areas(points, faces)
        radii, diameters = [], []
        for group in range(count):
            nodes = active[labels == group]
            item = support(points[nodes], force[nodes])
            item.update(
                time_s=time,
                component=group,
                force_nodes=len(nodes),
                reference_lumped_support_area_m2=float(reference_areas[nodes].sum()),
                current_lumped_support_area_m2=float(current_areas[nodes].sum()),
                preexisting_reconstruction_radius_m=radius,
                model_radius_over_observed_support_rms=radius / item["force_magnitude_weighted_radius_rms_m"]
                if item["force_magnitude_weighted_radius_rms_m"] > 0
                else None,
            )
            components.append(item)
            radii.append(item["force_magnitude_weighted_radius_rms_m"])
            diameters.append(item["maximum_pairwise_node_distance_m"])
        records.append(
            {
                "time_s": time,
                "active_force_nodes": len(active),
                "connected_force_supports": count,
                "reference_lumped_support_area_m2": float(reference_areas[active].sum()),
                "current_lumped_support_area_m2": float(current_areas[active].sum()),
                "maximum_support_rms_m": max(radii, default=0.0),
                "maximum_support_diameter_m": max(diameters, default=0.0),
                "preexisting_reconstruction_radius_m": radius,
            }
        )
        sources.append(file_id(packet))
    with (args.output / "support-history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "components": len(components),
        "single_node_components": sum(row["force_nodes"] == 1 for row in components),
        "observations_without_force": sum(row["active_force_nodes"] == 0 for row in records),
    }
    for key in (
        "active_force_nodes",
        "connected_force_supports",
        "reference_lumped_support_area_m2",
        "current_lumped_support_area_m2",
        "maximum_support_rms_m",
        "maximum_support_diameter_m",
    ):
        row = max(records, key=lambda row: row[key])
        summary[f"maximum_{key}"] = {"value": row[key], "time_s": row["time_s"]}
    write_json(
        args.output / "result.json",
        {
            "kind": "discrete-recorded-contact-force-support-audit",
            "identity": runtime_identity(),
            "reference": file_id(args.reference / "result.json"),
            "packet_receipt": file_id(args.packets / "result.json"),
            "analytic_controls": controls,
            "force_threshold_n": args.force_floor,
            "observations": len(records),
            "summary": summary,
            "complete_reference_grid": True,
            "history": records,
            "components": components,
            "source_packets": sources,
            "reference_qualified": False,
            "scope": "Actual recorded total nodal contact forces and deformed material positions, with mesh-connected active supports at the original1e-12N packet threshold. Reference/current areas sum boundary face-area/3 weights at active nodes; these are discrete support measures, not continuum pressure areas. Components can combine contacts; no opposing body or pure normal/friction split is inferred. The old reconstruction radius is compared descriptively and never fitted. Local relative tangential velocity, frozen friction state and per-primitive friction gradients are absent from these packets, so this audit cannot certify local stick/slip or Coulomb saturation. Macroscopic hand-relative COM drift is reported separately.",
        },
    )


if __name__ == "__main__":
    main()
