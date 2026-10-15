"""Retain contact-template roles, density hotspots and matched source-wrench changes across covered CAD levels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .summarize_fine_references import closed_source


def relative_change(first, second):
    return abs(first / second - 1) if second != 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.source_audit)
    assert int((args.source_audit / "returncode.txt").read_text()) == 0
    summary_path = args.source_audit / "results/summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["selector_mesh_reconstruction_qualified"]
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {str(summary_path): hashlib.sha256(summary_path.read_bytes()).hexdigest()}
    producers = list(dict.fromkeys(row["run"] for row in summary["rows"]))
    rows, comparisons = [], []
    previous_by_step = {}
    for i_case, run in enumerate(producers):
        for original in [row for row in summary["rows"] if row["run"] == run]:
            field_path = args.source_audit / f"results/case{i_case:02d}-step{original['step']:04d}.npz"
            inputs[str(field_path)] = hashlib.sha256(field_path.read_bytes()).hexdigest()
            with np.load(field_path) as saved:
                groups, area, current, rest, faces = [
                    saved[key].copy()
                    for key in [
                        "groups",
                        "current_vertex_lumped_area_m2",
                        "current_surface_positions_m",
                        "rest_surface_positions_m",
                        "surface_triangles",
                    ]
                ]
                normal, shear = [saved[key].copy() for key in ["normal_force_by_source_N", "shear_force_by_source_N"]]
            source_rows = []
            face_vectors = rest[faces]
            face_normals = np.cross(face_vectors[:, 1] - face_vectors[:, 0], face_vectors[:, 2] - face_vectors[:, 0])
            face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]
            for source in original["sources"]:
                source_id = source["source"]
                templates = {}
                finger_edges = {}
                for stencil in original["stencils"]:
                    if stencil["source"] != source_id:
                        continue
                    nodes = np.asarray(stencil["surface_nodes"])
                    body_count = int((groups[nodes] == 0).sum())
                    key = f"{stencil['type']}:body_support_nodes={body_count}"
                    record = templates.setdefault(
                        key, {"count": 0, "stored_weights": [], "unsigned_body_resultant_N": 0.0}
                    )
                    record["count"] += 1
                    record["stored_weights"].append(stencil["stored_weight"])
                    force = np.asarray(stencil["normal_stencil_force_N"])
                    record["unsigned_body_resultant_N"] += float(np.linalg.norm(force[groups[nodes] == 0].sum(axis=0)))
                    finger_nodes = np.sort(nodes[groups[nodes] == source_id])
                    if len(finger_nodes) == 2:
                        edge_key = tuple(int(value) for value in finger_nodes)
                        if edge_key not in finger_edges:
                            incident = np.flatnonzero(np.isin(faces, finger_nodes).sum(axis=1) == 2)
                            assert len(incident) > 0 and np.all(groups[faces[incident]] == source_id)
                            angle = None
                            if len(incident) == 2:
                                cosine = abs(float(face_normals[incident[0]] @ face_normals[incident[1]]))
                                angle = float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
                            finger_edges[edge_key] = {
                                "surface_nodes": list(edge_key),
                                "reference_endpoints_m": np.asarray(rest[finger_nodes], dtype=float).tolist(),
                                "reference_edge_length_m": float(
                                    np.linalg.norm(rest[finger_nodes[1]] - rest[finger_nodes[0]])
                                ),
                                "incident_surface_face_ids": incident.tolist(),
                                "unsigned_angle_between_incident_face_planes_deg": angle,
                                "normal_stencils": 0,
                                "unsigned_body_resultant_N": 0.0,
                            }
                        finger_edges[edge_key]["normal_stencils"] += 1
                        finger_edges[edge_key]["unsigned_body_resultant_N"] += float(
                            np.linalg.norm(force[groups[nodes] == 0].sum(axis=0))
                        )
                for record in templates.values():
                    record["stored_weight_minimum_median_maximum"] = np.percentile(
                        record["stored_weights"], [0, 50, 100]
                    ).tolist()
                hotspots = {}
                for label, force in [("normal", normal[source_id - 1]), ("shear", shear[source_id - 1])]:
                    for surface, group in [("body", 0), ("finger", source_id)]:
                        selected = np.flatnonzero(groups == group)
                        magnitudes = np.linalg.norm(force[selected], axis=1)
                        densities = magnitudes / area[selected]
                        i_local = int(np.argmax(densities))
                        i_vertex = int(selected[i_local])
                        hotspot = {
                            "surface_vertex": i_vertex,
                            "maximum_discrete_density_Pa": float(densities[i_local]),
                            "force_norm_N": float(magnitudes[i_local]),
                            "force_vector_N": force[i_vertex].tolist(),
                            "lumped_area_m2": float(area[i_vertex]),
                            "reference_position_m": np.asarray(rest[i_vertex], dtype=float).tolist(),
                            "current_position_m": np.asarray(current[i_vertex], dtype=float).tolist(),
                        }
                        original_density = source[f"{label}_{surface}_density"][
                            "maximum_sample_force_over_lumped_area_Pa"
                        ]
                        np.testing.assert_allclose(
                            hotspot["maximum_discrete_density_Pa"], original_density, rtol=1e-12, atol=0
                        )
                        hotspots[f"{label}_{surface}"] = hotspot
                source_rows.append(
                    {
                        "source": source_id,
                        "captured_finger_link": source["captured_finger_link"],
                        "normal_template_roles": templates,
                        "two_node_finger_support_edges": list(finger_edges.values()),
                        "density_hotspots_all_nodes": hotspots,
                    }
                )
            row = {
                "run": run,
                "step": original["step"],
                "time_s": original["time_s"],
                "body_tetrahedra": original["body_tetrahedra"],
                "sources": source_rows,
            }
            rows.append(row)
            old = previous_by_step.get(original["step"])
            if old is not None:
                changes = []
                for first, second in zip(old["sources"], original["sources"], strict=True):
                    assert (
                        first["source"] == second["source"]
                        and first["captured_finger_link"] == second["captured_finger_link"]
                    )
                    changes.append(
                        {
                            "source": second["source"],
                            "body_force_difference_N": (
                                np.asarray(second["total_body_wrench"]["force_N"])
                                - first["total_body_wrench"]["force_N"]
                            ).tolist(),
                            "body_moment_difference_about_reference_origin_Nm": (
                                np.asarray(second["total_body_wrench"]["moment_about_reference_origin_Nm"])
                                - first["total_body_wrench"]["moment_about_reference_origin_Nm"]
                            ).tolist(),
                            "body_force_reference_norm_N": float(
                                np.linalg.norm(second["total_body_wrench"]["force_N"])
                            ),
                            "body_force_relative_L2_change": (
                                float(
                                    np.linalg.norm(
                                        np.asarray(second["total_body_wrench"]["force_N"])
                                        - first["total_body_wrench"]["force_N"]
                                    )
                                    / np.linalg.norm(second["total_body_wrench"]["force_N"])
                                )
                                if np.linalg.norm(second["total_body_wrench"]["force_N"]) > 1e-12
                                else None
                            ),
                            "body_normal_density_L2_relative_change": relative_change(
                                first["normal_body_density"]["sample_force_density_L2_N_per_m"],
                                second["normal_body_density"]["sample_force_density_L2_N_per_m"],
                            ),
                            "body_normal_maximum_density_relative_change": relative_change(
                                first["normal_body_density"]["maximum_sample_force_over_lumped_area_Pa"],
                                second["normal_body_density"]["maximum_sample_force_over_lumped_area_Pa"],
                            ),
                            "body_normal_active_lumped_area_relative_change": relative_change(
                                first["normal_body_density"]["active_lumped_area_m2"],
                                second["normal_body_density"]["active_lumped_area_m2"],
                            ),
                        }
                    )
                comparisons.append(
                    {
                        "step": original["step"],
                        "coarse_tetrahedra": old["body_tetrahedra"],
                        "fine_tetrahedra": original["body_tetrahedra"],
                        "sources": changes,
                    }
                )
            previous_by_step[original["step"]] = original
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    result = {
        "scope": __doc__,
        "rows": rows,
        "comparisons": comparisons,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Every original force, threshold tail and component remains in the qualified parent arrays. All nodes participate in density hotspots. Template support counts and actual stored weights distinguish body/finger vertex and edge roles; reaction interpolation on coarse rigid triangles can produce different lumped-density diagnostics without identifying a continuous pressure field. Force/moment differences are retained in physical units; the nonzero source-force norm normalizes its vector change. These diagnostics alone establish neither contact-pressure convergence nor a causal/additive split of bulk/geometry/contact/model errors.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
