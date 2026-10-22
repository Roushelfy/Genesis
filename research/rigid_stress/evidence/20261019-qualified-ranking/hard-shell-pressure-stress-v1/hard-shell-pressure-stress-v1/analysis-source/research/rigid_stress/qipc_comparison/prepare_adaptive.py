"""Refine conforming surface edges around observed contact rays without changing the faceted continuum or contact law."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .prepare_inputs import make_mesh, mass_properties


def split_edges(outer, inner, faces, selected_edges):
    count = len(outer)
    face_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
    edges, inverse = np.unique(face_edges.reshape(-1, 2), axis=0, return_inverse=True)
    selected = np.array([tuple(edge) in selected_edges for edge in edges])
    midpoint_index = np.full(len(edges), -1, dtype=np.int64)
    midpoint_index[selected] = count + np.arange(selected.sum())
    midpoint = (outer + inner) / 2
    outer = np.concatenate((outer, outer[edges[selected]].mean(axis=1)))
    inner = np.concatenate((inner, inner[edges[selected]].mean(axis=1)))
    midpoint = np.concatenate((midpoint, midpoint[edges[selected]].mean(axis=1)))
    children = []
    for triangle, edge_ids in zip(faces, inverse.reshape(-1, 3)):
        middle = midpoint_index[edge_ids]
        active = np.flatnonzero(middle >= 0)
        if len(active) == 0:
            children.append(triangle)
        elif len(active) == 1:
            shift = active[0]
            a, b, c = np.roll(triangle, -shift)
            m = middle[shift]
            children.extend(((a, m, c), (m, b, c)))
        elif len(active) == 2:
            unsplit = np.flatnonzero(middle < 0)[0]
            shift = (unsplit + 1) % 3
            a, b, c = np.roll(triangle, -shift)
            ab, bc = middle[shift], middle[(shift + 1) % 3]
            children.append((ab, b, bc))
            if np.linalg.norm(midpoint[a] - midpoint[bc]) < np.linalg.norm(midpoint[ab] - midpoint[c]):
                children.extend(((a, ab, bc), (a, bc, c)))
            else:
                children.extend(((a, ab, c), (ab, bc, c)))
        else:
            a, b, c = triangle
            ab, bc, ca = middle
            children.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return outer, inner, np.array(children, dtype=np.int64)


def topology(faces, vertices):
    _, multiplicity = np.unique(
        np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True
    )
    assert np.all(multiplicity == 2), "Surface refinement introduces an open or nonmanifold edge"
    edge_count = len(multiplicity)
    assert vertices - edge_count + len(faces) == 2


def split_controls():
    triangle = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    edge_list = [(0, 1), (1, 2), (0, 2)]
    for pattern in range(8):
        active = {edge for index, edge in enumerate(edge_list) if pattern & (1 << index)}
        outer, _, children = split_edges(triangle, triangle - np.array([0, 0, 0.1]), faces, active)
        area_vectors = np.cross(
            outer[children[:, 1]] - outer[children[:, 0]], outer[children[:, 2]] - outer[children[:, 0]]
        )
        assert (area_vectors[:, 2] > 0).all()
        np.testing.assert_allclose(area_vectors.sum(axis=0), [0, 0, 1], rtol=0, atol=2e-15)


def distance_to_triangles(points, triangle):
    edges = triangle[:, 1:] - triangle[:, :1]
    inverse = np.linalg.inv(np.einsum("fai,fbi->fab", edges, edges))
    nearest = np.full(len(triangle), np.inf)
    for point in points:
        free = np.einsum("fij,fj->fi", inverse, np.einsum("fai,fi->fa", edges, point - triangle[:, 0]))
        bary = np.column_stack((1 - free.sum(axis=1), free))
        plane_distance = np.linalg.norm(np.einsum("fi,fij->fj", bary, triangle) - point, axis=1)
        plane_distance[bary.min(axis=1) < 0] = np.inf
        for start, stop in ((0, 1), (1, 2), (2, 0)):
            edge = triangle[:, stop] - triangle[:, start]
            fraction = np.einsum("fi,fi->f", point - triangle[:, start], edge) / (edge * edge).sum(axis=1)
            projected = triangle[:, start] + fraction.clip(0, 1)[:, None] * edge
            plane_distance = np.minimum(plane_distance, np.linalg.norm(projected - point, axis=1))
        nearest = np.minimum(nearest, plane_distance)
    return nearest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base-mesh", type=Path, required=True)
    parser.add_argument("--target-edges", type=float, nargs="+", default=[0.0005, 0.00025])
    parser.add_argument("--padding", type=float, default=0.002)
    parser.add_argument("--force-floor", type=float, default=1e-6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--max-faces", type=int, default=6666)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.padding >= 0 and args.force_floor > 0 and args.layers > 0
    assert args.max_rounds > 0 and all(edge > 0 for edge in args.target_edges)
    args.output.mkdir(parents=True, exist_ok=False)
    split_controls()
    source_inputs = json.loads((args.reference / "inputs.json").read_text())
    source_result = json.loads((args.reference / "result.json").read_text())
    assert source_result["completed_requested_history"]
    with np.load(source_inputs["mesh"]["path"]) as saved:
        source_midpoint = (saved["outer_vertices"] + saved["inner_vertices"]) / 2
    rays, sources = set(), []
    for path in sorted((args.reference / "frames").glob("step*.npz")):
        with np.load(path) as saved:
            active = np.flatnonzero(np.linalg.norm(saved["contact_forces_n"], axis=1) > args.force_floor)
        rays.update((active % len(source_midpoint)).tolist())
        sources.append(file_id(path))
    assert rays
    seed_points = source_midpoint[sorted(rays)]
    with np.load(args.base_mesh) as saved:
        outer_base, inner_base, faces_base = saved["outer_vertices"], saved["inner_vertices"], saved["mid_faces"]
    base_v, base_t, _, base_volume = make_mesh(outer_base, inner_base, faces_base, args.layers)
    base_properties = mass_properties(base_v, base_t, base_volume)
    write_json(
        args.output / "selection.json",
        {
            "reference": file_id(args.reference / "result.json"),
            "base_mesh": file_id(args.base_mesh),
            "source_frames": sources,
            "active_material_rays": sorted(rays),
            "seed_points_m": seed_points.tolist(),
            "padding_m": args.padding,
            "activity_force_floor_n": args.force_floor,
            "scope": "A-posteriori numerical resolution mask from an existing reference history. No prescribed pressure footprint, traction, reconstruction radius, stiffness or contact-law change; contacts remain free everywhere, including coarse regions. Hold this mask fixed for finer comparisons/held-out cases.",
        },
    )
    records = []
    for target in args.target_edges:
        outer, inner, faces = outer_base.copy(), inner_base.copy(), faces_base.copy()
        rounds = []
        converged = False
        for iteration in range(args.max_rounds):
            midpoint = (outer + inner) / 2
            triangle = midpoint[faces]
            edge = triangle[:, [1, 2, 0]] - triangle
            length = np.linalg.norm(edge, axis=2)
            marked = (distance_to_triangles(seed_points, triangle) <= args.padding) & (
                length.max(axis=1) > target * (1 + 1e-12)
            )
            if not marked.any():
                converged = True
                break
            chosen = length.argmax(axis=1)
            selected = {
                tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
                for face, index in zip(faces[marked], chosen[marked])
            }
            outer, inner, faces = split_edges(outer, inner, faces, selected)
            topology(faces, len(outer))
            rounds.append({"round": iteration, "rays": len(outer), "faces": len(faces), "split_edges": len(selected)})
            if len(faces) > args.max_faces:
                break
        np.savez_compressed(
            args.output / f"surface-edge{target:.6g}.npz", outer_vertices=outer, inner_vertices=inner, mid_faces=faces
        )
        record = {"target_edge_m": target, "converged_resolution_mask": converged, "rounds": rounds}
        if converged:
            vertices, tets, boundary, volume = make_mesh(outer, inner, faces, args.layers)
            assert len(boundary) == 2 * len(faces), "Unexpected internal crack in the extruded solid"
            assert np.all((boundary < len(outer)).all(axis=1) | (boundary >= args.layers * len(outer)).all(axis=1))
            properties = mass_properties(vertices, tets, volume)
            np.testing.assert_allclose(properties["mass_kg"], base_properties["mass_kg"], rtol=5e-13, atol=0)
            np.testing.assert_allclose(
                properties["exact_volume_inertia_kg_m2"],
                base_properties["exact_volume_inertia_kg_m2"],
                rtol=2e-12,
                atol=1e-20,
            )
            destination = args.output / f"adaptive-edge{target:.6g}-wall{args.layers}.npz"
            np.savez_compressed(
                destination,
                vertices=vertices,
                tetrahedra=tets,
                surface_triangles=boundary,
                outer_vertices=outer,
                inner_vertices=inner,
                mid_vertices=(outer + inner) / 2,
                mid_faces=faces,
                volumes_m3=volume,
            )
            record.update(
                mesh=file_id(destination), vertices=len(vertices), tetrahedra=len(tets), properties=properties
            )
        else:
            record["failure"] = (
                "Resolution target not reached within the declared face/round resource bounds; no solid reference mesh admitted"
            )
        records.append(record)
    write_json(
        args.output / "result.json",
        {
            "kind": "conforming-fixed-domain-adaptive-contact-inputs",
            "identity": runtime_identity(),
            "base_properties": base_properties,
            "records": records,
            "split_pattern_area_orientation_controls_passed": True,
            "reference_qualified": False,
        },
    )


if __name__ == "__main__":
    main()
