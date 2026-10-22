"""Tessellate the captured curved outer material surface for matched collision controls."""

import argparse
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .study_quadratic_rest_surface import quadratic_basis


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--material", type=Path, required=True)
    cli.add_argument("--original-urdf", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.frame), file_id(args.material), file_id(args.original_urdf)],
            "scope": "Geometry preprocessing only. Tessellate the explicitly selected outer quadratic material faces at1/2/4/8 divisions; preserve original URDF mass, inertia and frame. Weld by shared parent topology, verify a closed triangular surface and measure interpolation/normal errors against the same quadratic charts. Collision and stress admission require a separate native run.",
        },
    )
    with np.load(args.frame) as frame:
        vertices = frame["material_nodes_m"]
        material_faces = frame["material_surface_nodes"].T
        outward = frame["material_face_normal"]
    with np.load(args.material) as material:
        outer = material["surface_triangles"]
        np.testing.assert_array_equal(vertices[: len(material["vertices"])], material["vertices"])
    assert material_faces.shape[1] == 6 and outer.shape[1] == 3
    selected = {tuple(sorted(face[:3])): i for i, face in enumerate(material_faces)}
    ids = np.array([selected[tuple(sorted(face))] for face in outer])
    assert len(np.unique(ids)) == len(ids)
    material_faces, outward = material_faces[ids], outward[ids]
    nodes = vertices[material_faces]
    rows = []
    for divisions in [1, 2, 4, 8]:
        coordinates = [(i, j) for i in range(divisions + 1) for j in range(divisions + 1 - i)]
        local_ids = {value: i for i, value in enumerate(coordinates)}
        bary = np.array([[divisions - i - j, i, j] for i, j in coordinates]) / divisions
        values, _, _ = quadratic_basis(bary)
        points = np.einsum("qn,fni->fqi", values, nodes)
        registry, merged, maps = {}, [], np.empty(points.shape[:2], dtype=np.int64)
        for i_f, corners in enumerate(material_faces[:, :3]):
            for i_q, (i, j) in enumerate(coordinates):
                key = tuple(
                    sorted(
                        (int(vertex), weight) for vertex, weight in zip(corners, [divisions - i - j, i, j]) if weight
                    )
                )
                if key not in registry:
                    registry[key] = len(merged)
                    merged.append(points[i_f, i_q])
                index = registry[key]
                assert np.linalg.norm(merged[index] - points[i_f, i_q]) < 1e-15
                maps[i_f, i_q] = index
        triangles = []
        for i, j in coordinates:
            if i + j < divisions:
                triangles.append([local_ids[i, j], local_ids[i + 1, j], local_ids[i, j + 1]])
                if i + j < divisions - 1:
                    triangles.append([local_ids[i + 1, j], local_ids[i + 1, j + 1], local_ids[i, j + 1]])
        triangles = np.array(triangles)
        assert len(triangles) == divisions**2
        output_vertices = np.array(merged)
        output_faces = maps[:, triangles].reshape(-1, 3)
        area = np.cross(
            points[:, triangles[:, 1]] - points[:, triangles[:, 0]],
            points[:, triangles[:, 2]] - points[:, triangles[:, 0]],
        )
        is_reversed = np.einsum("fqi,fi->fq", area, outward).reshape(-1) < 0
        output_faces[is_reversed] = output_faces[is_reversed][:, [0, 2, 1]]
        edges, counts = np.unique(
            np.sort(output_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True
        )
        assert np.all(counts == 2) and len(output_vertices) - len(edges) + len(output_faces) == 2
        # Child vertices are exact chart samples. Check independent interior and
        # edge-midpoint samples where flat collision triangles depart from it.
        sample = np.array([[1 / 3] * 3, [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        child_bary = np.einsum("sa,tai->tsi", sample, bary[triangles]).reshape(-1, 3)
        values, gradient, _ = quadratic_basis(child_bary)
        curved = np.einsum("qn,fni->fqi", values, nodes).reshape(len(nodes), len(triangles), len(sample), 3)
        flat = np.einsum("sa,ftai->ftsi", sample, points[:, triangles])
        error = np.linalg.norm(curved - flat, axis=3)
        tangent = np.einsum("qna,fni->fqia", gradient, nodes).reshape(len(nodes), len(triangles), len(sample), 3, 2)
        normal = np.cross(tangent[..., 0], tangent[..., 1])
        normal /= np.linalg.norm(normal, axis=3)[..., None]
        flat_normal = area / np.linalg.norm(area, axis=2)[..., None]
        degrees = np.rad2deg(np.arccos(np.clip(np.abs(np.einsum("ftsi,fti->fts", normal, flat_normal)), 0, 1)))
        root = args.output / f"division{divisions}"
        root.mkdir()
        mesh = root / "egg_outer.obj"
        with mesh.open("x") as stream:
            for point in output_vertices:
                stream.write("v " + " ".join(f"{x:.17g}" for x in point) + "\n")
            for face in output_faces:
                stream.write("f " + " ".join(str(index + 1) for index in face) + "\n")
        tree = ElementTree.parse(args.original_urdf)
        old_inertial = ElementTree.tostring(tree.find("link/inertial"))
        for item in tree.iter("mesh"):
            item.set("filename", mesh.name)
        assert ElementTree.tostring(tree.find("link/inertial")) == old_inertial
        urdf = root / "egg_shell.urdf"
        tree.write(urdf, encoding="utf-8", xml_declaration=True)
        fields = root / "mesh.npz"
        np.savez_compressed(
            fields,
            vertices=output_vertices,
            triangles=output_faces,
            parent_faces=np.repeat(ids, len(triangles)),
            point_error_m=error,
            normal_error_degrees=degrees,
        )
        rows.append(
            {
                "divisions": divisions,
                "vertices": len(output_vertices),
                "triangles": len(output_faces),
                "maximum_sampled_position_error_m": float(error.max()),
                "rms_sampled_position_error_m": float(np.sqrt(np.mean(error**2))),
                "maximum_sampled_normal_error_degrees": float(degrees.max()),
                "mesh": file_id(mesh),
                "urdf": file_id(urdf),
                "fields": file_id(fields),
            }
        )
    assert all(
        rows[i + 1]["maximum_sampled_position_error_m"] < rows[i]["maximum_sampled_position_error_m"] / 3.9
        for i in range(3)
    )
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
