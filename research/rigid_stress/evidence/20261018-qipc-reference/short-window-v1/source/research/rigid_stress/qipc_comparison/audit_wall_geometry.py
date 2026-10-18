"""Quantify shell input mass, collision-envelope and normal second-moment mismatches geometrically."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json
from .prepare_inputs import make_mesh


def normal_second_moment(outer, inner, faces):
    midpoint = (outer + inner) / 2
    edge = midpoint[faces[:, 1:]] - midpoint[faces[:, :1]]
    cross = np.cross(edge[:, 0], edge[:, 1])
    area = np.linalg.norm(cross, axis=1) / 2
    normal = cross / (2 * area[:, None])
    vertices, tets, _, volume = make_mesh(outer, inner, faces, 1)
    face = np.tile(np.arange(len(faces)), 3)
    z = np.einsum("tvi,ti->tv", vertices[tets] - midpoint[faces].mean(axis=1)[face, None], normal[face])
    integrated = volume * ((z * z).sum(axis=1) + z.sum(axis=1) ** 2) / 20
    return np.bincount(face, weights=integrated, minlength=len(faces)) / area, area, normal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solid-inputs", type=Path, required=True)
    parser.add_argument("--wall-inputs", type=Path, required=True)
    parser.add_argument("--shell-layouts", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    h = 0.0005
    control, _, _ = normal_second_moment(triangle + [0, 0, h / 2], triangle - [0, 0, h / 2], np.array([[0, 1, 2]]))
    np.testing.assert_allclose(control, h**3 / 12, rtol=2e-14, atol=0)
    with np.load(args.solid_inputs / "base1-surface2-wall4.npz") as saved:
        outer, inner, faces = saved["outer_vertices"], saved["inner_vertices"], saved["mid_faces"]
    _, _, normal = normal_second_moment(outer, inner, faces)
    directors = outer - inner
    directors /= np.linalg.norm(directors, axis=1)[:, None]
    directions = np.unique(np.concatenate((normal, directors, np.eye(3), -np.eye(3))), axis=0)
    records, data = [], {"support_directions": directions}
    for level, wall in ((0, 2), (1, 2), (2, 4)):
        source = args.solid_inputs / f"base1-surface{level}-wall{wall}.npz"
        supplied = args.wall_inputs / f"surface{level}.npz"
        with np.load(source) as saved:
            outer, inner, faces = saved["outer_vertices"], saved["inner_vertices"], saved["mid_faces"]
        actual_moment, area, _ = normal_second_moment(outer, inner, faces)
        midpoint = (outer + inner) / 2
        layout_path = args.shell_layouts[level]
        with np.load(layout_path) as saved:
            actual_vertices, actual_faces = saved["vertices_m"], saved["faces"]
            contact_faces = saved["collision_vertex_face"]
            h_volume = saved["face_thickness_m"]
            softness = saved["rigid_geometry_coupling_softness_m"]
        distance, order = cKDTree(midpoint[faces].mean(axis=1)).query(actual_vertices[actual_faces].mean(axis=1))
        assert distance.max() < 1e-12 and len(np.unique(order)) == len(faces)
        area, actual_moment = area[order], actual_moment[order]
        solid_support = (outer @ directions.T).max(axis=0)
        for variant, thickness in (("nominal500um", np.full(len(faces), 0.0005)), ("wall-volume", h_volume)):
            # The original coupler selects the first fan face per vertex.
            # Keep SDF softness/influence cutoff separate from this nominal
            # half-thickness support; neither is a penetration guarantee.
            shell_support = (actual_vertices @ directions.T + thickness[contact_faces, None] / 2).max(axis=0)
            gap = shell_support - solid_support
            moment = thickness**3 / 12
            relative = (moment - actual_moment) / actual_moment
            name = f"surface{level}_{variant}"
            data[f"{name}_support_gap_m"] = gap
            data[f"{name}_normal_second_moment_relative_difference"] = relative
            records.append(
                {
                    "level": level,
                    "variant": variant,
                    "solid_source": file_id(source),
                    "wall_input": file_id(supplied),
                    "actual_shell_layout": file_id(layout_path),
                    "maximum_geometry_influence_cutoff_padding_m": (softness.max() * np.log(10)).item(),
                    "minimum_sampled_support_gap_m": gap.min().item(),
                    "maximum_sampled_support_gap_m": gap.max().item(),
                    "maximum_absolute_sampled_support_gap_m": np.abs(gap).max().item(),
                    "normal_second_moment_area_weighted_relative_difference": (
                        np.linalg.norm(np.sqrt(area) * (moment - actual_moment))
                        / np.linalg.norm(np.sqrt(area) * actual_moment)
                    ).item(),
                    "minimum_face_normal_second_moment_relative_difference": relative.min().item(),
                    "maximum_face_normal_second_moment_relative_difference": relative.max().item(),
                }
            )
    np.savez_compressed(args.output / "geometry.npz", **data)
    write_json(
        args.output / "result.json",
        {
            "kind": "shell-wall-collision-and-bending-geometry-audit",
            "identity": runtime_identity(),
            "parallel_plate_second_moment_control_passed": True,
            "sampled_support_directions": len(directions),
            "records": records,
            "scope": "Pure rest geometry, no force/stress fitting or density scaling. Actual PR22 topology selects its FIRST fan face thickness at each physical vertex. Solid support is the fixed outer polyhedron; nominal Shell half-thickness support is sampled on common face normals, wall directors and axes. SDF softness adds a separately recorded influence padding at the0.1 cutoff; the velocity-projection law is not an IPC barrier and no penetration guarantee is implied. This is not an exact all-direction Hausdorff bound or replacement collider. Exact tet normal second moments versus h^3/12 quantify geometry, not complete nonlinear bending error. Matched mass does not imply matched contact or bending geometry.",
        },
    )


if __name__ == "__main__":
    main()
