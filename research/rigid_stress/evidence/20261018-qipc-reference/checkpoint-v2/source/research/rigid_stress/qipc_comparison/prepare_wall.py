"""Derive shell face masses and thicknesses from the unchanged three-dimensional wall volume."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .prepare_inputs import make_mesh, mass_properties


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for level, wall in ((0, 2), (1, 2), (2, 4)):
        path = args.inputs / f"base1-surface{level}-wall{wall}.npz"
        with np.load(path) as saved:
            outer, inner, faces = saved["outer_vertices"], saved["inner_vertices"], saved["mid_faces"]
        midpoint = (outer + inner) / 2
        vertices, tets, _, volume = make_mesh(outer, inner, faces, 1)
        face_volume = volume.reshape(3, len(faces)).sum(axis=0)
        edges = midpoint[faces[:, 1:]] - midpoint[faces[:, :1]]
        area = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1) / 2
        thickness = face_volume / area
        face_mass = face_volume * 2000
        vertex_mass = np.bincount(faces.reshape(-1), weights=np.repeat(face_mass / 3, 3), minlength=len(midpoint))
        properties = mass_properties(vertices, tets, volume)
        np.testing.assert_allclose(vertex_mass.sum(), properties["mass_kg"], rtol=2e-13, atol=0)
        center = vertex_mass @ midpoint / vertex_mass.sum()
        relative = midpoint - center
        second = np.einsum("v,vi,vj->ij", vertex_mass, relative, relative)
        inertia = np.eye(3) * np.trace(second) - second
        exact_inertia = np.array(properties["exact_volume_inertia_kg_m2"])
        output = args.output / f"surface{level}.npz"
        np.savez_compressed(
            output,
            mid_vertices_m=midpoint,
            mid_faces=faces,
            face_volume_m3=face_volume,
            face_mass_kg=face_mass,
            face_thickness_m=thickness,
            vertex_mass_kg=vertex_mass,
        )
        records.append(
            {
                "level": level,
                "source": file_id(path),
                "input": file_id(output),
                "mass_kg": vertex_mass.sum().item(),
                "nominal_500um_mass_kg": (area.sum() * 2000 * 0.0005).item(),
                "minimum_volume_average_thickness_m": thickness.min().item(),
                "maximum_volume_average_thickness_m": thickness.max().item(),
                "exact_solid_com_m": properties["com_m"],
                "shell_lumped_com_m": center.tolist(),
                "shell_lumped_inertia_kg_m2": inertia.tolist(),
                "exact_solid_inertia_kg_m2": exact_inertia.tolist(),
                "lumped_inertia_relative_difference": (
                    np.linalg.norm(inertia - exact_inertia) / np.linalg.norm(exact_inertia)
                ).item(),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "volume-consistent-shell-rest-input",
            "identity": runtime_identity(),
            "records": records,
            "scope": "Geometry-only face wall volumes divided by original midpoint areas. No density change, fitted stiffness/radius or reference field. This matches rest mass; shell director geometry, omitted thickness inertia and vertex/contact representation remain model approximations. Preserve nominal500um stock as a separate baseline.",
        },
    )


if __name__ == "__main__":
    main()
