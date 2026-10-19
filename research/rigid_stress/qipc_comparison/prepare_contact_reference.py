"""Estimate polar contact scales and build conforming locally refined smooth-egg solid meshes."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from research.rigid_stress.validation_meshes import transform_shell

from .artifacts import file_id, runtime_identity, write_json
from .prepare_inputs import make_mesh, mass_properties


def refine_contact_surface(minimum_size, global_size, refinement_radius=0.0):
    sphere = trimesh.creation.icosphere(subdivisions=1)
    unit, faces = sphere.vertices.copy(), sphere.faces.copy()
    for _ in range(24):
        outer = transform_shell("egg", unit)
        edges, inverse = np.unique(
            np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_inverse=True
        )
        midpoint = unit[edges].sum(axis=1)
        midpoint /= np.linalg.norm(midpoint, axis=1)[:, None]
        mapped = transform_shell("egg", midpoint)
        polar_distance = np.linalg.norm(mapped[:, :2], axis=1)
        target = np.minimum(
            global_size, np.maximum(minimum_size, 0.2 * np.maximum(polar_distance - refinement_radius, 0))
        )
        selected = np.linalg.norm(outer[edges[:, 0]] - outer[edges[:, 1]], axis=1) > target
        if not selected.any():
            return unit, faces
        indices = np.full(len(edges), -1)
        indices[selected] = len(unit) + np.arange(selected.sum())
        unit = np.concatenate((unit, midpoint[selected]))
        mids = indices[inverse.reshape(-1, 3)]
        triangles = []
        for face, mid in zip(faces, mids):
            count = (mid >= 0).sum()
            if count == 0:
                triangles.append(face)
            elif count == 3:
                a, b, c = face
                ab, bc, ca = mid
                triangles.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
            else:
                rotation = np.flatnonzero(mid >= 0)[0] if count == 1 else (np.flatnonzero(mid < 0)[0] + 1) % 3
                a, b, c = np.roll(face, -rotation)
                ab, bc, _ = np.roll(mid, -rotation)
                if count == 1:
                    triangles.extend(((a, ab, c), (ab, b, c)))
                else:
                    triangles.extend(((a, ab, c), (ab, bc, c), (ab, b, bc)))
        faces = np.array(triangles)
    raise RuntimeError("Contact surface refinement did not reach its declared size field")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cases", nargs="+", default=["5e-5:0.002:4", "2.5e-5:0.0015:6", "1.25e-5:0.001:8"])
    cli.add_argument("--wall-depths", type=float, nargs="+")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    estimates, meshes = [], []
    for pole in (-1, 1):
        radius = (0.022 * (1 - 0.18 * pole)) ** 2 / 0.030
        for load in (0.1, 0.3, 1.0):
            contact_radius = (3 * load * radius * (1 - 0.3**2) / (4 * 1e10)) ** (1 / 3)
            estimates.append(
                {
                    "pole": pole,
                    "load_n": load,
                    "curvature_radius_m": radius,
                    "hertz_radius_m": contact_radius,
                    "hertz_local_indentation_m": contact_radius**2 / radius,
                    "hertz_maximum_pressure_pa": 3 * load / (2 * np.pi * contact_radius**2),
                    "radius_over_wall": contact_radius / 0.0005,
                }
            )
    for i_m, specification in enumerate(args.cases):
        minimum, global_size, layers = specification.split(":")
        unit, faces = refine_contact_surface(float(minimum), float(global_size))
        outer = transform_shell("egg", unit)
        inner = transform_shell("egg", 0.92 * unit)
        depths = np.linspace(0, 1, int(layers) + 1) if args.wall_depths is None else np.asarray(args.wall_depths)
        vertices, tetrahedra, boundary, volumes = make_mesh(outer, inner, faces, int(layers), depths)
        boundary_edges, counts = np.unique(
            np.sort(boundary[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True
        )
        assert (counts == 2).all()
        edges = outer[faces[:, [1, 2, 0]]] - outer[faces]
        lengths = np.linalg.norm(edges, axis=2)
        cosines = -np.einsum("fij,fij->fi", edges, edges[:, [2, 0, 1]]) / (lengths * lengths[:, [2, 0, 1]])
        path = args.output / f"smooth-contact-{i_m}.npz"
        np.savez_compressed(
            path,
            vertices=vertices,
            tetrahedra=tetrahedra,
            surface_triangles=boundary,
            outer_vertices=outer,
            inner_vertices=inner,
            mid_vertices=(outer + inner) / 2,
            mid_faces=faces,
            wall_depths=depths,
        )
        polar_distance = np.linalg.norm(outer[:, :2], axis=1)
        meshes.append(
            {
                "input": file_id(path),
                "specification": specification,
                "wall_depths": depths.tolist(),
                "outer_vertices": len(outer),
                "surface_faces": len(faces),
                "tetrahedra": len(tetrahedra),
                "closed_boundary_edges": len(boundary_edges),
                "minimum_surface_angle_deg": np.degrees(np.arccos(np.clip(cosines, -1, 1))).min().item(),
                "minimum_volume_m3": volumes.min().item(),
                "vertices_in_100um_polar_circles": int((polar_distance < 1e-4).sum()),
                **mass_properties(vertices, tetrahedra, volumes),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "contact_scale_estimates": estimates,
            "meshes": meshes,
            "source": "https://ocw.mit.edu/courses/2-76-multi-scale-system-design-fall-2004/a529c6243da095dfd8b9fb80cc421521_lec12cont_read.pdf",
            "scope": "Hertz half-space estimates size local refinement only; they do not qualify hollow-shell compliance, friction or stress. Every new surface vertex is projected onto the same smooth tapered egg and its 500um normal offset. Full model; no symmetry reduction. Mesh generation alone is not convergence evidence.",
        },
    )


if __name__ == "__main__":
    main()
