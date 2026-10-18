"""Prepare midpoint surfaces of the existing full normal-offset egg, not a different egg geometry."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from ..mechanics import shell_mesh
from .common import file_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for level in (1, 2, 3):
        vertices, tets, faces, geometry = shell_mesh(level, 2, 0.0005)
        count = geometry.outer_surface_vertices
        midpoint = vertices[count : 2 * count]
        surface = trimesh.Trimesh(midpoint, faces, process=False)
        with (args.output / f"egg-mid-level{level}.obj").open("w") as stream:
            np.savetxt(stream, midpoint, fmt="v %.17g %.17g %.17g")
            np.savetxt(stream, faces + 1, fmt="f %d %d %d")
        volumes = np.abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
        boundary, counts = np.unique(
            np.sort(tets[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1),
            axis=0,
            return_counts=True,
        )
        np.savez_compressed(
            args.output / f"egg-level{level}.npz",
            vertices=vertices,
            tetrahedra=tets,
            mid_vertices=midpoint,
            mid_faces=faces,
            surface_triangles=boundary[counts == 1],
        )
        rows.append(
            {
                "level": level,
                "vertices": count,
                "faces": len(faces),
                "thickness_m": 0.0005,
                "young_pa": 1e10,
                "poisson": 0.3,
                "density_kg_per_m3": 2000,
                "volume_mass_kg": float(volumes.sum() * 2000),
                "mid_surface_lumped_mass_kg": float(surface.area * 0.0005 * 2000),
                "mid_surface_area_m2": float(surface.area),
                "continuum_shape": "22/22/30mm outer semiaxes, taper0.18, normal-offset500um wall",
                "mass_mismatch_relative": float(surface.area * 0.0005 / volumes.sum() - 1),
                "source": file_identity(args.output / f"egg-level{level}.npz"),
                "surface": file_identity(args.output / f"egg-mid-level{level}.obj"),
            }
        )
    write_json(args.output / "result.json", {"kind": "matched-egg-inputs", "levels": rows})


if __name__ == "__main__":
    main()
