"""Resolve case banks and shell contact meshes on the fixed reference geometry without radius fitting."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank_path = args.inputs / "conditions.npz"
    with np.load(bank_path) as bank:
        cases = []
        for case in range(len(bank["position"])):
            path = args.output / f"case{case}.npz"
            np.savez_compressed(path, **{key: bank[key][case : case + 1] for key in bank.files})
            cases.append(
                {
                    "case": case,
                    "bank": file_id(path),
                    "friction_coefficient": 0.6 * bank["friction"][case, 0].item(),
                    "radius_m": bank["radius"][case].item(),
                    "delay_ticks": bank["delays"][case].item(),
                    "position_m": bank["position"][case].tolist(),
                    "quaternion_wxyz": bank["quaternion"][case].tolist(),
                }
            )
    surfaces = []
    for refinement, wall in ((0, 2), (1, 2), (2, 4)):
        mesh_path = args.inputs / f"base1-surface{refinement}-wall{wall}.npz"
        with np.load(mesh_path) as mesh:
            midpoint, faces = mesh["mid_vertices"], mesh["mid_faces"]
        directory = args.output / f"surface{refinement}"
        directory.mkdir()
        path = directory / "egg-mid-level1.obj"
        with path.open("w") as stream:
            np.savetxt(stream, midpoint, fmt="v %.17g %.17g %.17g")
            np.savetxt(stream, faces + 1, fmt="f %d %d %d")
        edges = midpoint[faces[:, 1:]] - midpoint[faces[:, :1]]
        area = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1).sum() / 2
        surfaces.append(
            {
                "refinement": refinement,
                "vertices": len(midpoint),
                "faces": len(faces),
                "source": file_id(mesh_path),
                "surface": file_id(path),
                "nominal_thickness_shell_mass_kg": area * 0.0005 * 2000,
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "common-fixed-geometry-case-inputs",
            "bank": file_id(bank_path),
            "cases": cases,
            "surfaces": surfaces,
            "contract": "Shell surfaces subdivide the same level1 piecewise-planar geometry. No re-rounding or radius/reference footprint fitting; retain nominal shell thickness and disclose its area-thickness mass/inertia approximation.",
        },
    )


if __name__ == "__main__":
    main()
