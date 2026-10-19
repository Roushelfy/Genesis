"""Independent wrench, work, scale and mass-projection controls for free static loads."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import runtime_identity, write_json
from .static_inertia_relief import StaticInertiaRelief


def wrench(points, force):
    return np.r_[force.sum(axis=0), np.cross(points, force).sum(axis=0)]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(623017)
    vertices = rng.normal(size=(24, 3)) * 0.03
    masses = rng.uniform(0.0001, 0.0005, len(vertices))
    loads = rng.normal(size=vertices.shape)
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rows = []
    for geometry_scale in (0.1, 1, 10):
        points = vertices @ rotation.T * geometry_scale + np.array([0.2, -0.4, 0.3])
        relief = StaticInertiaRelief.build(points, masses)
        for force_scale in (1e-6, 1, 1e6):
            force = loads @ rotation.T * force_scale
            corrected, correction = relief.apply(force)
            # Independent resultant and moment in world coordinates.
            defect = wrench(points, corrected)
            force_bound = np.linalg.norm(force, axis=1).sum()
            arm = np.linalg.norm(points - points.mean(axis=0), axis=1).max()
            relative = max(np.linalg.norm(defect[:3]) / force_bound, np.linalg.norm(defect[3:]) / (force_bound * arm))
            assert relative < 1e-12
            repeated, second_correction = relief.apply(corrected)
            np.testing.assert_allclose(repeated, corrected, rtol=0, atol=1e-13 * force_scale)
            assert np.linalg.norm(second_correction) <= 1e-13 * np.linalg.norm(force)
            virtual = np.tile([0.3, -0.5, 0.7], (len(points), 1)) + np.cross([0.4, 0.2, -0.3], points)
            assert abs(np.sum(corrected * virtual)) <= 1e-12 * force_bound
            # A declared rigid acceleration has exactly its mass-distributed
            # body load removed; arbitrary balanced elastic content remains.
            acceleration = np.array([0.2, -0.4, 0.3, 1.2, -0.8, 0.5]) * force_scale
            rigid_load = (relief.mass_modes @ acceleration).reshape(-1, 3)
            remaining, known = relief.apply(rigid_load)
            assert np.linalg.norm(remaining) <= 1e-13 * np.linalg.norm(rigid_load)
            np.testing.assert_allclose(known, rigid_load, rtol=1e-12, atol=1e-15 * force_scale)
            rows.append(
                {
                    "geometry_scale": geometry_scale,
                    "force_scale": force_scale,
                    "independent_wrench_relative_defect": float(relative),
                    "correction_norm_n": float(np.linalg.norm(correction)),
                    "passed": True,
                }
            )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": True,
            "cases": rows,
            "scope": "Independent complete-load rigid-wrench, arbitrary rigid-work, exact rigid-acceleration and idempotence controls across rotations/translations and force/geometry scales. Does not admit arbitrary unbalanced physical pad loads; the Shell worker separately limits its correction to the existing reference budget.",
        },
    )


if __name__ == "__main__":
    main()
