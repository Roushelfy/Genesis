"""Verify the physical support and consistent nodal forces of a retained P2 contact solution."""

import argparse
from pathlib import Path

import numpy as np
from skfem import Basis, ElementTetP2, MeshTet

from .artifacts import file_id, runtime_identity, write_json
from .p2_elastic import quadratic_triangle_minimum


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--field", type=Path, required=True)
    cli.add_argument("--load", type=float, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.mesh) as saved:
        vertices, tetrahedra, faces = saved["vertices"], saved["tetrahedra"], saved["mid_faces"]
    mesh = MeshTet(vertices.T, tetrahedra.T, sort_t=False)
    basis = Basis(mesh, ElementTetP2(), intorder=2)
    lookup = {tuple(edge): node for edge, node in zip(mesh.edges.T, basis.edge_dofs[0])}
    mid = np.array([[lookup[tuple(sorted((a, b)))] for a, b in zip(face, np.roll(face, -1))] for face in faces])
    with np.load(args.field) as saved:
        coordinates = saved["vertices_m"]
        np.testing.assert_array_equal(coordinates, basis.doflocs.T)
        outer = saved["outer_nodes"]
        nodes = outer[saved["candidate_nodes"]]
        interpolation = saved["contact_constraint_interpolation"]
        multiplier = saved["contact_compressive_multipliers"]
        point = interpolation @ coordinates[nodes]
        np.testing.assert_allclose(interpolation.sum(axis=1), 1, rtol=0, atol=1e-13)
        sign = np.where(coordinates[nodes, 2] > 0, -1.0, 1.0)
        point_sign = interpolation @ sign
        np.testing.assert_allclose(np.abs(point_sign), 1, rtol=0, atol=1e-13)
        forces = multiplier[:, None] * args.load * point_sign[:, None] * np.array([0.0, 0.0, 1.0])
        nodal = np.zeros_like(coordinates)
        nodal[nodes] = interpolation.T @ forces
        np.testing.assert_allclose(nodal, saved["contact_force_n"], rtol=1e-10, atol=1e-14 * args.load)
        moment_difference = np.cross(point, forces).sum(axis=0) - np.cross(coordinates, nodal).sum(axis=0)
        assert np.linalg.norm(moment_difference) < 1e-12 * args.load
        quadratic = np.searchsorted(outer, np.column_stack((faces, mid)))
        triangle = vertices[faces]
        area = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1) / 2
        rows = []
        for upper in (False, True):
            values = saved["top_gap_m" if upper else "bottom_gap_m"][quadratic]
            minimum, bary = quadratic_triangle_minimum(values, return_points=True)
            # Independent P2 evaluation must reproduce the reported extremum.
            weights = np.column_stack(
                (
                    bary * (2 * bary - 1),
                    4 * bary[:, 0] * bary[:, 1],
                    4 * bary[:, 1] * bary[:, 2],
                    4 * bary[:, 2] * bary[:, 0],
                )
            )
            np.testing.assert_allclose(np.sum(weights * values, axis=1), minimum, atol=1e-16, rtol=1e-8)
            maximum = -quadratic_triangle_minimum(-values)
            on_side = (triangle[:, :, 2].mean(axis=1) > 0) == upper
            closed = (maximum <= 1e-7 * args.load / (1e10 * 5e-5)) & on_side
            carried = ((point_sign < 0) == upper) & (multiplier > 0)
            rows.append(
                {
                    "upper": upper,
                    "minimum_whole_surface_gap_m": float(minimum.min()),
                    "maximum_constraint_radius_m": float(np.linalg.norm(point[carried, :2], axis=1).max()),
                    "compressive_constraint_points": int(carried.sum()),
                    "normal_reaction_n": float(np.linalg.norm(forces[carried].sum(axis=0))),
                    "fully_closed_quadratic_face_area_m2": float(area[closed].sum()),
                }
            )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "field": file_id(args.field),
            "footprints": rows,
            "force_transfer_verified": True,
            "moment_transfer_error_nm": moment_difference.tolist(),
            "physical_reference_qualified": False,
            "scope": "Nonnegative point multipliers, not signs of consistent P2 nodal forces, define load support. Fully closed face area uses the exact quadratic maximum gap; it does not assert continuous pressure support. Full-surface gap minima are independent of candidate nodes.",
        },
    )


if __name__ == "__main__":
    main()
