"""Generate the declared centroid pad quadrature and its work-conjugate Shell face map."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.mesh) as saved:
        outer, midpoint, faces = saved["outer_vertices"], saved["mid_vertices"], saved["mid_faces"]
    corners = midpoint[faces]
    outer_corners = outer[faces]
    points = outer_corners.mean(axis=1)
    offset = (outer_corners - corners).mean(axis=1)
    # Form the small face lever arms before averaging. Subtracting a world
    # centroid loses digits when refined faces are small relative to the body;
    # the inverse rotational Gram matrix then amplifies the translation defect.
    local = corners - corners[:, :1]
    relative = local - local.mean(axis=1, keepdims=True)
    relative -= relative.mean(axis=1, keepdims=True)
    gram = np.eye(3) * (relative**2).sum(axis=(1, 2))[:, None, None] - np.einsum("fki,fkj->fij", relative, relative)
    inverse = np.linalg.inv(gram)
    jacobians = np.empty((len(faces), 3, 3, 3))
    for axis, force in enumerate(np.eye(3)):
        couple = np.cross(offset, force)
        multiplier = np.einsum("fij,fj->fi", inverse, couple)
        jacobians[:, :, axis] = force[None, None] / 3 + np.cross(multiplier[:, None], relative)
    np.testing.assert_allclose(jacobians.sum(axis=1), np.broadcast_to(np.eye(3), (len(faces), 3, 3)), atol=1e-13)
    for rotation in np.eye(3):
        mapped = np.einsum("fkij,fkj->fi", jacobians, np.cross(rotation, corners))
        np.testing.assert_allclose(mapped, np.cross(rotation, points), atol=1e-14)
    rng = np.random.default_rng(727824)
    delta = rng.normal(size=corners.shape)
    point_force = rng.normal(size=points.shape)
    point_delta = np.einsum("fkij,fkj->fi", jacobians, delta)
    corner_force = np.einsum("fkij,fi->fkj", jacobians, point_force)
    np.testing.assert_allclose(np.sum(point_force * point_delta), np.sum(corner_force * delta), atol=1e-10)
    areas = (
        np.abs(np.cross(outer_corners[:, 1] - outer_corners[:, 0], outer_corners[:, 2] - outer_corners[:, 0])[:, 2]) / 2
    )
    path = args.output / "pad-geometry.npz"
    np.savez_compressed(path, corners_m=corners, points_m=points, jacobians=jacobians, projected_area_m2=areas)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "geometry": file_id(path),
            "quadrature": "One centroid point per original outer face with its projected reference area.",
            "rigid_motion_and_virtual_work_controls_passed": True,
            "projected_area_m2": float(areas.sum()),
            "maximum_corner_jacobian_frobenius": float(np.linalg.norm(jacobians, axis=(-2, -1)).max()),
            "scope": "Fixed linear displacement map and transpose force transfer for the declared small-deformation physical pad variant. It preserves rigid motion and force/couple work; spatial/centroid quadrature convergence and actual continuous coupled equations require separate checks.",
        },
    )


if __name__ == "__main__":
    main()
