"""Check contact-centred face integration under the declared pad friction law."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from research.rigid_stress.mechanics import P2Shell, SurfaceGeometry, shell_mesh
from research.rigid_stress.wrench import FinitePatchMapper, PadPressureFit, WrenchPatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    shell = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    geometry = SurfaceGeometry(shell, 10)
    mapper = FinitePatchMapper(geometry, anchor_to_surface=True)
    gauss, gauss_weights = np.polynomial.legendre.leggauss(10)
    gauss, gauss_weights = (gauss + 1) / 2, gauss_weights / 2
    u, v = np.meshgrid(gauss, gauss, indexing="ij")
    bary = np.stack((1 - u, u * (1 - v), u * v), axis=-1).reshape(-1, 3)
    quadrature_weights = (gauss_weights[:, None] * gauss_weights[None, :] * u).reshape(-1)
    result = []
    for row in json.loads(args.fixture.read_text())["cases"]:
        patch = WrenchPatch(
            np.array(row["position"]),
            np.array(row["force"]),
            row["radius"],
            row["friction"],
            np.zeros(3),
            np.array(row["normal"]),
        )
        anchor = mapper.surface_anchor(patch.center_m, patch.force_n)
        face_bary = np.einsum("fai,fi->fa", mapper.face_dual, anchor - mapper.face_vertices[:, 0])
        plane_distance = np.einsum("fi,fi->f", mapper.face_normals, anchor - mapper.face_vertices[:, 0])
        eligible = (
            (np.abs(plane_distance) < 1e-12) & (face_bary.min(axis=1) >= -2e-12) & (face_bary.sum(axis=1) <= 1 + 2e-12)
        )
        face_id = np.flatnonzero(eligible)[0]
        nodes = mapper.face_vertices[face_id]
        positions = mapper.positions.copy()
        weights = mapper.weights.copy()
        keep = np.arange(len(positions)) // 100 != face_id
        positions, weights = positions[keep], weights[keep]
        anchor_bary = np.r_[1 - face_bary[face_id].sum(), face_bary[face_id]]
        inner = anchor + anchor_bary[:, None] * (nodes - anchor)
        triangles = [inner]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            triangles.append(np.stack((nodes[first], nodes[second], inner[second])))
            triangles.append(np.stack((nodes[first], inner[second], inner[first])))
        for triangle in triangles:
            area = np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
            positions = np.concatenate((positions, bary @ triangle))
            weights = np.concatenate((weights, area * quadrature_weights))
        distance = np.sum((positions - anchor) ** 2, axis=1) / patch.radius_m**2
        weights *= np.exp(-0.5 * distance / 0.45**2) * np.maximum(0, 1 - distance) ** 2
        keep = (distance < 1) & (weights > 0)
        positions, weights = positions[keep], weights[keep]
        direction = patch.force_n / np.linalg.norm(patch.force_n)
        first = np.cross(direction, np.eye(3)[np.argmin(abs(direction))])
        first /= np.linalg.norm(first)
        coordinates = np.column_stack(
            (
                np.ones(len(positions)),
                (positions - anchor) @ np.stack((first, np.cross(direction, first))).T / patch.radius_m,
            )
        )
        feasibility = linprog(np.zeros(len(weights)), A_eq=coordinates.T, b_eq=[1, 0, 0], bounds=(0, None))
        diagnostic = {"env": row["env"], "face": int(face_id), "samples": len(weights), "lp_status": feasibility.status}
        anchored = WrenchPatch(anchor, patch.force_n, patch.radius_m, patch.friction, np.zeros(3), patch.inward_normal)
        try:
            _force, errors = PadPressureFit(positions, weights, anchored).solve()
            diagnostic.update(vars(errors))
        except ValueError as error:
            diagnostic["error"] = str(error)
        result.append(diagnostic)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
