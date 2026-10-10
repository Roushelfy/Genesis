"""Compare discrete traction feasibility for the actual apex contact."""

import json

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from research.rigid_stress.mechanics import P2Shell, SurfaceGeometry, shell_mesh
from research.rigid_stress.wrench import FinitePatchMapper, WrenchFit, WrenchPatch


def main():
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    shell = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    mapper = FinitePatchMapper(SurfaceGeometry(shell, 10), anchor_to_surface=True)
    patch = WrenchPatch(
        np.array([4.0606709579473816e-5, 1.3926954540923124e-5, 0.0299207658748014]),
        np.array([0.004688181349230815, 0.0023709449777518836, -0.0015963625737377865]),
        0.005998121586384792,
        1.0,
        np.zeros(3),
        np.array([0.45061547141681174, 0.1545483800796497, -0.8792385882879348]),
    )
    anchor = mapper.surface_anchor(patch.center_m, patch.force_n)
    ids = np.array(mapper.tree.query_ball_point(anchor, patch.radius_m, return_sorted=True))
    distance = np.sum((mapper.positions[ids] - anchor) ** 2, axis=1) / patch.radius_m**2
    weights = np.exp(-0.5 * distance / 0.45**2) * np.maximum(0, 1 - distance) ** 2 * mapper.weights[ids]
    direction = patch.force_n / np.linalg.norm(patch.force_n)
    tangent = np.cross(direction, np.eye(3)[np.argmin(abs(direction))])
    tangent /= np.linalg.norm(tangent)
    basis = np.stack((tangent, np.cross(direction, tangent)))
    transverse = (mapper.positions[ids] - anchor) @ basis.T
    operator = np.column_stack((np.ones(len(ids)), transverse / patch.radius_m))
    feasibility = linprog(np.zeros(len(ids)), A_eq=operator.T, b_eq=[1, 0, 0], bounds=(0, None), method="highs")
    hull = ConvexHull(transverse)
    normal = patch.inward_normal / np.linalg.norm(patch.inward_normal)
    compression = normal @ patch.force_n
    report = {
        "samples": len(ids),
        "anchor": anchor.tolist(),
        "original_position": patch.center_m.tolist(),
        "constant_ratio_lp_status": feasibility.status,
        "hull_outside_m": float(np.max(hull.equations[:, -1])),
        "tangent_to_normal_ratio": float(np.linalg.norm(patch.force_n - compression * normal) / compression),
    }
    for name, normals in (("pad", np.broadcast_to(normal, (len(ids), 3))), ("surface", mapper.normals[ids])):
        fit = WrenchFit(mapper.positions[ids], weights, normals, patch)
        try:
            force, diagnostic = fit.solve()
            report[name] = vars(diagnostic)
            report[name]["evaluated_residual"] = np.linalg.norm(
                np.einsum("qij,qj->i", fit.operator, force) - fit.target
            )
        except ValueError as error:
            report[name] = {"error": str(error)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
