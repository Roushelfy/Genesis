"""Certify infeasibility using a circumscribed polygonal outer bound of surface Coulomb cones."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from research.rigid_stress.mechanics import P2Shell, SurfaceGeometry, shell_mesh
from research.rigid_stress.wrench import FinitePatchMapper, WrenchFit, WrenchPatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    shell = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    mapper = FinitePatchMapper(SurfaceGeometry(shell, 10), anchor_to_surface=True)
    rows = []
    for case in json.loads(args.fixture.read_text())["cases"]:
        patch = WrenchPatch(
            np.array(case["position"]),
            np.array(case["force"]),
            case["radius"],
            case["friction"],
            np.zeros(3),
            np.array(case["normal"]),
        )
        anchor = mapper.surface_anchor(patch.center_m, patch.force_n)
        ids = np.array(mapper.tree.query_ball_point(anchor, patch.radius_m, return_sorted=True))
        normals = mapper.normals[ids]
        first = np.cross(normals, np.eye(3)[np.argmin(abs(normals), axis=1)])
        first /= np.linalg.norm(first, axis=1)[:, None]
        second = np.cross(normals, first)
        angles = 2 * np.pi * np.arange(32) / 32
        directions = first[:, None] * np.cos(angles)[None, :, None] + second[:, None] * np.sin(angles)[None, :, None]
        inequalities = np.concatenate((directions - patch.friction * normals[:, None], -normals[:, None]), axis=1)
        cone = sparse.block_diag(inequalities, format="csr")
        fit = WrenchFit(mapper.positions[ids], np.ones(len(ids)), normals, patch)
        force_only = linprog(
            np.zeros(3 * len(ids)),
            A_ub=cone,
            b_ub=np.zeros(cone.shape[0]),
            A_eq=fit.operator[:, :3].transpose(1, 0, 2).reshape(3, -1),
            b_eq=patch.force_n / np.linalg.norm(patch.force_n),
            bounds=(None, None),
        )
        wrench = linprog(
            np.zeros(3 * len(ids)),
            A_ub=cone,
            b_ub=np.zeros(cone.shape[0]),
            A_eq=fit.operator.transpose(1, 0, 2).reshape(6, -1),
            b_eq=fit.target / np.linalg.norm(patch.force_n),
            bounds=(None, None),
        )
        rows.append(
            {
                "env": case["env"],
                "samples": len(ids),
                "outer_cone_planes": 33,
                "force_only_lp_status": force_only.status,
                "full_wrench_lp_status": wrench.status,
                "interpretation": "Infeasible outer approximation certifies infeasibility of the contained circular cones. A feasible outer approximation leaves circular-cone feasibility undecided.",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
