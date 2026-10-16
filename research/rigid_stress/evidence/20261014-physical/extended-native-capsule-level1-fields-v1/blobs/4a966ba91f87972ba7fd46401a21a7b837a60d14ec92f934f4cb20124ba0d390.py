"""Verify saved-corner and physical-probe conventions using independent manufactured P2 fields."""

import argparse
import json
from pathlib import Path

import numpy as np
from skfem import Basis, ElementTetP2, ElementVector, MeshTet
from skfem.helpers import sym_grad
from skfem.models.elasticity import lame_parameters, linear_stress

from .sample_load_fields import common_physical_probes, locate_probes, sample_field


def bending_displacement(points, curvature, poisson):
    x, y, z = points.T
    return curvature * np.column_stack((-x * y, 0.5 * (x * x + poisson * (y * y - z * z)), poisson * y * z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    young, poisson, curvature = 1e10, 0.3, 0.1
    stress = linear_stress(*lame_parameters(young, poisson))
    strain = np.array([[2e-6, 0.7e-6, 0.2e-6], [0.7e-6, -1e-6, -0.4e-6], [0.2e-6, -0.4e-6, 3e-6]])
    corners = np.column_stack((np.zeros(3), np.eye(3)))
    rows = []
    for mesh_path in sorted(args.runtime_root.glob("matrix-mesh-*-v1/meshes/*.npz")):
        geometry = json.loads(mesh_path.with_suffix(".json").read_text())["geometry"]["name"]
        with np.load(mesh_path) as saved:
            mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
        basis = Basis(mesh, ElementVector(ElementTetP2()), intorder=4)
        corner_basis = Basis(mesh, basis.elem, quadrature=(corners, np.ones(4)))
        components = np.stack(basis.split_indices(), axis=-1)
        coordinates = basis.doflocs.T[components[:, 0]]
        points = common_physical_probes(geometry)
        elements, selected, barycentric = locate_probes(mesh, points)
        assert selected.all(), (geometry, np.flatnonzero(~selected))
        for name in ("uniform", "bending"):
            vector = np.empty(basis.N)
            expected = np.zeros((len(points), 3, 3))
            if name == "uniform":
                vector[components] = coordinates @ strain
                expected[:] = 2 * lame_parameters(young, poisson)[1] * strain + lame_parameters(young, poisson)[
                    0
                ] * np.trace(strain) * np.eye(3)
                expected_u = points @ strain
            else:
                vector[components] = bending_displacement(coordinates, curvature, poisson)
                expected[:, 0, 0] = -young * curvature * points[:, 1]
                expected_u = bending_displacement(points, curvature, poisson)
            tensor = stress(sym_grad(corner_basis.interpolate(vector))).transpose(2, 3, 0, 1)
            sampled, _, sampled_u = sample_field(
                basis, elements, selected, barycentric, points, {"tensor_Pa": tensor, "displacement": vector}
            )
            np.testing.assert_allclose(sampled, expected, rtol=1e-10, atol=1e-5)
            np.testing.assert_allclose(sampled_u, expected_u, rtol=1e-10, atol=1e-14)
            row = {
                "geometry": geometry,
                "mesh": str(mesh_path),
                "field": name,
                "maximum_tensor_difference_Pa": float(abs(sampled - expected).max()),
                "maximum_displacement_difference_m": float(abs(sampled_u - expected_u).max()),
            }
            if name == "bending":
                wrong, _, _ = sample_field(
                    basis,
                    elements,
                    selected,
                    barycentric,
                    points,
                    {"tensor_Pa": tensor[:, [0, 3, 2, 1]], "displacement": vector},
                )
                row["wrong_corner_order_maximum_difference_Pa"] = float(abs(wrong - expected).max())
                assert row["wrong_corner_order_maximum_difference_Pa"] > 1e-5
            rows.append(row)
    assert len(rows) == 42
    (args.output / "summary.json").write_text(
        json.dumps(
            {"scope": "Exact manufactured P2 physical-probe reconstruction on 21 full3D assets", "rows": rows}, indent=2
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "maximum_tensor_difference_Pa": max(row["maximum_tensor_difference_Pa"] for row in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
