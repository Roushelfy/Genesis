"""Freeze the shared tangent-field/resultant sampling rule and validate exact polynomial column integrals."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import TetSampler, build_columns, corotation, resultants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-mesh", type=Path, required=True)
    parser.add_argument("--native-mesh", type=Path, required=True)
    parser.add_argument("--surface-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samplers = []
    for path in (args.reference_mesh, args.native_mesh):
        with np.load(path) as mesh:
            samplers.append(TetSampler.allocate(mesh["vertices"], mesh["tetrahedra"]))
    with np.load(args.surface_mesh) as surface:
        columns = build_columns(*samplers, surface["outer_vertices"], surface["inner_vertices"], surface["mid_faces"])
    constant = np.array([[3.0, 2, 1], [2, -4, 5], [1, 5, 7]])
    gradient = constant * 1234
    field = constant + columns.z[:, None, None] * gradient
    membrane, bending = resultants(field, columns)
    np.testing.assert_allclose(membrane, columns.thickness[:, None, None] * constant, rtol=1e-12, atol=1e-17)
    np.testing.assert_allclose(bending, columns.thickness[:, None, None] ** 3 / 12 * gradient, rtol=1e-11, atol=1e-19)
    for sampler, elements, barycentric in zip(
        samplers,
        (columns.reference_elements, columns.native_elements),
        (columns.reference_barycentric, columns.native_barycentric),
    ):
        recovered = np.einsum("qi,qij->qj", barycentric, sampler.vertices[sampler.tetrahedra[elements]])
        np.testing.assert_allclose(recovered, columns.points, rtol=0, atol=2e-16)
    authored = samplers[0].vertices
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    recovered = corotation(authored, authored @ rotation.T + np.array([0.6, 0.02, 0.1]), np.ones(len(authored)))
    np.testing.assert_allclose(recovered, rotation, rtol=0, atol=2e-14)
    np.savez_compressed(args.output / "columns.npz", **asdict(columns))
    write_json(
        args.output / "result.json",
        {
            "kind": "shared-material-column-metric",
            "identity": runtime_identity(),
            "inputs": [file_id(path) for path in (args.reference_mesh, args.native_mesh, args.surface_mesh)],
            "columns": len(columns.area),
            "quadrature_points": len(columns.points),
            "minimum_projected_wall_m": columns.thickness.min().item(),
            "maximum_projected_wall_m": columns.thickness.max().item(),
            "weighted_domain_measure_m3": columns.weights.sum().item(),
            "analytic_constant_tensor_linear_bending_affine_interpolation_rotation_checks_passed": True,
            "metric": "Area-weighted material face-centroid columns; tangent Cauchy tensors in the corotated authored frame. Two Gauss points on every union P1/P2 tet intersection segment. This is one common three-way metric domain, not an exact full-volume integral or global-extremum search.",
        },
    )


if __name__ == "__main__":
    main()
