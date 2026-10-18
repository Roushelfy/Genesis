"""Preserve an original interval-only rejection and independently inspect its nearest tetrahedron."""

import argparse
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import TetSampler, build_columns


def determinant(a):
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def decimal_barycentric(vertices, point):
    with localcontext() as context:
        context.prec = 80
        v = [[Decimal.from_float(float(x)) for x in row] for row in vertices]
        p = [Decimal.from_float(float(x)) for x in point]
        matrix = [[v[j + 1][i] - v[0][i] for j in range(3)] for i in range(3)]
        rhs = [p[i] - v[0][i] for i in range(3)]
        denominator = determinant(matrix)
        free = []
        for column in range(3):
            replaced = [[rhs[i] if j == column else matrix[i][j] for j in range(3)] for i in range(3)]
            free.append(determinant(replaced) / denominator)
        return [str(x) for x in [1 - sum(free), *free]]


class GapProbe(TetSampler):
    def intervals(self, inner, outer):
        self.last_line = np.stack((inner, outer))
        return super().intervals(inner, outer)

    def locate_on_column(self, points, fractions, intervals):
        elements, lower, upper = intervals
        included = (fractions[:, None] >= lower[None] - 1e-11) & (fractions[:, None] <= upper[None] + 1e-11)
        missing = np.flatnonzero(~included.any(axis=1))
        if len(missing):
            self.gap_records = []
            for index in missing[:8]:
                point = points[index]
                free = np.einsum("tij,tj->ti", self.inverse, point - self.vertices[self.tetrahedra[:, 0]])
                bary = np.column_stack((1 - free.sum(axis=1), free))
                best = bary.min(axis=1).argmax()
                vertices = self.vertices[self.tetrahedra[best]]
                independent = np.linalg.solve((vertices[1:] - vertices[0]).T, point - vertices[0])
                self.gap_records.append(
                    {
                        "fraction": fractions[index].item(),
                        "query_point_m": point.tolist(),
                        "column_inner_outer_m": self.last_line.tolist(),
                        "best_minimum_barycentric_tet": best.item(),
                        "best_tet_vertices_m": vertices.tolist(),
                        "inverse_barycentric": bary[best].tolist(),
                        "independent_solve_barycentric": np.r_[1 - independent.sum(), independent].tolist(),
                        "decimal80_barycentric_exact_binary_input_coordinates": decimal_barycentric(vertices, point),
                        "best_tet_selected_interval_bounds": [
                            [lo.item(), hi.item()] for lo, hi in zip(lower[elements == best], upper[elements == best])
                        ],
                    }
                )
            raise AssertionError("A common material column has a gap in the FEM domain")
        return super().locate_on_column(points, fractions, intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--surface-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samplers = []
    for cls, folder in ((TetSampler, args.reference), (GapProbe, args.candidate)):
        with np.load(folder / "rest.npz") as saved:
            samplers.append(cls.allocate(saved["authored_vertices_m"], saved["tetrahedra"]))
    with np.load(args.surface_mesh) as saved:
        outer, inner, faces = saved["outer_vertices"], saved["inner_vertices"], saved["mid_faces"]
    failure = None
    try:
        build_columns(*samplers, outer, inner, faces)
    except AssertionError as error:
        failure = str(error)
    records = getattr(samplers[1], "gap_records", [])
    assert failure == "A common material column has a gap in the FEM domain" and records, failure
    write_json(
        args.output / "result.json",
        {
            "kind": "common-column-coverage-failure-diagnostic",
            "identity": runtime_identity(),
            "sources": [file_id(folder / "rest.npz") for folder in (args.reference, args.candidate)]
            + [file_id(args.surface_mesh)],
            "failure": failure,
            "first_failing_column_queries": records,
            "scope": "The original interval-only preselection is deliberately retained by this diagnostic probe, even when the main sampler supports a direct point-query fallback. Independent float64 solve and80-digit Decimal Cramer solve inspect the nearest tetrahedron at the exact binary query coordinates. No tolerance relaxation, coordinate clipping, mesh repair or field score is applied. This diagnoses a query, not solver/reference qualification.",
        },
    )


if __name__ == "__main__":
    main()
