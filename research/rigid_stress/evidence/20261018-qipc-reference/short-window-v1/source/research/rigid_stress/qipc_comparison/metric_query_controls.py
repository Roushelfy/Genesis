"""Validate an actual rejected material point, genuine outside rejection and immutable legacy query arrays."""

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import TetSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-gap", type=Path, required=True)
    parser.add_argument("--old-metric", type=Path, required=True)
    parser.add_argument("--new-metric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved_gap = json.loads(args.recorded_gap.read_text())
    rest_path = Path(saved_gap["sources"][1]["path"])
    with np.load(rest_path) as saved:
        sampler = TetSampler.allocate(saved["authored_vertices_m"], saved["tetrahedra"])
    query_records = []
    for record in saved_gap["first_failing_column_queries"]:
        assert min(Decimal(value) for value in record["decimal80_barycentric_exact_binary_input_coordinates"]) >= 0
        point = np.array([record["query_point_m"]])
        fraction = np.array([record["fraction"]])
        inner, outer = np.array(record["column_inner_outer_m"])
        intervals = sampler.intervals(inner, outer)
        elements, barycentric = sampler.locate_on_column(point, fraction, intervals)
        tetrahedron = sampler.vertices[sampler.tetrahedra[elements[0]]]
        independent = np.linalg.solve((tetrahedron[1:] - tetrahedron[0]).T, point[0] - tetrahedron[0])
        np.testing.assert_allclose(barycentric[0], np.r_[1 - independent.sum(), independent], rtol=0, atol=2e-12)
        np.testing.assert_allclose(barycentric @ tetrahedron, point, rtol=0, atol=2e-16)
        query_records.append(
            {"query": record, "selected_tetrahedron": elements[0].item(), "barycentric": barycentric[0].tolist()}
        )
    assert sampler.direct_point_query_fallbacks == len(query_records)

    unit = TetSampler.allocate(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 1, 2, 3]]))
    outside_rejected = False
    try:
        unit.locate_on_column(
            np.array([[2.0, 2, 2]]), np.array([0.5]), (np.array([], dtype=int), np.array([]), np.array([]))
        )
    except AssertionError:
        outside_rejected = True
    assert outside_rejected and unit.direct_point_query_fallbacks == 0
    # Cell selection follows the inside point while interpolation keeps an
    # exact boundary coordinate; no coordinate clipping or stress smoothing.
    boundary = np.array([[1.0, 0, 0]])
    cells, weights = unit.locate_on_column(
        boundary,
        np.array([1 - 1e-8]),
        (np.array([], dtype=int), np.array([]), np.array([])),
        selection_points=boundary * (1 - 1e-8),
    )
    np.testing.assert_array_equal(weights, [[0.0, 1, 0, 0]])
    np.testing.assert_array_equal(cells, [0])
    adjacent = TetSampler.allocate(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]),
        np.array([[0, 1, 2, 3], [0, 4, 2, 3]]),
    )
    face = np.array([[0.0, 0.2, 0.2], [0.0, 0.2, 0.2]])
    cells, weights = adjacent.locate_on_column(
        face,
        np.array([0.5, 0.5]),
        (np.array([], dtype=int), np.array([]), np.array([])),
        selection_points=face + np.array([[1e-8, 0, 0], [-1e-8, 0, 0]]),
    )
    np.testing.assert_array_equal(cells, [0, 1])
    recovered = np.einsum("qi,qij->qj", weights, adjacent.vertices[adjacent.tetrahedra[cells]])
    np.testing.assert_allclose(recovered, face, rtol=0, atol=2e-16)

    checked_arrays = []
    for filename in ("columns.npz", "surfaces.npz"):
        with np.load(args.old_metric / filename) as old, np.load(args.new_metric / filename) as new:
            assert set(old.files) == set(new.files)
            for key in old.files:
                assert old[key].dtype == new[key].dtype
                np.testing.assert_array_equal(old[key], new[key])
                checked_arrays.append(f"{filename}:{key}")
    new_receipt = json.loads((args.new_metric / "result.json").read_text())
    assert new_receipt["analytic_constant_tensor_linear_bending_affine_interpolation_rotation_checks_passed"]
    assert new_receipt["direct_point_query_fallbacks"] == [0, 0]
    write_json(
        args.output / "result.json",
        {
            "kind": "common-material-direct-query-and-legacy-regression-controls",
            "identity": runtime_identity(),
            "sources": [
                file_id(args.recorded_gap),
                file_id(rest_path),
                file_id(args.old_metric / "result.json"),
                file_id(args.new_metric / "result.json"),
            ],
            "actual_recorded_inside_queries": query_records,
            "actual_inside_query_fallbacks": sampler.direct_point_query_fallbacks,
            "genuine_outside_query_rejected": outside_rejected,
            "one_sided_selection_keeps_exact_boundary_interpolation": True,
            "opposite_interior_points_select_distinct_cells_at_same_exact_boundary": True,
            "legacy_arrays_exactly_equal": checked_arrays,
            "all_controls_passed": True,
            "scope": "Existing2e-9 barycentric tolerance, coordinates, quadrature weights and one-sided boundary contract are unchanged. These are geometry/observation controls; no physical reference, PCG convergence or whole-history field score is qualified.",
        },
    )


if __name__ == "__main__":
    main()
