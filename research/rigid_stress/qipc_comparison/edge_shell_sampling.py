"""Partition the complete shell into graded volume and material-column queries."""

import argparse
import json
from pathlib import Path

import numpy as np
from skfem.quadrature import get_quadrature_tet

from .artifacts import file_id, runtime_identity, write_json


def partition(cells, width, along, minimum, companions=()):
    """Longest-edge bisection preserves the exact original material partition."""
    pairs = np.array([(i, j) for i in range(cells.shape[1]) for j in range(i + 1, cells.shape[1])])
    kept, extra, history = [], [[] for _ in companions], []
    for level in range(36):
        center = cells.mean(axis=1)
        radius = np.linalg.norm(cells - center[:, None], axis=-1).max(axis=1)
        dx = np.maximum.reduce([-center[:, 0], center[:, 0] - width, np.zeros(len(cells))])
        dy = np.maximum(np.abs(center[:, 1]) - along, 0)
        distance = np.sqrt(dx * dx + dy * dy + (0.03 - center[:, 2]) ** 2)
        lower = np.maximum(distance - radius, 0)
        lengths = np.linalg.norm(cells[:, pairs[:, 1]] - cells[:, pairs[:, 0]], axis=-1)
        edge = lengths.argmax(axis=1)
        diameter = lengths[np.arange(len(cells)), edge]
        refine = (lower < 8 * along) & (diameter > 2 * np.maximum(lower, minimum))
        history.append({"level": level, "cells": len(cells), "split": int(refine.sum())})
        kept.append(cells[~refine])
        for saved, companion in zip(extra, companions):
            saved.append(companion[~refine])
        if not refine.any():
            break
        selected = cells[refine]
        edges = pairs[edge[refine]]
        index = np.arange(len(selected))

        def children(values):
            first, second = values.copy(), values.copy()
            middle = (values[index, edges[:, 0]] + values[index, edges[:, 1]]) / 2
            first[index, edges[:, 0]], second[index, edges[:, 1]] = middle, middle
            return np.concatenate((first, second))

        cells = children(selected)
        companions = tuple(children(companion[refine]) for companion in companions)
        assert len(cells) + sum(len(value) for value in kept) < 500000, history
    else:
        raise RuntimeError("Contact integration partition did not reach the declared scale.")
    return np.concatenate(kept), [np.concatenate(saved) for saved in extra], history


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh-study", type=Path, required=True)
    cli.add_argument("--pressure", type=Path, required=True)
    cli.add_argument("--minimum-over-width", type=float, required=True)
    cli.add_argument("--depth-order", type=int, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.pressure / "start.json").read_text())
    width, along = start["candidate"]["width"], start["candidate"]["along"]
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.pressure / "start.json")]
            + [file_id(args.mesh_study / (name + "-mesh.npz")) for name in ["coarse", "medium", "fine"]],
            "minimum_over_width": args.minimum_over_width,
            "depth_order": args.depth_order,
            "scope": "Exact original-domain partition; contact-focused longest-edge bisection for both tetrahedra "
            "and paired outer/inner triangles. Positive volume and column quadratures cover the full domain "
            "without a removed contact box. Peak cutoffs apply only to peak searches, never field integrals.",
        },
    )
    with np.load(args.mesh_study / "medium-mesh.npz") as mesh:
        original = mesh["vertices"][mesh["tetrahedra"]]
    minimum = args.minimum_over_width * width
    cells, _, volume_history = partition(original, width, along, minimum)
    old_volume = np.abs(np.linalg.det(original[:, 1:] - original[:, :1])).sum() / 6
    volumes = np.abs(np.linalg.det(cells[:, 1:] - cells[:, :1])) / 6
    np.testing.assert_allclose(volumes.sum(), old_volume, rtol=1e-12, atol=0)
    q, weights = get_quadrature_tet(2)
    bary = np.column_stack((1 - q.sum(axis=0), q.T))
    volume_points = np.einsum("qc,tci->tqi", bary, cells).reshape(-1, 3)
    volume_weights = (volumes[:, None] * (6 * weights)).reshape(-1)
    assert np.all(volume_weights > 0)
    with np.load(args.mesh_study / "coarse-mesh.npz") as mesh:
        faces = mesh["mid_faces"]
        outer, inner = mesh["outer_vertices"][faces], mesh["inner_vertices"][faces]
    original_mid = (outer + inner) / 2
    original_area = (
        np.linalg.norm(
            np.cross(original_mid[:, 1] - original_mid[:, 0], original_mid[:, 2] - original_mid[:, 0]), axis=1
        ).sum()
        / 2
    )
    outer, companions, column_history = partition(outer, width, along, minimum, (inner,))
    inner = companions[0]
    mid = (outer + inner) / 2
    vectors = np.cross(mid[:, 1] - mid[:, 0], mid[:, 2] - mid[:, 0])
    areas = np.linalg.norm(vectors, axis=1) / 2
    np.testing.assert_allclose(areas.sum(), original_area, rtol=1e-12, atol=0)
    normals = vectors / (2 * areas[:, None])
    bary = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    outside, inside = [np.einsum("qc,tci->tqi", bary, array).reshape(-1, 3) for array in [outer, inner]]
    normals, areas = np.repeat(normals, 3, axis=0), np.repeat(areas / 3, 3)
    thickness = np.abs(np.einsum("ni,ni->n", outside - inside, normals))
    projection = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    knots, gauss_weights = np.polynomial.legendre.leggauss(args.depth_order)
    knots, gauss_weights = (knots + 1) / 2, gauss_weights / 2
    points, weights, moments, indices, depths = [], [], [], [], []
    for i, (top, bottom, height) in enumerate(zip(outside, inside, thickness)):
        near = np.linalg.norm(top - [width / 2, 0, 0.03]) < 8 * along
        cuts = (
            np.unique(
                np.r_[0, np.clip(np.array([0.25 * width, width, 4 * width, 0.25 * along, along]) / height, 0, 1), 1]
            )
            if near
            else np.array([0, 1])
        )
        value, measure = [], []
        for j, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
            if near and j == 0:
                value.extend(right * knots**2)
                measure.extend(2 * right * knots * gauss_weights)
            else:
                value.extend(left + (right - left) * knots)
                measure.extend((right - left) * gauss_weights)
        value, measure = np.array(value), np.array(measure)
        np.testing.assert_allclose([measure.sum(), measure @ value], [1, 0.5], rtol=1e-12, atol=1e-14)
        points.append(top * (1 - value[:, None]) + bottom * value[:, None])
        weights.extend(measure * height)
        moments.extend(measure * height**2 * (0.5 - value))
        indices.extend([i] * len(value))
        depths.extend(value)
    points = np.concatenate(points)
    # Three explicit positive-depth planes plus all original global vertices
    # beyond the smallest plane. Nested along-edge samples expose peak sampling.
    cutoffs = width * np.array([0.25, 1, 0.25 * along / width])
    x = width * np.unique(np.r_[np.linspace(-2, 2, 33), [-4, 4]])
    y = along * np.linspace(-1.5, 1.5, 121)
    grid = np.stack(np.meshgrid(x, y, cutoffs, indexing="ij"), axis=-1).reshape(-1, 3)
    grid[:, 2] = 0.03 - grid[:, 2]
    own = []
    for name in ["coarse", "medium", "fine"]:
        with np.load(args.mesh_study / (name + "-mesh.npz")) as mesh:
            own.append(mesh["vertices"])
    own = np.unique(np.concatenate(own), axis=0)
    own = own[0.03 - own[:, 2] >= cutoffs.min()]
    peaks = np.concatenate((grid, own))
    np.savez_compressed(
        args.output / "queries.npz",
        volume_points_m=volume_points,
        volume_weights_m3=volume_weights,
        column_points_m=points,
        column_index=np.array(indices),
        column_depth_fraction=np.array(depths),
        column_weights_m=np.array(weights),
        column_moment_weights_m2=np.array(moments),
        column_projection=projection,
        column_areas_m2=areas,
        column_outer_m=outside,
        column_inner_m=inside,
        peak_points_m=peaks,
        peak_cutoffs_m=cutoffs,
        peak_grid_shape=np.array([len(x), len(y), len(cutoffs)]),
    )
    result = {
        "passed": True,
        "volume_partition": volume_history,
        "column_partition": column_history,
        "original_volume_m3": float(old_volume),
        "partition_volume_m3": float(volume_weights.sum()),
        "original_mid_area_m2": float(original_area),
        "partition_mid_area_m2": float(areas.sum()),
        "volume_points": len(volume_points),
        "column_points": len(points),
        "material_columns": len(areas),
        "peak_points": len(peaks),
        "peak_cutoffs_m": cutoffs.tolist(),
        "minimum_volume_depth_m": float((0.03 - volume_points[:, 2]).min()),
        "minimum_column_depth_m": float((0.03 - points[:, 2]).min()),
        "total_queries": len(volume_points) + len(points) + len(peaks),
        "whole_goal_complete": False,
    }
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
