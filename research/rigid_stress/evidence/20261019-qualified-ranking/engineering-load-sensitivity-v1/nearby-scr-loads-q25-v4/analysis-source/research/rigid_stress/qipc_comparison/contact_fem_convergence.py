"""Compare complete contact FEM fields on a shared smooth material parameterization without trimming peaks."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json


def material_faces(outer, faces, directions, material_unit=None):
    if material_unit is None:
        unit = outer.copy()
        unit[:, 2] /= 0.03
        unit[:, :2] /= (0.022 * (1 - 0.18 * unit[:, 2]))[:, None]
    else:
        unit = material_unit.copy()
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    triangles = unit[faces]
    centers = triangles.mean(axis=1)
    tree = cKDTree(centers / np.linalg.norm(centers, axis=1)[:, None])
    chosen = np.empty(len(directions), dtype=int)
    bary = np.empty((len(directions), 3))
    for first in range(0, len(directions), 2048):
        query = directions[first : first + 2048]
        _, candidate = tree.query(query, k=32)
        a = triangles[candidate, 0]
        e = triangles[candidate, 1:] - a[:, :, None]
        normal = np.cross(e[:, :, 0], e[:, :, 1])
        distance = (normal * a).sum(axis=-1) / (normal * query[:, None]).sum(axis=-1)
        relative = distance[:, :, None] * query[:, None] - a
        gram = np.einsum("qkai,qkbi->qkab", e, e)
        free = np.linalg.solve(gram, np.einsum("qkai,qki->qka", e, relative)[..., None])[..., 0]
        weights = np.concatenate((1 - free.sum(axis=-1, keepdims=True), free), axis=-1)
        best = weights.min(axis=-1).argmax(axis=-1)
        chosen[first : first + len(query)] = candidate[np.arange(len(query)), best]
        bary[first : first + len(query)] = weights[np.arange(len(query)), best]
    assert bary.min() >= -1e-9, "Material ray locator has an uncovered direction"
    return chosen, bary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--meshes", type=Path, nargs="+", required=True)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--loads", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    cli.add_argument("--depth-samples", type=int, default=32)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert len(args.meshes) == len(args.runs)
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.meshes[-1]) as saved:
        common_outer, common_inner, common_faces = (
            saved[key] for key in ("outer_vertices", "inner_vertices", "mid_faces")
        )
    unit = common_outer.copy()
    unit[:, 2] /= 0.03
    unit[:, :2] /= (0.022 * (1 - 0.18 * unit[:, 2]))[:, None]
    direction = unit[common_faces].mean(axis=1)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    normal = (common_outer - common_inner)[common_faces].mean(axis=1)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    axis = np.eye(3)[np.abs(normal).argmin(axis=1)]
    tangent = np.cross(normal, axis)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    basis = np.stack((tangent, np.cross(normal, tangent)), axis=-1)
    mid = (common_outer + common_inner) / 2
    triangle = mid[common_faces]
    area = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1) / 2
    depth_values = (np.arange(args.depth_samples) + 0.5) / args.depth_samples
    rows = []
    for load in args.loads:
        sampled = []
        raw = []
        for mesh, run in zip(args.meshes, args.runs):
            with np.load(mesh) as saved:
                vertices, tets, outer, faces = (
                    saved[key] for key in ("vertices", "tetrahedra", "outer_vertices", "mid_faces")
                )
            path = run / f"load-{load:g}.npz"
            with np.load(path) as saved:
                stress = saved["stress_pa"]
            selected, weights = material_faces(outer, faces, direction)
            weights = np.take_along_axis(weights, np.argsort(faces[selected], axis=1), axis=1)
            prism = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]])
            corners = prism[[[0, 1, 2, 5], [0, 1, 4, 5], [0, 3, 4, 5]]]
            layers = len(vertices) // len(outer) - 1
            tensor = np.empty((args.depth_samples, len(direction), 2, 2))
            minimum_barycentric = 1.0
            for index, depth in enumerate(depth_values):
                layer = min(int(depth * layers), layers - 1)
                candidates = (layer * 3 + np.arange(3))[None] * len(faces) + selected[:, None]
                point = np.column_stack((weights[:, 1:], np.full(len(weights), depth * layers - layer)))
                free = np.linalg.solve(
                    (corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1)[None],
                    (point[:, None] - corners[None, :, 0])[..., None],
                )[..., 0]
                bary = np.concatenate((1 - free.sum(axis=-1, keepdims=True), free), axis=-1)
                best = bary.min(axis=-1).argmax(axis=-1)
                selected_bary = bary[np.arange(len(direction)), best]
                minimum_barycentric = min(minimum_barycentric, selected_bary.min())
                assert selected_bary.min() >= -1e-7, "Mapped material point is outside its declared solid prism"
                element = candidates[np.arange(len(direction)), best]
                tensor[index] = basis.transpose(0, 2, 1) @ stress[element] @ basis
            sampled.append(tensor)
            dev = stress - np.trace(stress, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
            mises = np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))
            peak = mises.argmax()
            raw.append(
                {
                    "mesh": file_id(mesh),
                    "field": file_id(path),
                    "raw_maximum_mises_pa": float(mises[peak]),
                    "raw_peak_element": int(peak),
                    "raw_peak_position_m": vertices[tets[peak]].mean(axis=0).tolist(),
                    "raw_maximum_tension_pa": float(np.linalg.eigvalsh(stress)[:, -1].max()),
                    "minimum_material_point_barycentric": float(minimum_barycentric),
                }
            )
        comparisons = []
        for index in range(len(sampled) - 1):
            actual, reference = sampled[index], sampled[index + 1]
            numerator = np.einsum("dqij,dqij,q->", actual - reference, actual - reference, area)
            denominator = np.einsum("dqij,dqij,q->", reference, reference, area)
            membrane_a, membrane_r = actual.mean(axis=0) * 0.0005, reference.mean(axis=0) * 0.0005
            bending_a = np.einsum("d,dqij->qij", (0.5 - depth_values), actual) * 0.0005**2 / args.depth_samples
            bending_r = np.einsum("d,dqij->qij", (0.5 - depth_values), reference) * 0.0005**2 / args.depth_samples
            comparisons.append(
                {
                    "coarse_index": index,
                    "fine_index": index + 1,
                    "tangent_relative_l2": float(np.sqrt(numerator / denominator)),
                    "membrane_relative_l2": float(
                        np.sqrt(
                            np.einsum("qij,qij,q->", membrane_a - membrane_r, membrane_a - membrane_r, area)
                            / np.einsum("qij,qij,q->", membrane_r, membrane_r, area)
                        )
                    ),
                    "bending_relative_l2": float(
                        np.sqrt(
                            np.einsum("qij,qij,q->", bending_a - bending_r, bending_a - bending_r, area)
                            / np.einsum("qij,qij,q->", bending_r, bending_r, area)
                        )
                    ),
                    "raw_mises_relative_change": abs(
                        raw[index]["raw_maximum_mises_pa"] / raw[index + 1]["raw_maximum_mises_pa"] - 1
                    ),
                    "raw_tension_relative_change": abs(
                        raw[index]["raw_maximum_tension_pa"] / raw[index + 1]["raw_maximum_tension_pa"] - 1
                    ),
                }
            )
        np.savez_compressed(
            args.output / f"common-{load:g}.npz",
            direction=direction,
            area_m2=area,
            depth_fractions=depth_values,
            basis=basis,
            tangent_pa=np.stack(sampled),
        )
        rows.append({"load_n": load, "raw": raw, "adjacent_comparisons": comparisons})
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "cases": rows,
            "physical_reference_qualified": False,
            "scope": "Complete material-column tangent, membrane and bending comparisons with unsmoothed raw peaks. "
            "Angular material rays and normalized wall depth identify corresponding positions in each "
            "piecewise-planar geometry; geometric-domain changes are included, not called fixed-domain refinement. "
            "Common columns use finest-face centroids and midpoint through-wall integration. "
            "Sampling-depth sensitivity and independent elasticity controls remain required.",
        },
    )


if __name__ == "__main__":
    main()
