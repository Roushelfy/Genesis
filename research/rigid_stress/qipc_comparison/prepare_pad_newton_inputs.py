"""Prepare independent physical operators and cold guesses for native pressure checks."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky

from .artifacts import file_id, runtime_identity, write_json
from .compliant_plate_reference import pad_quadrature
from .p2_elastic import FreeElasticP2
from .pad_equilibrium import normal_pad_equilibrium


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start, result = (json.loads((args.study / name).read_text()) for name in ("start.json", "result.json"))
    assert result["passed"] and len(result["cases"]) >= 2
    for key in ("mesh", "seed_operator"):
        assert file_id(Path(start[key]["path"])) == start[key]
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "study": file_id(args.study / "result.json"),
            "inputs": [start["mesh"], start["seed_operator"]],
            "scope": "CPU reference preparation for the native numerical stage. Geometry, compliance condensation and the cold rigid guess are supplied independently here.",
        },
    )
    with np.load(start["mesh"]["path"]) as saved:
        vertices, tets, outer, faces = (saved[key] for key in ("vertices", "tetrahedra", "outer_vertices", "mid_faces"))
    with np.load(start["seed_operator"]["path"]) as saved:
        seed_nodes, compliance = saved["nodes"], saved["compliance_m_per_n"]
    model = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000, factorize=False)
    factor = cholesky(compliance, lower=True)
    center = vertices.mean(axis=0)
    records = []
    for order in (0, 6):
        nodes, interpolation, points, area, _ = pad_quadrature(model, len(outer), 0.003, order)
        np.testing.assert_array_equal(nodes, seed_nodes)
        top = model.vertices[nodes, 2] > center[2]
        sign = np.where(top, -1.0, 1.0)
        gap = np.where(top, outer[:, 2].max() - model.vertices[nodes, 2], model.vertices[nodes, 2] - outer[:, 2].min())
        rigid = np.column_stack(
            (
                sign,
                sign * (model.vertices[nodes, 1] - center[1]) / 0.03,
                -sign * (model.vertices[nodes, 0] - center[0]) / 0.03,
                top,
            )
        )
        mapping = np.column_stack((interpolation @ factor, interpolation @ rigid))
        gaps, weights, initial, scales, loads, elastic_scales, expected, references, springs = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for case_index, case in enumerate(result["cases"]):
            young, stiffness, load = (case[key] for key in ("young_pa", "pad_stiffness_n_per_m3", "load_n"))
            scale = load / (stiffness * np.pi * 0.002**2)
            stiffness_factor = 1 + case["stiffness_gradient"] * np.tanh(points[:, 0] / 0.001)
            spring = stiffness * stiffness_factor * area
            pose, _, _, _ = normal_pad_equilibrium(
                interpolation @ gap, interpolation @ rigid, area * stiffness_factor, stiffness, load
            )
            reference = next(
                row
                for row in result["records"]
                if row["case"] == case_index
                and row["order"] == order
                and row["mode"] == "coupled"
                and row["search_contains_actual_elastic_compression"]
            )
            assert file_id(Path(reference["field"]["path"])) == reference["field"]
            with np.load(reference["field"]["path"]) as saved:
                np.testing.assert_array_equal(saved["points_m"], points)
                expected.append(saved["point_force_n"] / load)
            references.append(reference["field"])
            gaps.append(interpolation @ gap / scale)
            weights.append(spring * scale / load)
            initial.append(np.r_[np.zeros(len(nodes)), pose / scale])
            elastic_scales.append(np.sqrt((1e10 / young) * load / scale))
            scales.append(scale)
            loads.append(load)
            springs.append(spring)
        path = args.output / f"q{order}.npz"
        linear = np.zeros((len(nodes) + 4, len(loads)))
        linear[-1] = 1.0
        np.savez_compressed(
            path,
            mapping=mapping,
            elastic=np.r_[np.ones(len(nodes)), np.zeros(4)],
            gap=np.array(gaps).T,
            weight=np.array(weights).T,
            linear=linear,
            initial=np.array(initial).T,
            elastic_scale=elastic_scales,
            expected_force=np.array(expected).T,
            loads_n=loads,
            length_scales_m=scales,
            compliance_m_per_n=compliance,
            interpolation=interpolation,
            rigid=rigid,
            nodal_gap_m=gap,
            spring_n_per_m=np.array(springs).T,
            young_pa=[case["young_pa"] for case in result["cases"]],
            nodes=nodes,
            node_sign=sign,
            points_m=points,
            area_m2=area,
            p2_coordinates_m=model.vertices,
            tetrahedra=tets,
        )
        records.append(
            {
                "order": order,
                "points": len(points),
                "unknowns": mapping.shape[1],
                "environments": len(loads),
                "inputs": file_id(path),
                "references": references,
            }
        )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "records": records,
            "study": file_id(args.study / "result.json"),
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
