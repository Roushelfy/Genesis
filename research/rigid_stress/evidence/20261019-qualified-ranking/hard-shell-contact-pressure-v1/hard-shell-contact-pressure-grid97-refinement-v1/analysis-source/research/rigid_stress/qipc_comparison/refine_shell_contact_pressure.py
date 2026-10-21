"""Refine only the unresolved pressure grid, reusing completed normal sources."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from skfem import Basis, ElementTetP2, MeshTet

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse
from .study_shell_contact_pressure import (
    contact_minimum,
    hat_compliance,
    hat_source_weights,
    points_on_grid,
    pressure_observation,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parent", type=Path, required=True)
    cli.add_argument("--halfspace-seed", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = args.parent
    assert (parent.parent / "returncode.txt").read_text().strip() == "0"
    original = json.loads((parent / "result.json").read_text())
    start = json.loads((parent / "start.json").read_text())
    assert original["execution_completed"] and len(original["rows"]) == 9
    assert len(original["normal_source_checks"]) == 25
    assert all(row["elastic_residual_relative"] < 1e-7 for row in original["normal_source_checks"])
    mesh_file = Path(start["mesh"]["path"])
    assert file_id(mesh_file) == start["mesh"]
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "parent_result": file_id(parent / "result.json"),
            "parent_start": file_id(parent / "start.json"),
            "source_table": file_id(parent / "normal-source-table.npz"),
            "previous_pressures": file_id(parent / "pressures.npz"),
            "halfspace_seed": file_id(args.halfspace_seed / "pressures.npz") if args.halfspace_seed else None,
            "mesh": start["mesh"],
            "targets": original["targets"],
            "scope": "One added 97x97 scalar pressure grid. Reuse the completed 25-source normal table, with no new elastic factorization or Hertz pressure/support prescribed. The initial pressure is an exact nested prolongation of the computed 33x33 half-space solution; all nodes remain free to enter or leave contact.",
        },
    )
    load, young, poisson = 0.2, 1e10, 0.3
    modulus = young / (1 - poisson**2)
    curvature = np.diag([50.0, 80.0])
    hertz_axes, _, _, peak = ellipse(load, modulus, curvature)
    np.testing.assert_array_equal(hertz_axes, original["hertz_axes_m"])
    with np.load(mesh_file) as mesh:
        vertices, tets = mesh["vertices"], mesh["tetrahedra"]
    begun = perf_counter()
    basis = Basis(MeshTet(vertices.T, tets.T, sort_t=False), ElementTetP2(), intorder=4)
    construction_seconds = perf_counter() - begun
    source_axes = [np.linspace(-1.5 * width, 1.5 * width, 5) for width in hertz_axes]
    with np.load(parent / "normal-source-table.npz") as table:
        np.testing.assert_array_equal(table["coordinates_m"], basis.doflocs.T)
        sources = np.column_stack((points_on_grid(source_axes), np.full(25, 0.03)))
        np.testing.assert_array_equal(table["source_points_m"], sources)
        normal_displacements = table["displacement_m_per_n"][:, 2, :]
    pressure_half_width = 1.3 * hertz_axes
    axes = [np.linspace(-width, width, 97) for width in pressure_half_width]
    old_axes = [np.linspace(-width, width, 33) for width in pressure_half_width]
    query_axes = [np.linspace(-width, width, 129) for width in pressure_half_width]
    points = points_on_grid(axes)
    widths = np.array([axis[1] - axis[0] for axis in axes])
    with np.load(parent / "pressures.npz") as old:
        old_force = old["grid33-sources0-force_n"].reshape(33, 33) / load
        previous = RegularGridInterpolator(
            [np.linspace(-width, width, 65) for width in pressure_half_width],
            old["grid65-sources5-pressure_pa"],
        )(points_on_grid(query_axes))
    old_area = np.prod([axis[1] - axis[0] for axis in old_axes])
    initial = RegularGridInterpolator(old_axes, old_force / old_area)(points) * widths.prod()
    if args.halfspace_seed:
        seed = args.halfspace_seed
        assert (seed.parent / "returncode.txt").read_text().strip() == "0"
        assert json.loads((seed / "result.json").read_text())["execution_completed"]
        with np.load(seed / "pressures.npz") as previous_refinement:
            initial = previous_refinement["grid97-sources0-force_n"] / load
    base = hat_compliance(axes, modulus, 8)
    rigid = np.column_stack((np.ones(len(points)), points / hertz_axes[0]))
    gap = np.einsum("ni,ij,nj->n", points, curvature, points) / 2
    target_points = np.column_stack((points, np.full(len(points), 0.03)))
    probes = sparse.vstack(
        [basis.probes(target_points[start : start + 128].T) for start in range(0, len(target_points), 128)],
        format="csr",
    )
    response = -(probes @ normal_displacements)
    scale = load / (modulus * hertz_axes[0])
    selections = {3: np.array([i * 5 + j for i in [0, 2, 4] for j in [0, 2, 4]]), 5: np.arange(25)}
    rows, arrays, pressures = [], {}, {}
    for table in [0, 3, 5]:
        correction = np.zeros_like(base)
        if table:
            regular_axes = source_axes if table == 5 else [axis[::2] for axis in source_axes]
            correction = response[:, selections[table]] @ hat_source_weights(regular_axes, points, widths)
        matrix = base + correction
        print(f"Refining 97x97 with {table} source-table points per axis.", flush=True)
        begun = perf_counter()
        solution = contact_minimum(matrix * load / scale, gap / scale, rigid, initial)
        seconds = perf_counter() - begun
        initial = solution.force
        observation, pressure, nodal = pressure_observation(axes, solution, load, query_axes, hertz_axes, peak)
        label = f"grid97-sources{table}"
        observation.update(label=label, nodes_per_axis=97, source_table_axis=table, solve_seconds=seconds)
        rows.append(observation)
        pressures[table] = pressure
        arrays.update(
            {
                label + "-pressure_pa": nodal,
                label + "-force_n": solution.force * load,
                label + "-scaled_gap": solution.gap,
                label + "-coordinates_m": solution.coordinates * scale,
            }
        )
        write_json(args.output / "contact-progress.json", {"rows": rows})
        del correction, matrix
    reference = pressures[5]
    change = float(np.linalg.norm(previous - reference) / np.linalg.norm(reference))
    source_change = float(np.linalg.norm(pressures[3] - reference) / np.linalg.norm(reference))
    np.savez_compressed(args.output / "pressures.npz", common_points_m=points_on_grid(query_axes), **arrays)
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "rows": rows,
            "construction_seconds_without_factorization": construction_seconds,
            "grid65_grid97_relative_pressure_change": change,
            "source_table_relative_pressure_change": source_change,
            "targets": original["targets"],
            "discrete_pressure_refinement_targets_met": bool(
                change <= original["targets"]["pressure_grid_change"]
                and source_change <= original["targets"]["source_table_change"]
            ),
            "independent_complete_stress_qualified": False,
            "finite_body_mesh_refinement_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
