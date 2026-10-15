"""Resolve one retained lagged-friction initialization failure with current-force Newton."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np

from .contact_reference import IndependentContactSystem
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) != 0
    config = json.loads((args.run / "solution/configuration.json").read_text())
    assert config["geometry"] == "egg"
    references = []
    for name, path, young, poisson, density in (
        ("body", Path(config["mesh"]), 1e10, 0.3, 2000),
        ("left_shoe", args.run / "solution/left_shoe_mesh.npz", config["pad_young"], 0, 1000),
        ("right_shoe", args.run / "solution/right_shoe_mesh.npz", config["pad_young"], 0, 1000),
    ):
        with np.load(path) as saved:
            references.append(
                (name, SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, density))
            )
    system = IndependentContactSystem(references, config["dhat"], 1e6, config["friction"], 1e-8)
    with np.load(args.run / "solution/initialization_failure.npz") as failed:
        np.testing.assert_array_equal(system.rest, failed["rest_positions"])
        np.testing.assert_array_equal(system.faces, failed["surface_triangles"])
        u, previous, fixed = (failed[key].copy() for key in ("displacement", "previous_displacement", "boundary_dofs"))
    initial = system.evaluate_updated_friction(u, previous, None, False)
    free = np.setdiff1d(np.arange(len(u)), fixed)
    initial_residual = float(np.linalg.norm(initial[1][free]))
    before_fixed = u[fixed].copy()
    solved, g, f, energy, records = system.solve_updated_friction(
        fixed,
        before_fixed,
        1,
        initial_displacement=u,
        initial_previous_displacement=previous,
        phase="identical_failed_egg_initialization_step",
        failure_path=args.output / "failure.npz",
    )
    np.testing.assert_array_equal(solved[fixed], before_fixed)
    residual = float(np.linalg.norm(g[free]))
    assert residual <= 1e-9
    normal, current = system.normal_collisions(solved)
    assert not ipctk.has_intersections(system.mesh, current)
    np.savez_compressed(
        args.output / "recovered_state.npz",
        rest_positions=system.rest,
        displacement=solved,
        previous_displacement=previous,
        lagged_displacement=solved,
        boundary_dofs=fixed,
        surface_triangles=system.faces,
        contact_force_N=f,
    )
    output = {
        "scope": __doc__,
        "source_failed_run": str(args.run),
        "initial_current_force_residual_N": initial_residual,
        "solved_current_force_residual_N": residual,
        "unchanged_force_tolerance_N": 1e-9,
        "additional_newton_iterations": len(records),
        "normal_stencils": len(normal),
        "diagnostic_potential_J": energy,
        "goal_completion": False,
        "limitations": "Geometry, materials, all fixed displacement rows, current target values and the saved previous friction state are unchanged. The Newton method differentiates current friction geometry/velocity rather than iterating lagged forces. This recovers only one initialization increment, not the remaining cold continuation, free-body gravity solve, nonlinear compliance, contact convergence or actual Franka matching.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
