"""Compare saved-mesh and regenerated-shoe operators on the same recovered egg state."""

import argparse
import json
from pathlib import Path

import numpy as np

from .contact_reference import IndependentContactSystem
from .grasp_contact_reference import compliant_shoe
from .skfem_reference import SkfemElasticReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.failed_run / "solution/configuration.json").read_text())
    with np.load(config["mesh"]) as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    with np.load(args.state) as saved:
        displacement, previous, fixed = (
            saved[key].copy() for key in ("displacement", "previous_displacement", "boundary_dofs")
        )
    free = np.setdiff1d(np.arange(len(displacement)), fixed)
    systems = []
    for regenerate in (False, True):
        references = [("body", body)]
        for name, side in (("left_shoe", -1), ("right_shoe", 1)):
            if regenerate:
                reference = compliant_shoe(
                    config["geometry"],
                    side,
                    config["pad_resolution"],
                    config["initial_gap"],
                    config["pad_thickness"],
                    config["edge_lift"],
                    config["pad_young"],
                )[0]
            else:
                with np.load(args.failed_run / "solution" / f"{name}_mesh.npz") as saved:
                    reference = SkfemElasticReference(
                        saved["vertices"], saved["tetrahedra"], config["pad_young"], 0, 1000
                    )
            references.append((name, reference))
        systems.append(IndependentContactSystem(references, config["dhat"], 1e6, config["friction"], 1e-8))
    first, second = systems
    np.testing.assert_array_equal(first.rest, second.rest)
    np.testing.assert_array_equal(first.faces, second.faces)
    gradients = [system.evaluate_updated_friction(displacement, previous, None, False)[1] for system in systems]
    difference = first.stiffness - second.stiffness
    rows = []
    for left, right in zip(first.parts, second.parts):
        a, b = left.reference, right.reference
        matrix = a.stiffness - b.stiffness
        rows.append(
            {
                "part": left.name,
                "stiffness_max_difference": float(abs(matrix.data).max(initial=0)),
                "mass_modes_max_difference": float(abs(a.mass_modes - b.mass_modes).max()),
                "gram_max_difference": float(abs(a.gram - b.gram).max()),
                "rigid_modes_max_difference": float(abs(a.rigid_modes - b.rigid_modes).max()),
            }
        )
    output = {
        "scope": __doc__,
        "parts": rows,
        "stiffness_max_difference": float(abs(difference.data).max(initial=0)),
        "saved_mesh_free_residual_N": float(np.linalg.norm(gradients[0][free])),
        "regenerated_mesh_free_residual_N": float(np.linalg.norm(gradients[1][free])),
        "free_gradient_difference_N": float(np.linalg.norm((gradients[0] - gradients[1])[free])),
        "goal_completion": False,
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
