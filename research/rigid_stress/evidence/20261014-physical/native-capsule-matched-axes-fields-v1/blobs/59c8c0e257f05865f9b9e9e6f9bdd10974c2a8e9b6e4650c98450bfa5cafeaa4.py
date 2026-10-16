"""Run the unchanged coupled protocol with wide body iterates and stricter internal convergence."""

import json
import sys
from pathlib import Path

from . import franka_contact_reference, run_articulated_franka
from .wide_coordinate_franka import WideArticulatedContact, WidePiolaBulk


def main():
    franka_contact_reference.StableNeoHookeanBulk = WidePiolaBulk
    run_articulated_franka.ArticulatedContact = WideArticulatedContact
    run_articulated_franka.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    path = output / "summary.json"
    summary = json.loads(path.read_text())
    summary["arithmetic"] = {
        "body_coordinates_and_piola_force": "longdouble",
        "contact_geometry_energy_tangent_and_stress_postprocessing": "float64",
        "internal_Newton_target_N": 1e-10,
        "original_external_force_budgets_N": 1e-9,
        "physical_law_quadrature_and_control_changed": False,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
