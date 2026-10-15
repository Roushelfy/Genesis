"""Run the original physical control with a separately qualified Newton initial guess."""

import json
import sys
from pathlib import Path

from . import franka_contact_reference, run_articulated_franka
from .predicted_wide_franka import PredictedWideContact
from .wide_coordinate_franka import WidePiolaBulk


def main():
    franka_contact_reference.StableNeoHookeanBulk = WidePiolaBulk
    systems = []

    class RecordedContact(PredictedWideContact):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            systems.append(self)

    run_articulated_franka.ArticulatedContact = RecordedContact
    run_articulated_franka.main()
    path = Path(sys.argv[sys.argv.index("--output") + 1]) / "summary.json"
    summary = json.loads(path.read_text())
    summary["arithmetic"] = {
        "body_coordinates_and_piola_force": "longdouble",
        "contact_geometry_energy_tangent_and_stress_postprocessing": "float64",
        "internal_Newton_target_N": 1e-10,
        "original_external_force_budgets_N": 1e-9,
        "physical_law_quadrature_and_control_changed": False,
    }
    summary["initial_guess_records"] = systems[0].initial_guess_records
    summary["initial_guess_qualification"] = "CCD-qualified velocity prediction; independent all-state comparison required"
    path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
