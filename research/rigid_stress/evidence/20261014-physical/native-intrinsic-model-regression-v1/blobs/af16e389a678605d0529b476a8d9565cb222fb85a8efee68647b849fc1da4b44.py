"""Run the original coupled control with extended-precision Piola-force contraction."""

import json
import sys
from pathlib import Path

from . import franka_contact_reference, run_articulated_franka
from .extended_piola_reference import ExtendedPiolaBulk


def main():
    franka_contact_reference.StableNeoHookeanBulk = ExtendedPiolaBulk
    run_articulated_franka.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    path = output / "summary.json"
    summary = json.loads(path.read_text())
    summary["bulk_force_contraction"] = "extended longdouble; original law/quadrature/tangent and1e-9 budgets"
    path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
