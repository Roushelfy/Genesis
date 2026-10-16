"""Declare a new contact-distance configuration for a fresh original-history reference solve."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-closure", type=Path, required=True)
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert math.isfinite(args.distance) and args.distance > 0
    closed_source(args.original_closure)
    original = args.original_closure / "results/configuration.json"
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    config = json.loads(original.read_text())
    capture = Path(config["capture"])
    closed_source(capture)
    assert int((capture / "returncode.txt").read_text()) == 0
    capture_summary = capture / "results/summary.json"
    capture_sha = hashlib.sha256(capture_summary.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "original_configuration.json").write_bytes(original.read_bytes())
    config.update(
        dhat=args.distance,
        output=str(args.output),
        configuration_input_role="Contact regularization input only; the downstream reference independently solves all64 held controls and five dynamic states",
        physical_states_generated=0,
    )
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    assert hashlib.sha256(capture_summary.read_bytes()).hexdigest() == capture_sha
    summary = {
        "scope": __doc__,
        "original_closure": str(args.original_closure),
        "actual_original_closure_wrapper_code": int((args.original_closure / "returncode.txt").read_text()),
        "declared_barrier_distance_m": args.distance,
        "physical_states_generated": 0,
        "input_sha256": {str(original): before, str(capture_summary): capture_sha},
        "goal_completion": False,
        "limitations": "The existing reference reads closure-run only for its capture and barrier distance, then performs a fresh64-control initializer from zero. This preparation supplies that declared configuration without supplying or claiming any solved closure, copied contact state or successful physical simulation. Geometry, material, mass, friction, actuation, timestep and original stage counts belong to the unchanged downstream solver inputs. Regularization sensitivity and physical convergence require those actual solves and independent audits.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
