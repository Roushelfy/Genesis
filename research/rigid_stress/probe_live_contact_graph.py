"""Trial a native contact graph in the actual rigid recovery lifecycle."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import recovery
from research.rigid_stress.probe_fused_contacts import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--trial-mode", choices=("native", "fused"), required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = None
    if args.trial_mode == "fused":
        fused, source_hash = generate(output.parent)
        epsilon = [0.0]

        def defer_anchor(source_epsilon, contact_state, surface_info):
            epsilon[0] = source_epsilon

        def defer_pressure(contact_state, surface_info, cooperative=True):
            assert cooperative

        def combined(contact_state, stress_state, stress_info, surface_info, cooperative=True, cached_bounds=True):
            assert cooperative and cached_bounds
            fused(epsilon[0], contact_state, stress_state, stress_info, surface_info)

        # Research-only scheduling substitution; the owned native passes and
        # association/acceptance/reset lifecycle are unchanged.
        recovery.kernel_anchor = defer_anchor
        recovery.kernel_pressure = defer_pressure
        recovery.kernel_scatter = combined
    sys.argv = [sys.argv[0], *remaining]
    benchmark.main()
    result = json.loads(output.read_text())
    result["development_contact_graph_trial"] = args.trial_mode
    result["trial_contact_source_hash"] = source_hash
    result["trial_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
