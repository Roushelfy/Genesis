"""Trial native graph scheduling inside the actual existing rigid recovery lifecycle."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import genesis as gs
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress.model import StressModel
from research.rigid_stress.probe_fused_elastic import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--trial-mode", choices=("native", "serial", "parallel"), required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    if args.trial_mode != "native":
        serial, parallel, hashes = generate(output.parent)
        selected = parallel if args.trial_mode == "parallel" else serial
        original = StressModel.recover

        def recover(self, omega, state, options=None, history=None, surface_load=False):
            if options is None:
                options = self.options
            if (
                gs.backend == gs.cuda
                and surface_load
                and self.surface_inverse is not None
                and history is None
                and options.inverse_corrections == 2
                and options.cached_peak
                and options.cooperative_solve
            ):
                selected(
                    options.young,
                    options.poisson,
                    options.tolerance,
                    options.absolute_tolerance,
                    omega,
                    state,
                    self.info,
                    self.surface_inverse.info,
                    self.inverse.info,
                    self.factor.info,
                    self.info.vertices.shape[0],
                )
            else:
                original(self, omega, state, options, history, surface_load)

        # Research-only substitution on the owned engine class; no callback or rigid dynamics change.
        StressModel.recover = recover
    sys.argv = [sys.argv[0], *remaining]
    benchmark.main()
    result = json.loads(output.read_text())
    result["development_elastic_graph_trial"] = args.trial_mode
    result["trial_native_source_hashes"] = hashes
    result["trial_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
