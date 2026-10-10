"""Trial device failure compaction within the existing native rigid lifecycle."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import genesis as gs
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import inverse, model
from genesis.utils.array_class import V
from research.rigid_stress.probe_failed_compaction import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--trial-mode", choices=("native", "packed"), required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    selections = {}
    if args.trial_mode == "packed":
        packed_inverse, packed_residual, hashes = generate(output.parent)
        original_inverse = inverse.kernel_apply_inverse
        original_residual = model.kernel_full_residual

        def buffers(state):
            key = id(state)
            if key not in selections:
                selections[key] = (V(dtype=gs.qd_int, shape=(state.active.shape[0],)), V(dtype=gs.qd_int, shape=()))
            return selections[key]

        def apply(young, state, info, correction, only_failed):
            if correction or only_failed:
                packed_inverse(young, state, info, correction, only_failed, *buffers(state))
            else:
                original_inverse(young, state, info, correction, only_failed)

        def residual(young, tolerance, absolute, state, info, only_active):
            if only_active:
                packed_residual(young, tolerance, absolute, state, info, only_active, *buffers(state))
            else:
                original_residual(young, tolerance, absolute, state, info, only_active)

        inverse.kernel_apply_inverse = apply
        model.kernel_full_residual = residual
    sys.argv = [sys.argv[0], *remaining]
    benchmark.main()
    result = json.loads(output.read_text())
    result["development_failed_compaction_trial"] = args.trial_mode
    result["trial_native_source_sha256"] = hashes
    result["trial_extra_scratch_bytes"] = sum((selected.shape[0] + 1) * 4 for selected, _count in selections.values())
    result["trial_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
