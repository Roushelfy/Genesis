"""Compare policy inference precision while native stress stays Quadrants FP64."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

from examples.speed_benchmark import rigid_stress as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--policy-precision", choices=("64", "32"), required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    warmup = int(remaining[remaining.index("--warmup") + 1])
    checks = {"samples": 0, "maximum_output_error": 0.0, "maximum_control_error_rad": 0.0}
    if args.policy_precision == "32":
        original = torch.nn.Sequential.forward
        policies = {}

        def forward(self, observation):
            key = id(self)
            if key not in policies:
                policies[key] = [copy.deepcopy(self), 0]
                self.to(dtype=torch.float32)
            reference, calls = policies[key]
            policies[key][1] += 1
            action = original(self, observation.to(dtype=torch.float32))
            if calls < warmup and calls % 100 == 0:
                expected = original(reference, observation)
                difference = (action.to(dtype=torch.float64) - expected).abs().max().item()
                checks["samples"] += 1
                checks["maximum_output_error"] = max(checks["maximum_output_error"], difference)
                checks["maximum_control_error_rad"] = 1e-4 * checks["maximum_output_error"]
                assert checks["maximum_control_error_rad"] < 1e-9
            return action

        # Research-only policy arithmetic; native rigid/stress classes are unchanged.
        torch.nn.Sequential.forward = forward
    sys.argv = [sys.argv[0], *remaining]
    benchmark.main()
    result = json.loads(output.read_text())
    result["policy"]["dtype"] = "float32" if args.policy_precision == "32" else "float64"
    result["policy"]["scope"] = "inference rollout"
    result["policy"]["fp64_action_reference_warmup"] = checks
    result["policy"]["control_error_budget_rad"] = 1e-9
    result["trial_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["note_policy_precision"] = "Native stress remains Quadrants FP64 with unchanged full-gauge budgets."
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
