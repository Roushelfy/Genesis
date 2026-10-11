"""Evaluate a Torch CUDA graph for policy inference only; stress stays Quadrants."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

from examples.speed_benchmark import rigid_stress as benchmark


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy-graph", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    command = [sys.executable, *sys.argv]
    original = torch.nn.Sequential.forward
    workspaces = {}
    if args.policy_graph:

        def forward(module, observation):
            if observation.dtype != torch.float32:
                return original(module, observation)
            if module not in workspaces:
                static_input = observation.clone()
                current = torch.cuda.current_stream(observation.device)
                side = torch.cuda.Stream(device=observation.device)
                side.wait_stream(current)
                with torch.cuda.stream(side):
                    for _ in range(3):
                        original(module, static_input)
                current.wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    static_output = original(module, static_input)
                current.wait_stream(side)
                workspaces[module] = (graph, static_input, static_output)
            graph, static_input, static_output = workspaces[module]
            static_input.copy_(observation)
            graph.replay()
            return static_output

        torch.nn.Sequential.forward = forward
    sys.argv = ["policy_graph_override", *remaining]
    benchmark.main()
    torch.nn.Sequential.forward = original
    output.output.with_suffix(".policy-variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "policy_graph": args.policy_graph,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "additional_static_policy_io_bytes": sum(
                    source.numel() * source.element_size() + target.numel() * target.element_size()
                    for _, source, target in workspaces.values()
                ),
                "scope_note": "Torch policy inference only. Identical seeded FP32 layers; graph capture/setup excluded from timing, every observation cast/static-input copy/replay/action use included. Graph pool memory is additional to static IO. Stress numerical functions, rigid integration, controls, contact model, reset and final budgets are unchanged. This is not RL training.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
