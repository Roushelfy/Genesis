"""Keep ordinary public control/reset calls while reusing configuration indices."""

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from examples.rigid import egg_stress_controller as controller
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark import rigid_stress as benchmark
from research.rigid_stress import native_trajectory_oracle as oracle


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control-ranges", action="store_true")
    parser.add_argument("--resident-resets", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    command = [sys.executable, *sys.argv]
    source = Path(controller.__file__).read_text()
    generated = output.output.with_suffix(".controller.py")
    if args.control_ranges:
        tree = ast.parse(source)

        class Ranges(ast.NodeTransformer):
            replacements = 0

            def visit_Attribute(self, node):
                if ast.unparse(node) == "workload.arm":
                    self.replacements += 1
                    return ast.copy_location(
                        ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(7)], keywords=[]), node
                    )
                if ast.unparse(node) == "workload.fingers":
                    self.replacements += 1
                    return ast.copy_location(
                        ast.Call(
                            func=ast.Name(id="range", ctx=ast.Load()),
                            args=[ast.Constant(7), ast.Constant(9)],
                            keywords=[],
                        ),
                        node,
                    )
                return self.generic_visit(node)

        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "native_step")
        transform = Ranges()
        transform.visit(function)
        # The Torch source selection also uses a basic slice, avoiding a
        # range interpreted as advanced indexing by Tensor.__getitem__.
        for node in ast.walk(function):
            if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "workspace['target']":
                assert isinstance(node.slice, ast.Tuple)
                node.slice.elts[1] = ast.Slice(lower=ast.Constant(0), upper=ast.Constant(7), step=None)
        assert transform.replacements == 4, transform.replacements
        ast.fix_missing_locations(tree)
        generated.write_text(ast.unparse(tree) + "\n")
        namespace = controller.__dict__.copy()
        exec(compile(tree, str(generated), "exec"), namespace)  # noqa: S102 - owned diagnostic source
        import examples.rigid.franka_egg_stress as example

        example.native_step = namespace["native_step"]

    caches = {}
    if args.resident_resets:
        original = FrankaEgg.reset

        def reset(workload, envs_idx=None):
            if envs_idx is None or workload.reset_state is None:
                return original(workload, envs_idx)
            # Arbitrary caller selections use the original route. Only repeat
            # CPU integer selections are cached; keys contain the complete IDs.
            if (
                not isinstance(envs_idx, np.ndarray)
                or envs_idx.ndim != 1
                or not np.issubdtype(envs_idx.dtype, np.integer)
            ):
                return original(workload, envs_idx)
            key = tuple(int(value) for value in envs_idx)
            if len(key) == 1:
                # Generic masking converts a one-element CUDA tensor with
                # item(); keep an ordinary Python integer for this case.
                selection = key[0]
            else:
                cache = caches.setdefault(workload, {})
                if key not in cache:
                    cache[key] = torch.as_tensor(envs_idx.copy(), dtype=torch.int64, device=gs.device)
                selection = cache[key]
            workload.scene.reset(state=workload.reset_state, envs_idx=selection)
            workload.phase[selection] = 0.0
            workload.reset_count += len(key)

        FrankaEgg.reset = reset
    sys.argv = ["native_indices_override", *remaining]
    (oracle.main if args.oracle else benchmark.main)()
    output.output.with_suffix(".indices-variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "control_ranges": args.control_ranges,
                "resident_resets": args.resident_resets,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "original_controller_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "generated_controller_sha256": hashlib.sha256(generated.read_bytes()).hexdigest()
                if args.control_ranges
                else None,
                "resident_reset_selection_bytes": sum(
                    value.numel() * value.element_size() for cache in caches.values() for value in cache.values()
                ),
                "resident_reset_selection_count": sum(len(cache) for cache in caches.values()),
                "scope_note": "Configuration indices only. Same public control/set-force-range/Scene.reset calls, same targets/actions/load model/numerical functions and all final budgets. Actual per-step index creation/cache-key and maintenance costs are timed. New resident selections initially allocate inside warmup/timing and are not assumed free. No contact position, count, direction, symmetry or friction is fixed.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
