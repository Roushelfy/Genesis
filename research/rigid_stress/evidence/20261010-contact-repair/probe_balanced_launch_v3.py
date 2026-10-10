"""Paired packed-application scheduling after native inertia-relief integration."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import pipeline, surface_inverse


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-dim", choices=(0, 256, 512), type=int, default=0)
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    source_path = Path("genesis/engine/solvers/rigid/stress/surface_inverse.py")
    source = source_path.read_text()
    generated_source = None
    if args.block_dim:
        node = next(
            item
            for item in ast.parse(source).body
            if isinstance(item, ast.FunctionDef) and item.name == "func_surface_apply_packed"
        )
        node.body.insert(0, ast.parse(f"qd.loop_config(block_dim={args.block_dim})").body[0])
        generated_source = (
            "from genesis.engine.solvers.rigid.stress.surface_inverse import *\n\n"
            + ast.unparse(ast.fix_missing_locations(node))
            + "\n"
        )
        directory = (
            Path(os.environ["RIGID_STRESS_DATA_ROOT"])
            / "cache"
            / "balanced_launch_trials"
            / os.environ["SLURM_JOB_ID"]
            / f"block{args.block_dim}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        module_path = directory / "packed_launch_generated.py"
        module_path.write_text(generated_source)
        spec = importlib.util.spec_from_file_location("packed_launch_generated", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        pipeline.func_surface_apply_packed = module.func_surface_apply_packed
        surface_inverse.func_surface_apply_packed = module.func_surface_apply_packed
    sys.argv = ["examples/speed_benchmark/rigid_stress.py", *remaining]
    benchmark.main()
    output.output.with_suffix(".variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "block_dim": args.block_dim,
                "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "generated_source_sha256": hashlib.sha256(generated_source.encode()).hexdigest()
                if generated_source
                else None,
                "generated_source": generated_source,
                "note": "Only CUDA block scheduling changes; native face scatter and cooperative inertia relief remain enabled. All ordinary reset, observation, retry and fallback costs are counted.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
