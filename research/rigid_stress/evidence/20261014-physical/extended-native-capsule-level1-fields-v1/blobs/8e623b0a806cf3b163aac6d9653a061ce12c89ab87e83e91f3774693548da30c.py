"""Evaluate Quadrants graph argument caching without changing numerical work."""

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import pipeline, recovery
from research.rigid_stress import native_trajectory_oracle as oracle


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fastcache", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    source = Path(pipeline.__file__).read_text()
    generated = output.output.with_suffix(".pipeline.py")
    if args.fastcache:
        tree = ast.parse(source)
        changed = 0
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "kernel_pipeline":
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "qd.kernel":
                        assert all(keyword.arg != "fastcache" for keyword in decorator.keywords)
                        decorator.keywords.append(ast.keyword(arg="fastcache", value=ast.Constant(value=True)))
                        changed += 1
        assert changed == 1, changed
        ast.fix_missing_locations(tree)
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(ast.unparse(tree) + "\n")
        namespace = pipeline.__dict__.copy()
        exec(compile(tree, str(generated), "exec"), namespace)  # noqa: S102 - owned diagnostic source
        recovery.kernel_pipeline = namespace["kernel_pipeline"]
    sys.argv = ["pipeline_fastcache_override", *remaining]
    (oracle.main if args.oracle else benchmark.main)()
    output.output.with_suffix(".fastcache-variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "fastcache": args.fastcache,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "original_pipeline_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "generated_pipeline_sha256": hashlib.sha256(generated.read_bytes()).hexdigest()
                if args.fastcache
                else None,
                "note": "Only the existing Quadrants graph decorator's fastcache option changes. All numerical expressions, typed fields, template arguments, controls, load model, resets and acceptance remain identical. Exact generated module is retained.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
