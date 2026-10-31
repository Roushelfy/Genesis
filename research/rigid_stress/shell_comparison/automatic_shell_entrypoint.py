"""Run the actual public shell CLI on a reproducible surface, preserving artifacts and failure diagnostics."""

import argparse
import sys
import time
from pathlib import Path

from examples.rigid.franka_egg_automatic_stress import main as example
from research.rigid_stress.qipc_comparison.artifacts import runtime_identity, write_json
from research.rigid_stress.shell_comparison.automatic_shell_geometry import surface_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shape",
        choices=("authored", "egg", "grasp", "round", "neck", "u", "plate", "dome", "dimple", "notch"),
        required=True,
    )
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--substeps", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    argv = [
        "automatic-shell",
        "--backend",
        "shell",
        "--envs",
        str(args.envs),
        "--steps",
        str(args.steps),
        "--log-every",
        "20",
        "--cache-dir",
        str(args.output / "cache"),
    ]
    if args.shape != "authored":
        surface = surface_recipe(args.shape, 1 if args.shape == "neck" else 0)
        source = args.output / "midsurface.obj"
        surface.export(source, digits=17)
        argv.extend(("--mesh", str(source)))
    if args.viewer:
        argv.append("--viewer")
    if args.substeps is not None:
        argv.extend(("--substeps", str(args.substeps)))
    identity = runtime_identity()
    write_json(args.output / "start.json", {"argv": argv, "identity": identity})
    previous = sys.argv
    started = time.perf_counter()
    try:
        sys.argv = argv
        example()
    except Exception as error:
        write_json(
            args.output / "result.json",
            {"passed": False, "error": str(error), "seconds": time.perf_counter() - started},
        )
        raise
    finally:
        sys.argv = previous
    write_json(args.output / "result.json", {"passed": True, "seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
