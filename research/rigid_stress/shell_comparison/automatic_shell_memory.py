"""Sample a fresh automatic-shell GPU process separately from primary throughput."""

import argparse
import json
import resource
import runpy
import sys
from pathlib import Path

from research.rigid_stress.qipc_comparison.artifacts import runtime_identity, write_json
from research.rigid_stress.shell_comparison.memory import ProcessMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--output-mode", choices=("max", "full"), required=True)
    parser.add_argument("--shape", default="egg")
    parser.add_argument("--substeps", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sampler = ProcessMemory(True)
    previous = sys.argv
    error = None
    sampler.start()
    try:
        sys.argv = [
            "automatic-shell-scene",
            "--output",
            str(args.output / "scene"),
            "--envs",
            str(args.envs),
            "--output-mode",
            args.output_mode,
            "--shape",
            args.shape,
            "--substeps",
            str(args.substeps),
            "--steps",
            "5",
            "--repeats",
            "1",
        ]
        runpy.run_module("research.rigid_stress.shell_comparison.automatic_shell_scene", run_name="__main__")
    except BaseException as exc:  # noqa: BLE001 - record the failure, finish sampling, then re-raise
        error = exc
    finally:
        sys.argv = previous
        memory = sampler.finish()
    scene_path = args.output / "scene/results.json"
    scene = json.loads(scene_path.read_text()) if scene_path.is_file() else {}
    write_json(
        args.output / "memory.json",
        {
            "identity": runtime_identity(),
            "passed": error is None and scene.get("passed", False) and bool(memory["samples"]),
            "memory": memory,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "driver_error": str(error) if error else None,
            "primary_timing": False,
            "scope": "Separate process NVIDIA polling, from before simulator import through build and admitted scene/module work. Sampled peaks can miss brief allocations; timings here are intrusive.",
        },
    )
    if error is not None:
        raise error
    assert memory["samples"], "No GPU process-memory samples were collected."


if __name__ == "__main__":
    main()
