"""Select the process-local CUDA-graph control before importing the frozen native benchmark.

All remaining arguments are forwarded verbatim. Graph metadata is recorded after the worker returns, outside its timer.
The wrapper uses only the Python standard library and requires an explicit Slurm rtx-mid/image 1_26 worker process.
"""

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, description=__doc__)
    parser.add_argument("--cuda-graphs", choices=("on", "off"), required=True)
    args, forwarded = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output_args, _unknown = output_parser.parse_known_args(forwarded)
    output = output_args.output.expanduser().resolve()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    if not output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Choose an explicit project data output under /mnt/data/zhaofeng.")
    loaded_before = [
        name
        for name in ("quadrants.lang.kernel", "genesis", "examples.speed_benchmark.rigid_stress")
        if name in sys.modules
    ]
    if loaded_before:
        raise RuntimeError(f"Graph control must precede native imports: {loaded_before}")
    environment_before = os.environ.get("QD_GRAPH")
    requested = "1" if args.cuda_graphs == "on" else "0"
    os.environ["QD_GRAPH"] = requested
    set_time_ns = time.time_ns()
    original_argv = sys.argv
    sys.argv = ["examples.speed_benchmark.rigid_stress", *forwarded]
    worker_returned = False
    metadata = {
        "schema": "genesis.planar-graph-control.v1",
        "requested": args.cuda_graphs,
        "environment_before_override": environment_before,
        "environment_set_before_benchmark_import": requested,
        "environment_set_time_ns": set_time_ns,
        "native_modules_loaded_before_override": loaded_before,
        "forwarded_arguments": forwarded,
        "pid": os.getpid(),
        "scope": "Bounded wall-time screening with internal repeats, not formal paired primary timing. The process-local environment is set before Quadrants imports; no production graph cache is modified.",
    }
    try:
        metadata["worker_dispatch_time_ns"] = time.time_ns()
        runpy.run_module("examples.speed_benchmark.rigid_stress", run_name="__main__", alter_sys=True)
        worker_returned = True
    finally:
        sys.argv = original_argv
        kernel_module = sys.modules.get("quadrants.lang.kernel")
        enabled = vars(kernel_module).get("_GRAPH_ENABLED") if kernel_module is not None else None
        metadata.update(
            worker_returned_normally=worker_returned,
            worker_finished_time_ns=time.time_ns(),
            environment_after_worker=os.environ.get("QD_GRAPH"),
            quadrants_import_time_graph_enabled=enabled,
            passed=worker_returned and type(enabled) is bool and enabled == (requested == "1"),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(output.name + ".graph-control.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if worker_returned:
            result = json.loads(output.read_text())
            result["graph_control"] = metadata
            output.write_text(json.dumps(result, indent=2) + "\n")
    if not metadata["passed"]:
        raise RuntimeError("The imported Quadrants graph gate does not match the pre-import process-local request.")


if __name__ == "__main__":
    main()
