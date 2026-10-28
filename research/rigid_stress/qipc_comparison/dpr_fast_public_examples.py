"""Run the ordinary public Franka example, with an actual viewer on an owned X server when requested."""

import argparse
import os
import runpy
import select
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("volume", "shell"), required=True)
    parser.add_argument("--output-mode", choices=("max", "full"), required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=730)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Run this public-example verification in Slurm rtx-mid/image1_26.")
    args.output.mkdir(parents=True, exist_ok=False)
    module = "examples.rigid.franka_egg_automatic_stress"
    argv = [
        module,
        "--backend",
        args.backend,
        "--output-mode",
        args.output_mode,
        "--envs",
        str(args.envs),
        "--steps",
        str(args.steps),
    ]
    if args.viewer:
        argv.append("--viewer")
    result = {
        "schema": "genesis.automatic-dpr-public-example.v1",
        "identity": runtime_identity(),
        "argv": argv,
        "complete": False,
        "passed": False,
        "viewer_requested": args.viewer,
        "viewer_scope": "Real public viewer and scene stepping on a private Xvfb display."
        if args.viewer
        else "Ordinary public headless scene.",
        "primary_timing": False,
    }
    write_json(args.output / "result.json", result)
    previous_argv, previous_display = sys.argv, os.environ.get("DISPLAY")
    server, server_log = None, None
    started = time.perf_counter()
    try:
        if args.viewer:
            executable = shutil.which("Xvfb")
            if executable is None:
                raise RuntimeError("The allocated image has no Xvfb executable for the requested viewer verification.")
            read_fd, write_fd = os.pipe()
            server_log = (args.output / "xvfb.log").open("x")
            try:
                server = subprocess.Popen(
                    [executable, "-displayfd", str(write_fd), "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(write_fd,),
                )
            finally:
                os.close(write_fd)
            try:
                ready, _, _ = select.select([read_fd], [], [], 15.0)
                number = os.read(read_fd, 128).decode().strip() if ready else ""
            finally:
                os.close(read_fd)
            if not number.isdecimal():
                raise RuntimeError(f"Private Xvfb did not publish a display number (returncode={server.poll()}).")
            os.environ["DISPLAY"] = ":" + number
            result["private_display"] = os.environ["DISPLAY"]
            result["xvfb_pid"] = server.pid
            result["xvfb_executable"] = executable
        sys.argv = argv
        namespace = runpy.run_module(module, run_name="__main__")
        gs = namespace["gs"]
        source = Path(gs.__file__).resolve().parent.parent
        result["public_source"] = [
            file_id(source / relative)
            for relative in (
                "examples/rigid/franka_egg_automatic_stress.py",
                "examples/rigid/franka_egg_stress.py",
                "genesis/options/rigid_stress.py",
            )
        ]
        result["public_command_completed"] = True
        result["passed"] = True
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        sys.argv = previous_argv
        native = sys.modules.get("genesis")
        if native is not None:
            native.destroy()
            result["native_cleanup_completed"] = True
        if previous_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = previous_display
        if server is not None:
            result["xvfb_alive_at_example_end"] = server.poll() is None
            server.terminate()
            try:
                server.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5.0)
            result["xvfb_returncode_after_cleanup"] = server.returncode
        if server_log is not None:
            server_log.close()
        result["elapsed_seconds_including_build_and_viewer"] = time.perf_counter() - started
        result["complete"] = True
        if args.viewer:
            result["passed"] &= result.get("xvfb_alive_at_example_end", False)
        write_json(args.output / "result.json", result)
    if not result["passed"]:
        raise RuntimeError("The public example or its private viewer server did not complete.")


if __name__ == "__main__":
    main()
