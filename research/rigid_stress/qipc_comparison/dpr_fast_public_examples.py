"""Run the ordinary public Franka example, with an actual viewer on an owned X server when requested."""

import argparse
import ctypes
import ctypes.util
import json
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

GL_ENVIRONMENT = (
    "DISPLAY",
    "PYOPENGL_PLATFORM",
    "LIBGL_ALWAYS_SOFTWARE",
    "__GLX_VENDOR_LIBRARY_NAME",
    "MESA_LOADER_DRIVER_OVERRIDE",
    "LD_LIBRARY_PATH",
    "XDG_SESSION_TYPE",
    "WAYLAND_DISPLAY",
    "PYGLET_HEADLESS",
    "GS_HEADLESS",
    "CUDA_VISIBLE_DEVICES",
)


def _probe_guard(parser, output):
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Display probes require Slurm rtx-mid/image1_26.")
    if not output.resolve().is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side display-probe output.")


def _probe_child():
    parser = argparse.ArgumentParser(description="Create and draw a real OpenGL3.3 X window before any physics build.")
    parser.add_argument("--gl-probe-child", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _probe_guard(parser, args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "genesis.public-viewer-gl-probe-child.v1",
        "complete": False,
        "passed": False,
        "environment": {name: os.environ.get(name) for name in GL_ENVIRONMENT},
        "libraries": {name: ctypes.util.find_library(name) for name in ("GL", "OpenGL", "GLX", "EGL", "GLX_mesa")},
        "scope": "Real visible X window, GLSL330 triangle and the same PyOpenGL glVertexAttribPointer context lookup that failed in the public viewer. Genesis import only; no scene/physics/kernel compilation or simulation.",
    }
    write_json(args.output / "result.json", result)
    window = None
    try:
        # Match the public example's import order, without its multi-minute scene build.
        import genesis as gs

        result["genesis_imported_before_graphics_probe"] = gs.__file__
        import OpenGL
        import pyglet
        from OpenGL import GL, contextdata, platform

        result["imports"] = {"genesis": gs.__file__, "pyglet": pyglet.__file__, "OpenGL": OpenGL.__file__}
        result["platform_class"] = type(platform.PLATFORM).__name__
        result["pyglet_headless"] = pyglet.options["headless"]
        if pyglet.options["headless"]:
            raise RuntimeError("This probe requires a real X window, not a headless pbuffer.")
        config = pyglet.gl.Config(major_version=3, minor_version=3, depth_size=24, alpha_size=8, double_buffer=True)
        window = pyglet.window.Window(width=128, height=96, config=config, visible=True, caption="Genesis GL preflight")
        window.switch_to()
        window.dispatch_events()
        result["visible"] = window.visible
        result["pyopengl_current_context"] = int(platform.GetCurrentContext() or 0)
        result["pyopengl_contextdata_context"] = int(contextdata.getContext())
        result["GL"] = {
            name: GL.glGetString(value).decode("utf-8", errors="replace")
            for name, value in (
                ("vendor", GL.GL_VENDOR),
                ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
                ("shading_language", GL.GL_SHADING_LANGUAGE_VERSION),
            )
        }
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        vertices = (ctypes.c_float * 9)(-1, -1, 0, 3, -1, 0, -1, 3, 0)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, ctypes.sizeof(vertices), vertices, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, ctypes.c_void_p(0))
        result["failing_public_vertex_attrib_call_passed"] = True
        shaders = []
        for kind, source in (
            (
                GL.GL_VERTEX_SHADER,
                "#version 330 core\nlayout(location=0) in vec3 p; void main(){gl_Position=vec4(p,1);}",
            ),
            (GL.GL_FRAGMENT_SHADER, "#version 330 core\nout vec4 c; void main(){c=vec4(0.25,0.5,0.75,1);}"),
        ):
            shader = GL.glCreateShader(kind)
            GL.glShaderSource(shader, source)
            GL.glCompileShader(shader)
            if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(shader).decode(errors="replace"))
            shaders.append(shader)
        program = GL.glCreateProgram()
        for shader in shaders:
            GL.glAttachShader(program, shader)
        GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(program).decode(errors="replace"))
        GL.glUseProgram(program)
        GL.glViewport(0, 0, 128, 96)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glFinish()
        pixel = (ctypes.c_ubyte * 4)()
        GL.glReadPixels(64, 48, 1, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixel)
        actual = list(pixel)
        result["actual_glReadPixels_RGBA"] = actual
        result["expected_RGBA"] = [64, 128, 191, 255]
        window.flip()
        window.dispatch_events()
        result["passed"] = bool(
            window.visible
            and result["pyopengl_current_context"]
            and all(abs(a - b) <= 1 for a, b in zip(actual, result["expected_RGBA"]))
        )
        if not result["passed"]:
            raise RuntimeError("The actual GL3 window/pixel witness failed.")
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if window is not None:
            window.close()
        if Path("/proc/self/maps").is_file():
            result["loaded_graphics_library_maps"] = [
                line
                for line in Path("/proc/self/maps").read_text().splitlines()
                if any(name in line for name in ("libGL", "libEGL", "swrast", "libLLVM", "libX11"))
            ]
        result["complete"] = True
        write_json(args.output / "result.json", result)


def _display_probe():
    parser = argparse.ArgumentParser(description="Fresh-process GLX/Mesa environment probes without building physics.")
    parser.add_argument("--gl-probe", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _probe_guard(parser, args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "genesis.public-viewer-display-probe.v1",
        "complete": False,
        "passed": False,
        "identity": runtime_identity(),
        "worker": file_id(Path(__file__)),
        "original_environment": {name: os.environ.get(name) for name in GL_ENVIRONMENT},
        "variants": [],
        "scope": "Graphics-environment probe only; no simulation, production source changes, render throughput claim or fake/offscreen viewer.",
    }
    write_json(args.output / "result.json", result)
    executable = shutil.which("Xvfb")
    if executable is None:
        raise RuntimeError("No installed Xvfb is available in this allocation.")
    for name, command in (
        ("ldconfig", ["/sbin/ldconfig", "-p"]),
        ("mesa-packages", ["dpkg-query", "-W", "libglx-mesa0", "libgl1-mesa-dri", "libglvnd0"]),
    ):
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        (args.output / (name + ".txt")).write_text(completed.stdout + completed.stderr)
    read_fd, write_fd = os.pipe()
    with (args.output / "xvfb.log").open("x") as log:
        try:
            server = subprocess.Popen(
                [
                    executable,
                    "-displayfd",
                    str(write_fd),
                    "-screen",
                    "0",
                    "1024x768x24",
                    "+extension",
                    "GLX",
                    "+render",
                    "-noreset",
                    "-nolisten",
                    "tcp",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(write_fd,),
            )
        finally:
            os.close(write_fd)
        try:
            try:
                ready, _, _ = select.select([read_fd], [], [], 15.0)
                number = os.read(read_fd, 128).decode().strip() if ready else ""
            finally:
                os.close(read_fd)
            if not number.isdecimal():
                raise RuntimeError("Private Xvfb failed to start.")
            result["display"] = ":" + number
            result["xvfb_pid"] = server.pid
            for name, extra in (
                ("inherited", {}),
                ("glx", {"PYOPENGL_PLATFORM": "glx", "GS_HEADLESS": None, "PYGLET_HEADLESS": "false"}),
                (
                    "mesa-glx",
                    {
                        "PYOPENGL_PLATFORM": "glx",
                        "LIBGL_ALWAYS_SOFTWARE": "1",
                        "__GLX_VENDOR_LIBRARY_NAME": "mesa",
                        "GS_HEADLESS": None,
                        "PYGLET_HEADLESS": "false",
                    },
                ),
            ):
                environment = os.environ.copy()
                for key, value in extra.items():
                    if value is None:
                        environment.pop(key, None)
                    else:
                        environment[key] = value
                environment["DISPLAY"] = result["display"]
                with (args.output / (name + ".log")).open("x") as child_log:
                    command = [
                        sys.executable,
                        "-m",
                        "research.rigid_stress.qipc_comparison.dpr_fast_public_examples",
                        "--gl-probe-child",
                        "--output",
                        str(args.output / name),
                    ]
                    completed = subprocess.run(
                        command, env=environment, stdout=child_log, stderr=subprocess.STDOUT, timeout=90, check=False
                    )
                child_path = args.output / name / "result.json"
                child = json.loads(child_path.read_text()) if child_path.is_file() else None
                result["variants"].append(
                    {
                        "name": name,
                        "returncode": completed.returncode,
                        "environment_override": extra,
                        "receipt": file_id(child_path) if child_path.is_file() else None,
                        "result": child,
                    }
                )
                write_json(args.output / "result.json", result)
        finally:
            result["xvfb_alive_at_end"] = server.poll() is None
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            result["xvfb_returncode_after_cleanup"] = server.returncode
    result["complete"] = True
    result["passed"] = bool(
        result["xvfb_alive_at_end"]
        and any(row["returncode"] == 0 and row["result"] and row["result"]["passed"] for row in result["variants"])
    )
    write_json(args.output / "result.json", result)
    if not result["passed"]:
        raise RuntimeError("No real GL3 window variant passed; preserve the graphics diagnostics.")


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
    if "--gl-probe-child" in sys.argv:
        _probe_child()
    elif "--gl-probe" in sys.argv:
        _display_probe()
    else:
        main()
