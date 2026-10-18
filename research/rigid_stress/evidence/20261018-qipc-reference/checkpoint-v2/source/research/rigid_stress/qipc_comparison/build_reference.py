"""Build the pinned original QIPC outside its source checkout, without changing an environment."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def file_id(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--cmake", type=Path, required=True)
    parser.add_argument("--ninja", type=Path, required=True)
    parser.add_argument("--libdwarf", type=Path, required=True)
    parser.add_argument("--zstd", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reuse-build", type=Path)
    parser.add_argument("--with-viewer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build = args.reuse_build or args.output / "native-build"
    if args.reuse_build is None:
        build.mkdir()
    else:
        assert (build / "CMakeCache.txt").is_file()
        shutil.copy2(build / "CMakeCache.txt", args.output / "previous-CMakeCache.txt")
        shutil.copy2(build / "compile_commands.json", args.output / "previous-compile_commands.json")
    environment = os.environ.copy()
    environment["PATH"] = str(args.toolkit / "bin") + ":" + str(Path(sys.executable).parent) + ":" + environment["PATH"]
    environment["LD_LIBRARY_PATH"] = str(args.toolkit / "lib64") + ":" + environment.get("LD_LIBRARY_PATH", "")
    nanobind = subprocess.check_output([sys.executable, "-m", "nanobind", "--cmake_dir"], text=True).strip()
    configure = [
        str(args.cmake),
        "-S",
        str(args.source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_MAKE_PROGRAM={args.ninja}",
        "-DCMAKE_CXX_COMPILER=/usr/bin/c++",
        f"-DCMAKE_CUDA_COMPILER={args.toolkit / 'bin/nvcc'}",
        f"-DCUDAToolkit_ROOT={args.toolkit}",
        "-DCMAKE_CUDA_ARCHITECTURES=120-real",
        "-DQIPC_CUDA_ARCHITECTURES=120-real",
        "-DQIPC_RUNTIME_CHECK=ON",
        "-DQIPC_BUILD_VIEWER=" + ("ON" if args.with_viewer else "OFF"),
        "-DQIPC_GENERATE_STUBS=OFF",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dnanobind_DIR={nanobind}",
        f"-DQIPC_CCCL_SOURCE_DIR={args.toolkit / 'include'}",
        f"-DFETCHCONTENT_SOURCE_DIR_LIBDWARF={args.libdwarf}",
        f"-DFETCHCONTENT_SOURCE_DIR_ZSTD={args.zstd}",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON",
    ]
    compile_command = [
        str(args.cmake),
        "--build",
        str(build),
        "--target",
        "qipc_solver_ext",
        "--parallel",
        str(args.jobs),
    ]
    if args.with_viewer:
        target = compile_command.index("qipc_solver_ext")
        compile_command.insert(target + 1, "qipc_viewer_ext")
    receipt = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "python": sys.executable,
        "configure": configure,
        "compile": compile_command,
        "toolkit": str(args.toolkit),
        "native_import_test": "pending",
    }
    (args.output / "commands.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for name, command in (("configure", configure), ("compile", compile_command)):
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        receipt[f"{name}_returncode"] = result.returncode
        (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(result.returncode)
    library = build / "qipc_solver_ext.abi3.so"
    assert library.is_file()
    runtime = args.output / "runtime"
    shutil.copytree(
        args.source / "qipc", runtime / "qipc", symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    destination = runtime / "qipc/_src/native/solver" / library.name
    shutil.copy2(library, destination)
    if args.with_viewer:
        shutil.copy2(build / "qipc_viewer_ext.abi3.so", runtime / "qipc/_src/native/viewer/qipc_viewer_ext.abi3.so")
    environment["PYTHONPATH"] = str(runtime)
    test = [
        sys.executable,
        "-c",
        "import qipc; import qipc._src.native.solver.qipc_solver_ext as ext; print(qipc.__file__); print(ext.__file__)",
    ]
    with (args.output / "import.log").open("w") as log:
        result = subprocess.run(test, cwd=runtime, env=environment, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(
        native=file_id(destination),
        native_import_test=result.returncode,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        runtime=str(runtime),
    )
    (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
