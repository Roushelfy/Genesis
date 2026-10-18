"""Recompile one declared pipeline variant and relink immutable original objects into a separate native library."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


def output_at(command, destination):
    command = command.copy()
    command[command.index("-o") + 1] = str(destination)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-build", type=Path, required=True)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ninja", type=Path, required=True)
    parser.add_argument("--snapshot-project-includes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    relative = Path("qipc/_src/native/solver/sim_engine_pipeline.cu")
    records = json.loads((args.base_build / "compile_commands.json").read_text())
    original = next(row for row in records if row["file"] == str(args.base_source / relative))
    compile_command = shlex.split(original["command"])
    compile_command = [str(args.source / relative) if item == original["file"] else item for item in compile_command]
    pipeline_object = args.output / "pipeline.cu.o"
    compile_command = output_at(compile_command, pipeline_object)
    if args.snapshot_project_includes:
        base_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.base_source, text=True).strip()
        variant_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip()
        assert base_revision == variant_revision
        assert not subprocess.check_output(["git", "diff", "HEAD", "--", "qipc"], cwd=args.base_source)
        headers = args.output / "headers"
        headers.mkdir()
        includes = []
        for index, item in enumerate(compile_command):
            prefix = (
                "-I" if item.startswith("-I") else "" if index and compile_command[index - 1] == "-isystem" else None
            )
            if prefix is None:
                continue
            original_path = Path(item[len(prefix) :])
            if not original_path.is_absolute() or not original_path.is_relative_to(args.base_source):
                continue
            destination = headers / f"include{len(includes)}"
            shutil.copytree(
                original_path,
                destination,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git", "*.so", "*.o", "*.a", "__pycache__"),
            )
            compile_command[index] = prefix + str(destination)
            includes.append(
                {
                    "original": str(original_path),
                    "snapshot": str(destination),
                    "files": [
                        file_id(path)
                        for path in sorted(destination.rglob("*"))
                        if path.is_file() and not path.is_symlink()
                    ],
                }
            )
        write_json(args.output / "project-includes.json", includes)
        assert (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.base_source, text=True).strip()
            == base_revision
        )
    commands = subprocess.check_output(
        [str(args.ninja), "-C", str(args.base_build), "-t", "commands", "qipc_solver_ext"], text=True
    ).splitlines()
    device_command = shlex.split(next(line for line in commands if " -dlink " in line))
    original_device = device_command[device_command.index("-o") + 1]
    device_object = args.output / "device-link.o"
    device_command = output_at(device_command, device_object)
    original_pipeline = str(Path(original["output"]).relative_to(args.base_build))
    device_command = [str(pipeline_object) if item == original_pipeline else item for item in device_command]
    link_command = shlex.split(commands[-1])
    assert link_command[:2] == [":", "&&"] and link_command[-2:] == ["&&", ":"]
    link_command = link_command[2:-2]
    link_command = [
        str(pipeline_object) if item == original_pipeline else str(device_object) if item == original_device else item
        for item in link_command
    ]
    library = args.output / "qipc_solver_ext.abi3.so"
    link_command = output_at(link_command, library)
    assert all(
        token not in ("&&", ";", "|")
        for command in (compile_command, device_command, link_command)
        for token in command
    )
    assert "-MD" not in compile_command and "-MF" not in compile_command
    base_library = args.base_build / "qipc_solver_ext.abi3.so"
    base_identity = file_id(base_library)
    dependencies = sorted(
        {
            args.base_build / item
            for item in link_command
            if item.endswith((".o", ".a")) and not Path(item).is_absolute()
        }
    )
    write_json(args.output / "reused-objects.json", [file_id(path) for path in dependencies])
    receipt = {
        "identity": runtime_identity(),
        "base_library": base_identity,
        "base_compile_commands": file_id(args.base_build / "compile_commands.json"),
        "original_translation_unit": file_id(args.base_source / relative),
        "variant_translation_unit": file_id(args.source / relative),
        "commands": [compile_command, device_command, link_command],
        "scope": "All physical energies, gradients, Hessians and other original objects are reused unchanged. Only the declared line-search comparison translation unit is recompiled; CUDA device linking and host linking write fresh owned files. This is a numerical diagnostic variant, not an unmodified baseline or a qualified reference.",
    }
    write_json(args.output / "commands.json", receipt)
    environment = os.environ.copy()
    toolkit_bin = Path(compile_command[0]).parent
    environment["PATH"] = str(toolkit_bin) + ":" + environment["PATH"]
    environment["LD_LIBRARY_PATH"] = str(toolkit_bin.parent / "lib64") + ":" + environment.get("LD_LIBRARY_PATH", "")
    for name, command in zip(("compile", "device-link", "link"), receipt["commands"]):
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=args.base_build, env=environment, stdout=log, stderr=subprocess.STDOUT)
        receipt[name + "_returncode"] = result.returncode
        write_json(args.output / "result.json", receipt)
        if result.returncode:
            raise SystemExit(result.returncode)
    assert file_id(base_library) == base_identity, "Original native library was modified"
    environment["PYTHONPATH"] = str(args.source)
    with (args.output / "import.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import qipc; import qipc._src.native.solver.qipc_solver_ext as ext; print(qipc.__file__); print(ext.__file__)",
            ],
            cwd=args.source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    receipt.update(native=file_id(library), import_returncode=result.returncode)
    write_json(args.output / "result.json", receipt)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
