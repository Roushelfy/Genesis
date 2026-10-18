"""Capture analysis inputs and submit sequential task groups as Slurm GPU arrays.

This stdlib-only login-node helper does not run scientific work. The source
checkout is pinned by a complete file inventory, and each allocation gets
isolated data-side caches. Tasks are module/argument lists, never shell snippets.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def inventory(root):
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]


def capture_source(source, output, stem):
    names = subprocess.check_output(
        ["git", "ls-files", "genesis", "pyproject.toml"], cwd=source, text=True
    ).splitlines()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    rows = [
        {
            "path": name,
            "bytes": (source / name).stat().st_size,
            "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest(),
        }
        for name in names
        if (source / name).is_file()
    ]
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", "genesis"], cwd=source)
    (output / f"{stem}.patch").write_bytes(patch)
    (output / f"{stem}.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "revision": revision,
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "files": rows,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tools = output / "analysis-source"
    here = Path(__file__).resolve().parent
    for rel in ("research/__init__.py", "research/rigid_stress/__init__.py"):
        dest = tools / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("")
    for path in here.parent.glob("*.py"):
        shutil.copy2(path, tools / "research/rigid_stress" / path.name)
    scripts = tools / "research/rigid_stress/scripts"
    scripts.mkdir()
    for path in (here.parent / "scripts").glob("*.py"):
        shutil.copy2(path, scripts / path.name)
    shutil.copytree(
        here, tools / "research/rigid_stress/shell_comparison", ignore=shutil.ignore_patterns("__pycache__")
    )
    groups = json.loads(args.tasks.read_text())
    (output / "tasks.json").write_text(json.dumps(groups, indent=2) + "\n")
    (output / "analysis-files.json").write_text(json.dumps(inventory(tools), indent=2) + "\n")
    (output / "logs").mkdir()
    source = args.source.resolve()
    capture_source(source, output, "source")
    for index, extra in enumerate(sorted({task["source"] for group in groups for task in group if "source" in task})):
        capture_source(Path(extra).resolve(), output, f"extra-source{index}")
    for group_id, group in enumerate(groups):
        run = output / f"group{group_id}"
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"task_run={shlex.quote(str(run))}",
            'test ! -e "$task_run"',
            'mkdir -p "$task_run/tmp" "$task_run/cache"',
            'export TMPDIR="$task_run/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1',
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
            'export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"',
            'export CUDA_CACHE_PATH="$task_run/cache/cuda" NUMBA_CACHE_DIR="$task_run/cache/numba"',
            'export MPLCONFIGDIR="$task_run/cache/matplotlib"',
            f"export PYTHONPATH={shlex.quote(str(tools) + ':' + str(source))}",
            f"cd {shlex.quote(str(source))}",
            'trap \'task_status=$?; printf "%s\\n" "$task_status" > "$task_run/returncode.txt"; exit "$task_status"\' EXIT',
        ]
        for task in group:
            command = [str(args.python), "-m", task["module"], *task["args"], "--output", str(run / task["name"])]
            task_source = str(Path(task.get("source", source)).resolve())
            invocation = shlex.join(command)
            if task_source != str(source):
                invocation = (
                    "( cd "
                    + shlex.quote(task_source)
                    + " && "
                    + shlex.join(["env", "PYTHONPATH=" + str(tools) + ":" + task_source, *command])
                    + " )"
                )
            lines.append(invocation + " > " + shlex.quote(str(run / (task["name"] + ".log"))) + " 2>&1")
        (output / f"workload-{group_id}.sh").write_text("\n".join(lines) + "\n")
    job = output / "job.sh"
    job.write_text(
        "#!/usr/bin/env bash\nset -eo pipefail\nsource /mnt/data/shared/config/env.sh\n"
        "GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash "
        + shlex.quote(str(output))
        + "/workload-${SLURM_ARRAY_TASK_ID}.sh\n"
    )
    command = [
        "sbatch",
        "--parsable",
        f"--array=0-{len(groups) - 1}" + (f"%{args.concurrency}" if args.concurrency is not None else ""),
        "--partition=rtx-mid",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        f"--mem={args.memory_gb}G",
        f"--time={args.minutes}",
        "--job-name=shell22-compare",
        f"--output={output}/logs/%A_%a.log",
        str(job),
    ]
    result = subprocess.check_output(command, text=True).strip()
    (output / "submission.json").write_text(json.dumps({"command": command, "job": result}, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
