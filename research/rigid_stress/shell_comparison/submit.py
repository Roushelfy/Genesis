"""Capture analysis inputs and submit sequential task groups as Slurm GPU arrays.

This stdlib-only login-node helper does not run scientific work. The source
checkout is pinned by a complete file inventory. Each allocation gets isolated
caches, optionally on node-local temporary storage; scientific outputs/logs
remain at the explicit durable output path. Tasks are module/argument lists,
never shell snippets.
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


def runtime_output_path(path, allow_home=False):
    """Honor the storage policy, including compatibility links into data."""
    output = path.resolve()
    if output.is_relative_to(Path.home().resolve()) and not allow_home:
        raise ValueError(
            f"Runtime output resolves under the user home directory: {output}. "
            "Choose a writable project data directory. A full/unwritable data "
            "mount must be reported; use --allow-home-output only for an explicitly user-authorized exception."
        )
    return output


def capture_source(source, output, stem):
    package = "qipc" if (source / "qipc").is_dir() else "genesis"
    paths = [package, "pyproject.toml", "CMakeLists.txt", "external/GKlib", "external/METIS"]
    if package == "genesis":
        paths.extend(
            (
                "examples/rigid/franka_egg_stress.py",
                "examples/rigid/egg_stress_controller.py",
                "examples/speed_benchmark",
                "examples/rigid/assets/hollow_egg",
            )
        )
    names = subprocess.check_output(
        ["git", "ls-files", *paths],
        cwd=source,
        text=True,
    ).splitlines()
    if not names:
        raise RuntimeError(f"Empty tracked-source inventory; source checkout is not ready: {source}")
    staged = subprocess.check_output(["git", "ls-files", "--stage", *paths], cwd=source, text=True).splitlines()
    gitlinks = {row.split("\t", 1)[1]: row.split()[1] for row in staged if row.startswith("160000 ")}
    names = [name for name in names if name not in gitlinks]
    if not names:
        raise RuntimeError(f"No tracked source files at {source}")
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete tracked-source checkout at {source}: {missing[:10]}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    rows = [
        {
            "path": name,
            "bytes": (source / name).stat().st_size,
            "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest(),
        }
        for name in names
    ]
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *paths], cwd=source)
    (output / f"{stem}.patch").write_bytes(patch)
    (output / f"{stem}.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "revision": revision,
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "files": rows,
                "gitlinks": gitlinks,
                "submodule_status": subprocess.check_output(["git", "submodule", "status"], cwd=source, text=True)
                if package == "qipc"
                else None,
            },
            indent=2,
        )
        + "\n"
    )


def workload_preamble(run):
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"task_run={shlex.quote(str(run))}",
        'test ! -e "$task_run"',
        'mkdir -p "$task_run"',
        "task_scratch=''",
        'trap \'task_status=$?; printf "%s\\n" "$task_status" > "$task_run/returncode.txt"; if test -n "$task_scratch"; then rm -rf -- "$task_scratch"; fi; exit "$task_status"\' EXIT',
        "trap 'exit 143' TERM",
        "trap 'exit 130' INT",
        "trap 'exit 129' HUP",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--dependency", help="Explicit Slurm dependency, e.g. afterok:12345 or afterany:12345:12346")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--exclude-nodes", help="Job-specific Slurm exclusion list; leaves cluster configuration unchanged")
    parser.add_argument("--cpu-only", action="store_true", help="Request no GPU for CPU analysis in rtx-mid")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--kernel-profiler", action="store_true", help="Enable Quadrants profiling in separate diagnostic runs"
    )
    parser.add_argument(
        "--allow-home-output", action="store_true", help="Record an explicitly user-authorized home-output exception"
    )
    parser.add_argument(
        "--node-local-cache", action="store_true", help="Use and remove private compute-node /tmp caches"
    )
    args = parser.parse_args()
    output = runtime_output_path(args.output, args.allow_home_output)
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
    qipc_analysis = here.parent / "qipc_comparison"
    if qipc_analysis.is_dir():
        shutil.copytree(
            qipc_analysis, tools / "research/rigid_stress/qipc_comparison", ignore=shutil.ignore_patterns("__pycache__")
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
        lines = workload_preamble(run)
        if args.node_local_cache:
            lines.extend(
                (
                    "task_scratch=$(mktemp -d /tmp/gs-shell-compare.XXXXXX)",
                    'task_cache="$task_scratch/cache"',
                    'task_tmp="$task_scratch/tmp"',
                    'printf "%s\\n" "$task_scratch" > "$task_run/node-local-scratch.txt"',
                    'df -Pk "$task_scratch" > "$task_run/node-local-filesystem.txt"',
                )
            )
        else:
            lines.extend(('task_cache="$task_run/cache"', 'task_tmp="$task_run/tmp"'))
        lines.extend(
            [
                'mkdir -p "$task_tmp" "$task_cache"',
                'export TMPDIR="$task_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1',
                "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                'export GS_CACHE_FILE_PATH="$task_cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_cache/quadrants"',
                'export CUDA_CACHE_PATH="$task_cache/cuda" NUMBA_CACHE_DIR="$task_cache/numba"',
                'export MPLCONFIGDIR="$task_cache/matplotlib"',
                'export QIPC_RTC_CACHE="$task_cache/qipc_rtc"',
                f"export PYTHONPATH={shlex.quote(str(tools) + ':' + str(source))}",
                f"cd {shlex.quote(str(source))}",
            ]
        )
        if args.kernel_profiler:
            lines.append("export QD_KERNEL_PROFILER=1")
        if args.keep_going:
            lines.append("campaign_status=0")
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
            invocation += " > " + shlex.quote(str(run / (task["name"] + ".log"))) + " 2>&1"
            if args.keep_going:
                status_path = shlex.quote(str(run / (task["name"] + ".returncode.txt")))
                lines.extend(
                    (
                        "if " + invocation + "; then",
                        "  printf '%s\\n' 0 > " + status_path,
                        "else",
                        "  task_exit=$?",
                        "  printf '%s\\n' \"$task_exit\" > " + status_path,
                        "  campaign_status=1",
                        "fi",
                    )
                )
            else:
                lines.append(invocation)
        if args.keep_going:
            lines.append('exit "$campaign_status"')
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
        # This runner requires fresh output directories and has no restart
        # protocol. Requeueing would fail its guard and overwrite Slurm logs.
        "--no-requeue",
        "--gres=none" if args.cpu_only else "--gres=gpu:1",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.memory_gb}G",
        f"--time={args.minutes}",
        "--job-name=shell22-compare",
        f"--output={output}/logs/%A_%a.log",
        str(job),
    ]
    if args.dependency:
        command.insert(-1, f"--dependency={args.dependency}")
    if args.exclude_nodes:
        command.insert(-1, f"--exclude={args.exclude_nodes}")
    result = subprocess.check_output(command, text=True).strip()
    (output / "submission.json").write_text(
        json.dumps(
            {
                "command": command,
                "job": result,
                "dependency": args.dependency,
                "exclude_nodes": args.exclude_nodes,
                "node_local_cache": args.node_local_cache,
                "kernel_profiler": args.kernel_profiler,
                "allow_home_output": args.allow_home_output,
            },
            indent=2,
        )
        + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
