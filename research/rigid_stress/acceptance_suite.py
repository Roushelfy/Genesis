"""Run selected reproducible checks with explicit artifact paths and fail on the first rejected check."""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class CheckCommand:
    name: str
    module: str
    arguments: tuple[str, ...] = ()


def selected_commands(gpu: bool, native: bool, grasp: bool, physical: bool) -> list[CheckCommand]:
    commands = [
        CheckCommand(name, f"research.rigid_stress.check_{name}") for name in ("cpu", "mechanics", "wrench", "history")
    ]
    if gpu:
        commands.extend(
            (
                CheckCommand(
                    "gpu-fields",
                    "research.rigid_stress.check_gpu",
                    ("--inertia", "quadratic", "--body-products", "fused", "--sparse-layout", "C"),
                ),
                CheckCommand("gpu-pressure", "research.rigid_stress.check_pressure", ("--force-line",)),
                CheckCommand("gpu-temporal", "research.rigid_stress.check_temporal_gpu"),
                CheckCommand("gpu-substeps", "research.rigid_stress.check_live"),
            )
        )
    if native:
        commands.extend(
            (
                CheckCommand("native-direct", "research.rigid_stress.check_cudss"),
                CheckCommand(
                    "native-temporal", "research.rigid_stress.check_temporal_gpu", ("--factor-backend", "cudss")
                ),
                CheckCommand("native-graph", "research.rigid_stress.check_graph_recovery"),
            )
        )
    if grasp:
        commands.extend(
            (
                CheckCommand("panda-varied32", "research.rigid_stress.check_panda", ("--temporal",)),
                CheckCommand(
                    "panda-nominal-substeps",
                    "research.rigid_stress.check_panda",
                    (
                        "--envs",
                        "4",
                        "--steps",
                        "650",
                        "--nominal",
                        "--synchronous",
                        "--substeps",
                        "4",
                        "--partial-reset-step",
                        "107",
                    ),
                ),
            )
        )
    if physical:
        for anchor in ("raw", "force-line"):
            commands.append(
                CheckCommand(
                    f"physical-{anchor}",
                    "research.rigid_stress.convergence",
                    (
                        "--levels",
                        "4",
                        "5",
                        "6",
                        "--factor-backend",
                        "cholmod",
                        "--ordering",
                        "column-nd",
                        "--footprint-anchor",
                        anchor,
                    ),
                )
            )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Artifact directory on the project's data volume")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--native", action="store_true", help="Also test installed cuDSS and captured recovery")
    parser.add_argument(
        "--grasp", action="store_true", help="Also run actual coarse Panda CPU-oracle integration checks"
    )
    parser.add_argument("--physical", action="store_true", help="Also run expensive full levels 4/5/6 CPU convergence")
    args = parser.parse_args()
    if (args.native or args.grasp) and not args.gpu:
        parser.error("Native and actual grasp checks require --gpu")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"passed": False, "arguments": {**vars(args), "output": str(args.output)}, "checks": []}
    manifest_path = args.output / "suite.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for check in selected_commands(args.gpu, args.native, args.grasp, args.physical):
        artifact = args.output / check.name / "result.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", check.module, *check.arguments, "--output", str(artifact)]
        log = artifact.with_suffix(".log")
        print(json.dumps({"check": check.name, "command": command, "log": str(log)}), flush=True)
        start = perf_counter()
        with log.open("w") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        manifest["checks"].append(
            {
                "name": check.name,
                "command": command,
                "return_code": process.returncode,
                "wall_seconds": perf_counter() - start,
                "artifact": str(artifact),
                "log": str(log),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"Check {check.name} rejected; inspect {log}")
    manifest["passed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(manifest["checks"]), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
