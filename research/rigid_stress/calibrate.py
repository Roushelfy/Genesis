"""Run explicit device ablations in isolated processes, preserving failed cases and measured memory limits."""

import argparse
import json
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    name: str
    arguments: tuple[str, ...]


def variants(suite: str) -> list[Variant]:
    temporal = ("--method", "temporal")
    if suite == "batches":
        return [Variant(f"direct-b{batch}", ("--envs", str(batch))) for batch in (1, 8, 32, 128, 512, 2048, 8192)]
    if suite == "history":
        return [
            Variant("direct", ()),
            Variant("prediction", (*temporal, "--history", "0")),
            Variant("history4", (*temporal, "--history", "4")),
            Variant("history8", (*temporal, "--history", "8")),
            Variant("padded", (*temporal, "--strategy", "padded")),
            Variant("adaptive", (*temporal, "--strategy", "adaptive")),
            Variant("rebuild-gram", (*temporal, "--rebuild-gram")),
            Variant("dof-major", (*temporal, "--layout", "dof-major")),
            Variant("cold-direct", ("--reset-every-frame", "--abrupt")),
            Variant("cold-history", (*temporal, "--reset-every-frame", "--abrupt")),
        ]
    if suite == "precision":
        return [
            Variant("direct64", ()),
            Variant("raw32-refine", (*temporal, "--precision", "32")),
            Variant("scaled32-refine", (*temporal, "--precision", "32", "--scaling")),
            Variant("scaled32-fallback", (*temporal, "--precision", "32", "--scaling", "--refinements", "0")),
            Variant("throughput64", (*temporal, "--profile", "throughput")),
            Variant("throughput32", (*temporal, "--precision", "32", "--scaling", "--profile", "throughput")),
        ]
    if suite == "chunks":
        return [Variant(f"chunk-{chunk}", (*temporal, "--chunk", str(chunk))) for chunk in (1, 2, 4, 8)]
    if suite == "peak-graph":
        return [
            Variant("fused-eager", ()),
            Variant("tensor-eager", ("--peak", "tensor")),
            Variant("fused-async-eager", ("--async-allocator",)),
            Variant("fused-graph", ("--graph",)),
            Variant("natural-order", ("--native-order", "natural")),
        ]
    raise ValueError("Explicit calibration suite required")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("batches", "history", "precision", "chunks", "peak-graph"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--scope", choices=("stress", "assembled-rhs"), default="assembled-rhs")
    parser.add_argument(
        "--reuse-factor", action="store_true", help="Reuse one immutable operator/factor across variants"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, memory_limit = [], False
    model, factor, shared_stream = None, None, None
    if args.reuse_factor:
        import cupy as cp
        from threadpoolctl import threadpool_limits

        from .benchmark_replay import main as benchmark
        from .cpu import EggConfig, EggRecoveryCPU
        from .cudss import SharedCuDSSFactor

        with threadpool_limits(limits=1):
            model = EggRecoveryCPU(
                1, EggConfig(level=args.level, ordering="column-nd", factor_backend="none"), history=0
            )
            shared_stream = cp.cuda.Stream(non_blocking=True)
            with shared_stream:
                factor = SharedCuDSSFactor(model.fem.k[model.fem.free][:, model.fem.free], 1)
    for variant in variants(args.suite):
        if memory_limit:
            rows.append({"variant": variant.name, "status": "skipped_after_measured_memory_limit"})
            continue
        command = [
            sys.executable,
            "-m",
            "research.rigid_stress.benchmark_replay",
            "--level",
            str(args.level),
            "--scope",
            args.scope,
            "--replay",
            str(args.replay),
            "--output",
            str(args.output / f"{variant.name}.json"),
        ]
        if args.suite != "batches":
            command.extend(("--envs", str(args.envs)))
        command.extend(variant.arguments)
        print(json.dumps({"variant": variant.name, "command": command}), flush=True)
        log = args.output / f"{variant.name}.log"
        with log.open("w") as stream:
            if args.reuse_factor:
                reuse = not any(
                    option in variant.arguments for option in ("--native-order", "--graph", "--async-allocator")
                )
                if not reuse and factor is not None:
                    factor.close()
                    factor = None
                    cp.get_default_memory_pool().free_all_blocks()
                with redirect_stdout(stream), redirect_stderr(stream), threadpool_limits(limits=1):
                    try:
                        benchmark(command[3:], model, factor if reuse else None, shared_stream)
                    except Exception:
                        traceback.print_exc()
                        code = 1
                    else:
                        code = 0
                cp.get_default_memory_pool().free_all_blocks()
            else:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
                code = result.returncode
        raw = log.read_text()
        memory_limit = (
            args.suite == "batches"
            and code != 0
            and any(marker in raw for marker in ("OutOfMemoryError", "MemoryError", "CUDA_ERROR_OUT_OF_MEMORY"))
        )
        row = {
            "variant": variant.name,
            "exit_code": code,
            "command": command,
            "status": "finished" if code == 0 else "failed",
            "measured_memory_limit": memory_limit,
        }
        rows.append(row)
        (args.output / "manifest.json").write_text(
            json.dumps(
                {
                    "arguments": {
                        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                    },
                    "runs": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(row), flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "runs": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
