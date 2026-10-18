"""Common warmed full-field prescribed-load timing protocol."""

import argparse
from pathlib import Path

from .common import identity, vars_json, write_json


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--loads", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--inverse-max-bytes", type=int)
    parser.add_argument("--geometry", choices=("egg", "sphere"), default="egg")
    return parser


def measure(args, solver):
    amplitudes = (0.8, 1.2, 1.0)
    warmup = [solver.solve(amplitude) for amplitude in amplitudes]
    rows = []
    for repeat in range(args.repetitions):
        cases = [solver.solve(amplitude) for amplitude in amplitudes]
        rows.append(
            {
                "repeat": repeat,
                "problems": len(cases),
                "seconds": sum(case["seconds"] for case in cases),
                "cases": cases,
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "same-prescribed-load-cost",
            "args": vars_json(args),
            "identity": identity(),
            "method": solver.description,
            "warmup": warmup,
            "repeats": rows,
            "scope": "device-resident prescribed force input to accepted full stress field; host field export excluded",
            "comparison_limit": "Common quadrature loads use each discretization's basis; native rigid reconstruction and deformable shell statics are different mechanical models.",
        },
    )
    solver.export(args.output)
