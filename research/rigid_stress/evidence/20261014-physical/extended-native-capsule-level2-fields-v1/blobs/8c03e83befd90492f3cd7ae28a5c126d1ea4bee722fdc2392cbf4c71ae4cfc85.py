"""Evaluate contact workspace initialization while preserving all valid contact work."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import contact, pipeline
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress import native_trajectory_oracle as oracle


def generate():
    source = Path("genesis/engine/solvers/rigid/stress/contact.py").read_text()
    function = next(
        item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef) and item.name == "func_anchor"
    )
    function.name = "func_anchor_valid"
    loop = function.body[0]
    assert isinstance(loop, ast.For) and isinstance(loop.body[-1], ast.If)
    assert ast.unparse(loop.body[0].targets[0]) == "contact_state.status[i_c, i_b]"
    valid = loop.body[-1]
    valid.body = [*loop.body[1:-1], *valid.body]
    loop.body = [loop.body[0], valid]
    generated = (
        "from genesis.engine.solvers.rigid.stress.contact import *\n\n"
        + ast.unparse(ast.fix_missing_locations(function))
        + "\n\n"
        + """@qd.kernel
def kernel_anchor_valid(source_epsilon: float, contact_state: StressContactState, surface_info: StressSurfaceInfo):
    func_anchor_valid(source_epsilon, contact_state, surface_info)
"""
    )
    cache = (
        Path(os.environ["RIGID_STRESS_DATA_ROOT"])
        / "cache"
        / "contact_initialization_trials"
        / os.environ["SLURM_JOB_ID"]
    )
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "contact_initialization_generated.py"
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location("contact_initialization_generated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, generated


def micro(module, remaining):
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(remaining)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True)
    for _ in range(args.warmup):
        workload.step()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    contacts, surface = entry.contacts, entry.surface.info
    epsilon = float(np.finfo(gs.np_float).eps)
    contact.kernel_anchor(epsilon, contacts, surface)
    valid = qd_to_numpy(contacts.valid)
    fields = (
        contacts.status,
        contacts.force_error,
        contacts.moment_error,
        contacts.candidate_count,
        contacts.evaluations,
        contacts.is_refined,
        contacts.center,
        contacts.anchor_face,
    )
    reference = [qd_to_numpy(field, copy=True)[valid].copy() for field in fields]
    rows = []
    for name, operation in (
        ("all_slots", lambda: contact.kernel_anchor(epsilon, contacts, surface)),
        ("valid_slots", lambda: module.kernel_anchor_valid(epsilon, contacts, surface)),
    ):
        operation()
        for expected, field in zip(reference, fields):
            np.testing.assert_array_equal(qd_to_numpy(field)[valid], expected)
        np.testing.assert_array_equal(qd_to_numpy(contacts.status)[~valid], 0)
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        row = {
            "name": name,
            "milliseconds": 1e3 * (time.perf_counter() - started) / args.repetitions,
            "valid_contact_fields_bit_equal": True,
            "absent_contact_status_zero": True,
            "valid_contacts": int(valid.sum()),
            "contact_workspace_slots": int(valid.size),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"envs": args.envs, "seed": args.seed, "reports": rows}, indent=2) + "\n")


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-slots", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    module, generated = generate()
    if args.micro:
        micro(module, remaining)
    else:
        if args.valid_slots:
            contact.func_anchor = module.func_anchor_valid
            pipeline.func_anchor = module.func_anchor_valid
        sys.argv = ["contact_initialization_override", *remaining]
        if args.oracle:
            oracle.main()
        else:
            benchmark.main()
    output.output.with_suffix(".variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "valid_slots": args.valid_slots,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "generated_source_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                "generated_source": generated,
                "note": "Every slot's diagnostic status is cleared. Other fit scratch fields are reset for every valid contact before anchor/friction checks, including zero loads. Absent slots retain unread scratch; reappearing contacts initialize it again. No contact/search/sample/budget changes.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
