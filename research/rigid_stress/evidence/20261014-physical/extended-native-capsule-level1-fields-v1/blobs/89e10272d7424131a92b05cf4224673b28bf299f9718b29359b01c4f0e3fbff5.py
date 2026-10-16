"""Measure deriving current pressure Gram from the already integrated P2 moments."""

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
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress import probe_contact_moments as reference


def derive(module_source):
    module = ast.parse(module_source)
    prepare = next(
        item for item in module.body if isinstance(item, ast.FunctionDef) and item.name == "func_prepare_moments"
    )

    class DeriveGram(ast.NodeTransformer):
        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == "gram":
                return None
            return self.generic_visit(node)

        def visit_For(self, node):
            self.generic_visit(node)
            if isinstance(node.target, ast.Tuple) and isinstance(node.body[0], ast.Assign):
                target = node.body[0].targets[0]
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "gram"
                ):
                    return None
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "lane"
            ):
                node.body = (
                    ast.parse("""
gram = qd.Matrix.zero(gs.qd_float, 3, 3)
for i_local in qd.static(range(6)):
    node = info.surface_nodes[i_f, i_local]
    relative = (info.vertices[node] - contacts.center[i_c, i_b]) / contacts.radius[i_c, i_b]
    coordinate = qd.Vector([1.0, relative.dot(first), relative.dot(second)])
    row = qd.Vector([load[i_local, a] for a in qd.static(range(3))])
    gram += coordinate.outer_product(row)
""").body
                    + node.body
                )
            return node

    DeriveGram().visit(prepare)
    return ast.unparse(ast.fix_missing_locations(module)) + "\n"


def import_generated(source):
    cache = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "cache/contact_moment_gram_trials" / os.environ["SLURM_JOB_ID"]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "contact_moment_gram_generated.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("contact_moment_gram_generated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def micro(original_module, derived_module, remaining):
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(remaining)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True)
    for _ in range(args.warmup):
        workload.step()
    recovery = workload.scene.rigid_solver.stress_recovery
    entry = recovery.links[0]
    original_cache = reference.create_cache(original_module, recovery)
    derived_cache = reference.create_cache(derived_module, recovery)
    inputs = (
        float(np.finfo(float).eps),
        entry.contacts,
        entry.state,
        entry.model.info,
        entry.surface.info,
        entry.scatter,
    )
    operations = (
        ("native_fit_scatter", lambda: original_module.kernel_contacts_original(*inputs)),
        ("moments_with_integrated_gram", lambda: original_module.kernel_contact_moments(*inputs, original_cache)),
        ("moments_with_derived_gram", lambda: derived_module.kernel_contact_moments(*inputs, derived_cache)),
    )
    operations[0][1]()
    expected_force = qd_to_numpy(entry.state.force, copy=True)
    expected_status = qd_to_numpy(entry.contacts.status, copy=True)
    entry.model.recover(entry.omega, entry.state, surface_load=True)
    expected_peak = qd_to_numpy(entry.state.peak, copy=True)
    budget = np.maximum(1e-11, 1e-8 * np.linalg.norm(expected_force, axis=(0, 2)))
    rows = []
    for name, operation in operations:
        operation()
        np.testing.assert_array_equal(qd_to_numpy(entry.contacts.status), expected_status)
        difference = np.linalg.norm(qd_to_numpy(entry.state.force) - expected_force, axis=(0, 2))
        assert (difference <= budget).all(), (name, (difference / budget).max())
        entry.model.recover(entry.omega, entry.state, surface_load=True)
        assert qd_to_numpy(entry.state.valid).all()
        np.testing.assert_allclose(qd_to_numpy(entry.state.peak), expected_peak, rtol=1e-4, atol=1e-3)
        operation()
        qd.sync()
        start = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        row = {
            "name": name,
            "milliseconds": 1e3 * (time.perf_counter() - start) / args.repetitions,
            "force_difference_budget_ratio_max": float((difference / budget).max()),
            "force_error_max_N": float(qd_to_numpy(entry.contacts.force_error).max()),
            "moment_error_max_Nm": float(qd_to_numpy(entry.contacts.moment_error).max()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"envs": args.envs, "seed": args.seed, "reports": rows}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--micro", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    original, source = reference.generate()
    derived_source = derive(source)
    derived = import_generated(derived_source)
    if args.micro:
        micro(original, derived, remaining)
    else:
        reference.generate = lambda: (derived, derived_source)
        reference.main()
    output.output.with_suffix(".gram-variant.json").write_text(
        json.dumps(
            {
                "command": [sys.executable, *sys.argv],
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reference_probe_sha256": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
                "original_generated_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "derived_generated_sha256": hashlib.sha256(derived_source.encode()).hexdigest(),
                "derived_generated_source": derived_source,
                "note": "Exact P2 linear completeness derives Gram from the same current Q10 load moments; no sample, footprint, pressure law or final acceptance budget changes.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
