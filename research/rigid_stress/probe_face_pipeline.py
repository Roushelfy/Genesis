"""Matched actual native pipeline trial with bounded face tasks and complete overflow fallback."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import quadrants as qd

import genesis as gs
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress import native_trajectory_oracle as oracle
from research.rigid_stress.probe_face_tasks import create_workspace, generate_fallback


def generate():
    _fallback, fallback_source = generate_fallback()
    root = Path("genesis/engine/solvers/rigid/stress")
    pipeline_source = (root / "pipeline.py").read_text()
    kernel = next(
        item
        for item in ast.parse(pipeline_source).body
        if isinstance(item, ast.FunctionDef) and item.name == "kernel_pipeline"
    )
    kernel.name = "kernel_face_pipeline"
    kernel.args.args += [
        ast.arg(arg="workspace", annotation=ast.Name(id="FaceTasks", ctx=ast.Load())),
        ast.arg(
            arg="fallback_counter",
            annotation=ast.Attribute(value=ast.Name(id="qd", ctx=ast.Load()), attr="Tensor", ctx=ast.Load()),
        ),
    ]
    index = next(
        i
        for i, item in enumerate(kernel.body)
        if isinstance(item, ast.Expr)
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Name)
        and item.value.func.id == "func_scatter_warp"
    )
    call = kernel.body[index].value
    call.func.id = "func_scatter_faces"
    call.args.append(ast.Name(id="workspace", ctx=ast.Load()))
    kernel.body[index + 1 : index + 1] = ast.parse(
        "if workspace.count[None] > workspace.tasks.shape[0]:\n"
        "    fallback_counter[None] += 1\n"
        "func_scatter_fallback(contacts, state, info, surface, cached_bounds, workspace)\n"
    ).body
    recovery_source = (root / "recovery.py").read_text()
    cls = next(
        item
        for item in ast.parse(recovery_source).body
        if isinstance(item, ast.ClassDef) and item.name == "RigidStressRecovery"
    )
    recover = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "recover")

    class Calls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "kernel_pipeline":
                node.func.id = "kernel_face_pipeline"
                node.args += ast.parse(
                    "[workspaces[id(entry.state)][0], workspaces[id(entry.state)][1]]", mode="eval"
                ).body.elts
            return node

    recover = Calls().visit(recover)
    header = (
        "from genesis.engine.solvers.rigid.stress.pipeline import *\n"
        "from genesis.engine.solvers.rigid.stress.recovery import *\n"
        "from research.rigid_stress.probe_face_tasks import FaceTasks, func_scatter_faces\n"
        "from scatter_fallback_generated import func_scatter_fallback\n"
        "workspaces = {}\n\n"
    )
    generated = (
        header
        + ast.unparse(ast.fix_missing_locations(kernel))
        + "\n\n"
        + ast.unparse(ast.fix_missing_locations(recover))
        + "\n"
    )
    directory = (
        Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "cache" / "face_pipeline" / os.environ.get("SLURM_JOB_ID", "local")
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "face_pipeline_generated.py"
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location("face_pipeline_generated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, {"pipeline": generated, "fallback": fallback_source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-tasks", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    generated = {}
    module = None
    if args.face_tasks:
        module, generated = generate()
        original_init = RigidStressRecovery.__init__

        def initialize(self, *init_args, **kwargs):
            original_init(self, *init_args, **kwargs)
            for entry in self.links:
                workspace = create_workspace(entry.contacts, entry.surface.info)
                counter = V(dtype=qd.i64, shape=())
                counter.fill(0)
                module.workspaces[id(entry.state)] = (workspace, counter)

        RigidStressRecovery.__init__ = initialize
        RigidStressRecovery.recover = module.recover
    sys.argv = ["native_trajectory_oracle" if args.oracle else "rigid_stress_benchmark", *remaining]
    (oracle.main if args.oracle else benchmark.main)()
    metadata = {
        "face_tasks": args.face_tasks,
        "oracle": args.oracle,
        "generated_sources": generated,
        "generated_sha256": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in generated.items()},
        "overflow_fallback_calls_including_warmup": sum(
            int(qd_to_numpy(counter)) for _, counter in module.workspaces.values()
        )
        if module
        else 0,
        "last_face_counts": [int(qd_to_numpy(workspace.count)) for workspace, _ in module.workspaces.values()]
        if module
        else [],
        "note": "Bounded task buffer of 32*B entries; all eligible faces counted. Overflow fully recomputes native scatter, with no load/contact/sample truncation. Complete solve/check/peak, observation, update, reset and fallback costs remain in ordinary live timing.",
    }
    output.output.with_suffix(".variant.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
