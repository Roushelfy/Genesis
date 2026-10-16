"""Evaluate serial rigid postprocessing, complete stress and integration fusion."""

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid import rigid_solver as rigid
from genesis.engine.solvers.rigid.stress import pipeline, recovery
from research.rigid_stress import native_trajectory_oracle as oracle


def install(generated):
    tree = ast.parse(Path(pipeline.__file__).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "kernel_pipeline")
    function.name = "kernel_combined_post"
    arguments = (
        ast.parse(
            "def extra(constraint_state: array_class.ConstraintState, dyn_info: array_class.DynInfo, "
            "rigid_info: array_class.RigidInfo, rigid_config: qd.template(), noslip: qd.template(), "
            "is_backward: qd.template(), constraints_enabled: qd.template()): pass"
        )
        .body[0]
        .args.args
    )
    function.args.args.extend(arguments)
    first = ast.parse(
        "if qd.static(constraints_enabled):\n"
        "    func_resolve_post(dyn, collider, constraint_state, dyn_info, rigid_info, rigid_config, noslip, errno)"
    ).body
    last = ast.parse("func_step_2(dyn, constraint_state, dyn_info, rigid_info, rigid_config, is_backward, errno)").body
    function.body = first + function.body + last
    ast.fix_missing_locations(tree)
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(ast.unparse(tree) + "\n")
    namespace = {**pipeline.__dict__, "func_resolve_post": rigid.func_resolve_post, "func_step_2": rigid.func_step_2}
    exec(compile(tree, str(generated), "exec"), namespace)  # noqa: S102 - owned diagnostic source
    graph = namespace["kernel_combined_post"]
    original_init = recovery.RigidStressRecovery.__init__
    original_recover = recovery.RigidStressRecovery.recover
    original_contacts = rigid.kernel_resolve_stress_contacts
    original_integration = rigid.kernel_step_2
    owners, context, completed = {}, [], set()
    counts = {"combined_calls": 0, "deferred_contact_calls": 0, "removed_integration_calls": 0}

    def initialize(self, solver):
        original_init(self, solver)
        assert len(self.links) == 1 and self.links[0].fused_pipeline, "Probe requires one fused stress link."
        owners[id(solver.dyn_state)] = self

    def recover(self, substep):
        context.append(self)
        try:
            return original_recover(self, substep)
        finally:
            context.pop()

    def combined(*args):
        owner = context[-1]
        solver = owner.solver
        graph(
            *args,
            solver.constraint_solver.constraint_state,
            solver.dyn_info,
            solver.rigid_info,
            solver.rigid_config,
            solver._options.noslip_iterations > 0,
            solver._is_backward,
            not solver._disable_constraint,
        )
        completed.add(id(solver.dyn_state))
        counts["combined_calls"] += 1

    def contacts(*args):
        if id(args[0]) in owners:
            counts["deferred_contact_calls"] += 1
            return None
        return original_contacts(*args)

    def integrate(*args):
        identity = id(args[0])
        if identity in completed:
            completed.remove(identity)
            counts["removed_integration_calls"] += 1
            return None
        return original_integration(*args)

    recovery.RigidStressRecovery.__init__ = initialize
    recovery.RigidStressRecovery.recover = recover
    recovery.kernel_pipeline = combined
    rigid.kernel_resolve_stress_contacts = contacts
    rigid.kernel_step_2 = integrate
    return counts


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fused-post", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    generated = output.output.with_suffix(".postgraph.py")
    counts = install(generated) if args.fused_post else None
    sys.argv = ["native_postgraph_override", *remaining]
    (oracle.main if args.oracle else benchmark.main)()
    if counts is not None:
        assert counts["combined_calls"] == counts["removed_integration_calls"]
    output.output.with_suffix(".postgraph-variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "fused_post": args.fused_post,
                "dispatch_counts": counts,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "original_pipeline_sha256": hashlib.sha256(Path(pipeline.__file__).read_bytes()).hexdigest(),
                "original_rigid_sha256": hashlib.sha256(Path(rigid.__file__).read_bytes()).hexdigest(),
                "generated_postgraph_sha256": hashlib.sha256(generated.read_bytes()).hexdigest()
                if args.fused_post
                else None,
                "note": "Single-link research probe. The serial graph prepends the original contact-force postprocessing and appends the original rigid integration to the unchanged complete stress pipeline. Numerical order, expressions, trajectories, public resets and acceptance budgets are preserved. No production source file is edited.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
