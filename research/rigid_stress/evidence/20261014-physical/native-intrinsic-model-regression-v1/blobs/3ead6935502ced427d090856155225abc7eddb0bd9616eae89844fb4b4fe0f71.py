"""Compare native selected-state copies and duplicate stress-notice suppression."""

import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers import base_solver
from genesis.engine.solvers.rigid import rigid_solver
from research.rigid_stress import native_trajectory_oracle as oracle


def configure(native_state, deduplicate, generated_path):
    source = textwrap.dedent(inspect.getsource(rigid_solver.RigidSolver.set_state))
    tree = ast.parse(source)
    changed = 0
    if native_state:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "gs.use_zerocopy" in ast.unparse(node.test):
                node.test = ast.Constant(value=False)
                changed += 1
        assert changed == 1, changed
        ast.fix_missing_locations(tree)
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(ast.unparse(tree) + "\n")
        namespace = rigid_solver.__dict__.copy()
        exec(compile(tree, str(generated_path), "exec"), namespace)  # noqa: S102 - owned diagnostic source
        rigid_solver.RigidSolver.set_state = namespace["set_state"]
    if deduplicate:
        original_init = FrankaEgg.__init__

        def initialize(workload, *args, **kwargs):
            original_init(workload, *args, **kwargs)
            recovery = workload.scene.rigid_solver.stress_recovery
            if recovery is None:
                return
            subscribers = getattr(recovery, "subscribers", None) or [recovery.subscriber]
            for subscriber in subscribers:
                callback = subscriber.callback

                def notify(change, envs_idx, callback=callback):
                    if (
                        change is base_solver.StateChange.DYNAMICS
                        and base_solver.StateChange.GEOMETRY in recovery.solver._mutation_changes
                    ):
                        return
                    callback(change, envs_idx)

                subscriber.callback = notify

        FrankaEgg.__init__ = initialize
    return {
        "native_state_restore": native_state,
        "deduplicate_stress_reset": deduplicate,
        "original_set_state_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "generated_source_path": str(generated_path) if native_state else None,
        "generated_source_sha256": hashlib.sha256(generated_path.read_bytes()).hexdigest() if native_state else None,
    }


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native-state-restore", action="store_true")
    parser.add_argument("--deduplicate-stress-reset", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    generated = output.output.with_suffix(".set-state.py")
    configuration = configure(args.native_state_restore, args.deduplicate_stress_reset, generated)
    sys.argv = ["native_reset_override", *remaining]
    (oracle.main if args.oracle else benchmark.main)()
    output.output.with_suffix(".reset-variant.json").write_text(
        json.dumps(
            {
                **configuration,
                "command": command,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "note": "Same public Scene.reset, saved state, native cache clear, forward kinematics, lifecycle notices, timestep, controls, contact model and acceptance. Native-state trial selects the existing Quadrants copy branch. Notice trial suppresses only a redundant dynamics clear when that same solver mutation also emits geometry. All actual reset work remains timed.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
