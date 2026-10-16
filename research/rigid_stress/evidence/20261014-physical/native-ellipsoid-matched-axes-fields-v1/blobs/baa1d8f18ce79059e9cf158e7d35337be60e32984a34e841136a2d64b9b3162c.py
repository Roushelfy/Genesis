"""Measure matched warmed corner-kernel costs while retaining complete fields and an analytic affine control."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.solve import kernel_peak
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    expected_inputs = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.baseline_source, args.mesh]
    }
    name = "genesis.engine.solvers.rigid.stress.corner_benchmark_baseline"
    spec = importlib.util.spec_from_file_location(name, args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    options = RigidStressOptions(mesh=args.mesh, method="pcg", output_mode="full")
    model = StressModel(options)
    vertices = qd_to_numpy(model.info.vertices, copy=True)
    derivative = np.array([[1.2, -0.4, 0.3], [0.1, -0.6, 0.2], [-0.1, 0.7, 0.4]]) * 1e-6
    displacement = vertices @ derivative.T + [1e-6, -2e-6, 3e-6]
    mu = options.young / (2 * (1 + options.poisson))
    lame_lambda = options.young * options.poisson / ((1 + options.poisson) * (1 - 2 * options.poisson))
    tensor = mu * (derivative + derivative.T) + lame_lambda * np.trace(derivative) * np.eye(3)
    expected_components = tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    functions = {"baseline": baseline.kernel_peak, "difference_form": kernel_peak}
    rows = []
    for batch in args.batch:
        assert batch > 0
        state = model.create_state(batch, "full")
        state.displacement.from_numpy(np.broadcast_to(displacement[:, None], (len(vertices), batch, 3)).copy())
        state.valid.fill(True)
        samples = {name: [] for name in functions}
        fields = {}
        peaks = {}
        for name, function in functions.items():
            for _iteration in range(5):
                function(options.young, options.poisson, state, model.info)
            qd.sync()
            fields[name] = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)
            peaks[name + "_peak_Pa"] = qd_to_numpy(state.peak, copy=True)
            error = abs(fields[name] - expected_components)
            budget = 1e-5 + 1e-9 * abs(expected_components)
            assert np.all(error <= budget)
        for i_repeat in range(7):
            order = list(functions) if i_repeat % 2 == 0 else list(reversed(functions))
            for name in order:
                started = perf_counter()
                for _iteration in range(10):
                    functions[name](options.young, options.poisson, state, model.info)
                qd.sync()
                samples[name].append((perf_counter() - started) / 10)
        difference = abs(fields["baseline"] - fields["difference_form"])
        row = {
            "batch": batch,
            "tetrahedra": model.info.elements.shape[0],
            "warmed_seconds_per_launch_samples": samples,
            "warmed_seconds_per_launch_median": {name: statistics.median(values) for name, values in samples.items()},
            "whole_field_maximum_absolute_change_Pa": float(difference.max()),
            "complete_affine_component_control_passed": True,
            "scope": "Identical full-output corner tensor/von-Mises/global-peak kernels, including Python launch and synchronization; no load fit or elastic solve",
        }
        row["difference_to_baseline_median_ratio"] = (
            row["warmed_seconds_per_launch_median"]["difference_form"]
            / row["warmed_seconds_per_launch_median"]["baseline"]
        )
        rows.append(row)
        np.savez_compressed(args.output / f"batch{batch}-fields.npz", **fields, **peaks)
        print(json.dumps(row), flush=True)
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in expected_inputs.items()
    )
    report = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": expected_inputs,
        "goal_completion": False,
        "limitations": "A controlled same-device changed-stage cost guard on the original low mesh and an analytic affine displacement field. Exact old source is loaded in its own module; production modules remain intact. Complete original/new fields are retained. This excludes recovery/contact/rigid/policy work and establishes neither physically converged accuracy nor a warmed application accuracy/cost range.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
