"""Compare four complete stationary spatial histories at ten declared common physical times."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .compare_recovered_spatial_state import main as compare_state
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=4, required=True)
    parser.add_argument("--fine-audit", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.sample_every > 1
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, geometries, physical = {}, [], None
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config_path, report_path, mesh = [
            run / "results" / name for name in ["configuration.json", "summary.json", "reference_geometry.npz"]
        ]
        config, report = [json.loads(path.read_text()) for path in [config_path, report_path]]
        assert config["steps"] == len(report["steps"]) == 80
        assert len(list((run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))) == 80
        law = {
            key: value
            for key, value in config.items()
            if key not in ["local_levels", "output", "steps", "source_requested_steps"]
        }
        if physical is None:
            physical = law
        assert law == physical
        with np.load(mesh) as saved:
            vertices, tets = saved["vertices"], saved["tetrahedra"]
            corners = vertices[tets]
            volume = abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
            com = np.einsum("n,ni->i", volume, corners.mean(axis=1)) / volume.sum()
        geometries.append(
            {
                "run": str(run),
                "tetrahedra": len(tets),
                "volume_m3": float(volume.sum()),
                "mass_kg_at_density2000": float(2000 * volume.sum()),
                "com_m": com.tolist(),
            }
        )
        np.testing.assert_allclose(volume.sum(), geometries[0]["volume_m3"], rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(com, geometries[0]["com_m"], rtol=0, atol=1e-13)
        for path in [config_path, report_path, mesh]:
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert [row["tetrahedra"] for row in geometries] == sorted({row["tetrahedra"] for row in geometries})
    closed_source(args.fine_audit)
    assert int((args.fine_audit / "returncode.txt").read_text()) == 0
    audit_path = args.fine_audit / "results/summary.json"
    audit = json.loads(audit_path.read_text())
    assert Path(audit["run"]).resolve() == args.run[-1].resolve()
    assert audit["states_checked"] == len(audit["rows"]) == 80
    assert audit["complete_source_requested_history"]
    assert all(
        row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
        for row in audit["rows"]
    )
    inputs[str(audit_path)] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    snapshots = []
    original_argv = sys.argv
    try:
        for step in range(args.sample_every, 81, args.sample_every):
            output = args.output / f"step{step:04d}"
            sys.argv = [
                "compare_recovered_spatial_state",
                "--run",
                *map(str, args.run),
                "--step",
                str(step),
                "--output",
                str(output),
            ]
            compare_state()
            summary_path = output / "summary.json"
            result = json.loads(summary_path.read_text())
            assert len(result["rows"]) == 4
            snapshots.append(
                {
                    "step": step,
                    "summary": str(summary_path),
                    "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    "rows": result["rows"],
                    "comparisons": result["comparisons"],
                }
            )
    finally:
        sys.argv = original_argv
    assert snapshots and snapshots[-1]["step"] == 80
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    result = {
        "scope": __doc__,
        "run": [str(run) for run in args.run],
        "fine_audit": str(args.fine_audit),
        "common_snapshots": snapshots,
        "geometry_and_mass_checks": geometries,
        "physical_configuration": physical,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "All four producer histories contain80 states with unchanged physical inputs and represented geometry/mass. The finest history has a separate all80-state independent audit; this comparison freshly reconstructs body/mapped balances, complete raw corner/global bounds, common-probe tensors and objective motion only at its explicitly sampled times. This is a stationary two-jaw history, separate from the original330-macro full-arm approach/lift/support protocol. Surface and bulk/contact resolution vary together; nonlinear local strain, geometry/contact/regularization, models/feedback and physically verified warmed costs remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"common_times": len(snapshots), "histories": len(args.run), "goal_completion": False}), flush=True
    )


if __name__ == "__main__":
    main()
