"""Audit independent rigid gauges, analytic controls, failed trials and figure provenance."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.rigid_stress.scripts.summarize_physical_initial import qualification
from research.rigid_stress.scripts.verify_nonconvex_publication import digest


def source_closure(run):
    before, after = run / "source.sha256", run / "source-after.sha256"
    assert before.read_bytes() == after.read_bytes(), run
    for line in before.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        source = run / "source" / relative
        if not source.exists():
            source = run / "source" / Path(relative).name
        if not source.exists():
            source = run / "source/launcher.sh"
        assert digest(source)[0] == expected, (run, relative)
    return True


def contact_matrix(run):
    source_closure(run)
    rows = []
    for returncode in sorted(run.glob("n*-d*.returncode")):
        success = int(returncode.read_text()) == 0
        case = run / returncode.stem
        row = {"case": case.name, "accepted": success}
        if success:
            result = json.loads((case / "result.json").read_text())
            with np.load(case / "fields.npz") as saved:
                nodes = saved["rest_positions"]
                fixed = np.isclose(nodes[:, 2], -0.005) | np.isclose(nodes[:, 2], 0.006)
                residual = np.linalg.norm(saved["reactions_N"].reshape(-1, 3)[~fixed])
                peak = max(saved["lower_vm_Pa"].max(), saved["upper_vm_Pa"].max())
            assert residual <= 1e-9
            assert abs(residual - result["complete_free_residual_N"]) <= 1e-18
            assert peak == max(part["peak_Pa"] for part in result["parts"])
            force = abs(result["parts"][0]["contact_force_N"][2])
            row.update(
                {
                    "free_residual_N": float(residual),
                    "force_N": force,
                    "raw_peak_Pa": float(peak),
                    "force_relative_error": abs(force / 0.5 - 1),
                    "peak_relative_error": abs(peak / 5000 - 1),
                }
            )
        else:
            assert (run / (case.name + ".log")).is_file()
            row["failure_snapshot_available"] = (case / "failure.npz").is_file()
            if row["failure_snapshot_available"]:
                assert (case / "failure.json").is_file()
        rows.append(row)
    assert len(rows) == 12
    return {"cases": rows, "accepted": sum(r["accepted"] for r in rows), "total": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_root
    closures = {name: source_closure(runtime / name) for name in args.run_names}
    tested = {}
    for backend in ("cpu", "cuda"):
        run = runtime / f"independent-gauges-{backend}-v3"
        record = qualification(run, run / "source")
        changed = []
        for line in (run / "source.sha256").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            current = digest(Path.cwd() / relative)[0]
            if current != expected:
                assert relative == "research/rigid_stress/contact_reference.py", relative
                for module in (
                    "research/rigid_stress/skfem_reference.py",
                    "tests/rigid/test_stress_independent_fem.py",
                    "tests/rigid/test_stress_nonconvex.py",
                ):
                    assert "contact_reference" not in (Path.cwd() / module).read_text()
                changed.append(
                    {
                        "file": relative,
                        "tested_sha256": expected,
                        "current_sha256": current,
                        "role": "Archived offline contact benchmark; not imported by this elasticity qualification",
                    }
                )
        record["unused_snapshot_source_changes"] = changed
        tested[backend] = record
    gauges = [field for record in tested.values() for test in record["test_properties"] for field in test["fields"]]
    assert all(
        f["independent_pinned_residual_budget_fraction"] <= 1 and f["independent_mass_residual_budget_fraction"] <= 1
        for f in gauges
    )
    shell_run = runtime / "spherical-shell-v2"
    assert (shell_run / "returncode.txt").read_text().strip() == "1"
    shell = json.loads((shell_run / "sweep/summary.json").read_text())["results"]
    assert len(shell) == len({r["mesh"]["mesh_sha256"] for r in shell}) == 4
    for row in shell:
        case = shell_run / "sweep" / f"h{row['mesh']['mesh_size']:g}"
        assert digest(case / "mesh.npz")[0] == row["mesh"]["mesh_sha256"]
        assert digest(case / "field.npz")[0] == row["field_sha256"]
        assert row["complete_residual_budget_fraction"] <= 1
    refined_run = runtime / "spherical-shell-pinned-v3"
    assert (refined_run / "returncode.txt").read_text().strip() == "0"
    refined = json.loads((refined_run / "sweep/summary.json").read_text())["results"]
    assert len(refined) == len({r["mesh"]["mesh_sha256"] for r in refined}) == 5
    for row in refined:
        case = refined_run / "sweep" / f"h{row['mesh']['mesh_size']:g}"
        assert digest(case / "mesh.npz")[0] == row["mesh"]["mesh_sha256"]
        assert digest(case / "field.npz")[0] == row["field_sha256"]
        assert row["complete_residual_budget_fraction"] <= 1
    with (
        np.load(refined_run / "sweep/h0.11/field.npz") as previous,
        np.load(refined_run / "sweep/h0.08/field.npz") as fine,
    ):
        assert np.array_equal(previous["probes_m"], fine["probes_m"])
        tensor_tail = np.linalg.norm(previous["probe_tensor_Pa"] - fine["probe_tensor_Pa"]) / np.linalg.norm(
            fine["probe_tensor_Pa"]
        )
        vm_tail = np.linalg.norm(previous["probe_vm_Pa"] - fine["probe_vm_Pa"]) / np.linalg.norm(fine["probe_vm_Pa"])
    matrices = {
        name: contact_matrix(runtime / name)
        for name in args.run_names
        if name.startswith("contact-blocks-convergence-")
    }
    figure_run = runtime / "control-figures-v3"
    assert (figure_run / "returncode.txt").read_text().strip() == "0"
    figure = json.loads((figure_run / "figures/figure_contract.json").read_text())
    for row in figure["inputs"]:
        assert digest(Path(row["path"]))[0] == row["sha256"]
    for exported in figure["exports"].values():
        assert {Path(path).suffix for path in exported.values()} == {".svg", ".pdf", ".png"}
        assert all(Path(path).is_file() for path in exported.values())
    result = {
        "scope": "Independent gauge/probe qualification and initial physical-control progress",
        "source_closures": closures,
        "qualification": tested,
        "gauge_fields": len(gauges),
        "maximum_independent_gauge_peak_difference_Pa": max(f["independent_gauge_peak_difference_Pa"] for f in gauges),
        "maximum_pinned_complete_budget_fraction": max(
            f["independent_pinned_residual_budget_fraction"] for f in gauges
        ),
        "maximum_mass_complete_budget_fraction": max(f["independent_mass_residual_budget_fraction"] for f in gauges),
        "completed_spherical_shell_levels": shell,
        "pinned_spherical_shell_levels": refined,
        "tensor_probe_tail_relative_change": float(tensor_tail),
        "vm_probe_tail_relative_change": float(vm_tail),
        "shell_fine_failure": "h0.08 SuperLU memory expansion fails; gstrf SystemError. SlurmFAILED, not a completed field.",
        "contact_matrices": matrices,
        "figure_contract": figure,
        "physical_stage_complete": False,
        "limitations": "This checkpoint qualifies reference algebra/gauge and initial physical controls only. "
        "Contact-reference regressions and unresolved fine configurations remain failed. "
        "Solver-scaling trials are excluded. Full shapes, "
        "four load models, friction and egg/interior-hotspot deformable contact remain required.",
        "qualification_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cpu_cases": 8,
                "cuda_cases": 8,
                "gauge_fields": len(gauges),
                "gauge_peak_difference_Pa": result["maximum_independent_gauge_peak_difference_Pa"],
                "contact_matrices": {name: r["accepted"] for name, r in matrices.items()},
                "physical_stage_complete": False,
            }
        )
    )


if __name__ == "__main__":
    main()
