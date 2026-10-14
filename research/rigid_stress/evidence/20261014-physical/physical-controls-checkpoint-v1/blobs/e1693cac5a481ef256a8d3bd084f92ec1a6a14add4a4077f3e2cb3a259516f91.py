"""Validate the first independent FEM qualification and preserve its exact source provenance."""

import argparse
import ast
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from research.rigid_stress.scripts.verify_nonconvex_publication import digest


def qualification(run, root):
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    source_changes = []
    for row in (run / "source.sha256").read_text().splitlines():
        expected, relative = row.split("  ", 1)
        recorded = run / "source" / relative
        assert digest(recorded)[0] == expected, relative
        if digest(root / relative)[0] != expected:
            assert relative == "tests/rigid/test_stress_independent_fem.py", relative
            assert ast.dump(ast.parse(recorded.read_text())) == ast.dump(ast.parse((root / relative).read_text()))
            source_changes.append(
                {
                    "file": relative,
                    "tested_sha256": expected,
                    "current_sha256": digest(root / relative)[0],
                    "AST_identical": True,
                }
            )
    report = ET.parse(run / "report.xml").getroot()
    suite = report if report.tag == "testsuite" else report.find("testsuite")
    assert int(suite.attrib["tests"]) == 8
    assert int(suite.attrib["errors"]) == int(suite.attrib["failures"]) == int(suite.attrib["skipped"]) == 0
    records = [
        json.loads(p.attrib["value"]) for p in report.iter("property") if p.attrib["name"] == "independent_scikit_fem"
    ]
    assert len(records) == 8
    expected = {
        (shape, method, analytic)
        for shape in ("u", "cavity")
        for method in ("direct", "pcg")
        for analytic in ("uniform_strain", "pure_bending")
    }
    assert {(r["shape"], r["method"], r["analytic"]) for r in records} == expected
    fields = [f for r in records for f in r["fields"]]
    assert len(fields) == 16 and all(f["budget_fraction"] <= 1 for f in fields)
    assert all(r["stiffness_relative_error"] < 1e-11 and r["mass_relative_error"] < 1e-11 for r in records)
    return {
        "suite": suite.attrib,
        "test_properties": records,
        "formatting_source_equivalence": source_changes,
        "maximum_complete_residual_budget_fraction": max(f["budget_fraction"] for f in fields),
        "maximum_peak_difference_Pa": max(f["peak_difference_Pa"] for f in fields),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    runtime = args.runtime_root
    assert not subprocess.check_output(["git", "diff", "76537c2f", "HEAD", "--", "genesis", "examples"])
    results = {backend: qualification(runtime / f"independent-{backend}-v1", root) for backend in ("cpu", "cuda")}
    environment = json.loads((runtime / "setup-v2/environment.json").read_text())
    assert environment["packages"]["scikit-fem"]["version"] == "12.0.2"
    assert environment["packages"]["gmsh"]["version"] == "4.14.1"
    setup = runtime / "setup-v2"
    assert (setup / "source.sha256").read_bytes() == (setup / "source-after.sha256").read_bytes()
    meshes = runtime / "mesh-smoke-v2"
    assert (meshes / "returncode.txt").read_text().strip() == "0"
    assert (meshes / "source.sha256").read_bytes() == (meshes / "source-after.sha256").read_bytes()
    records = []
    for path in sorted((meshes / "meshes").glob("*.json")):
        value = json.loads(path.read_text())
        assert hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() == value["mesh_sha256"]
        assert value["minimum_volume_m3"] > 0
        records.append(value)
    assert {r["geometry"]["name"] for r in records} == {
        "ellipsoid",
        "egg",
        "capsule",
        "u",
        "thin_neck",
        "cavity",
        "narrow_gap",
    }
    output = {
        "scope": "independent elasticity qualification and initial geometry construction",
        "numerical_engine_revision": "76537c2fb64cd469e83ba9921b0724b5c82a5f36",
        "environment": environment,
        "qualification": results,
        "initial_meshes": records,
        "limitations": "These checks do not establish load-model accuracy, physical convergence or deformable contact. "
        "The complete active stage remains unfinished.",
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps({"cpu_cases": 8, "cuda_cases": 8, "independent_fields": 32, "initial_geometries": 7, "passed": True})
    )


if __name__ == "__main__":
    main()
