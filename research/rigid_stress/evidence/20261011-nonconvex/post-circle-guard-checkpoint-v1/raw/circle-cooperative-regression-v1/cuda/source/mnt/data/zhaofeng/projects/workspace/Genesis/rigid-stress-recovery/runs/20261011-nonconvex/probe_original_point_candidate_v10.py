"""Evaluate sample-parallel scheduling with the current original-point termination budget."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from traction_original_point_candidate_v10 import kernel_fit_tractions
from traction_sample_cooperative_candidate_v2 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

from genesis.engine.solvers.rigid.stress import intrinsic


class CandidatePlugin:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            for name, function in (
                ("kernel_fit_tractions", kernel_fit_tractions),
                ("kernel_scatter_tractions", kernel_scatter_tractions),
            ):
                if name in item.module.__dict__:
                    item.module.__dict__[name] = function


with (
    patch.object(intrinsic, "kernel_fit_tractions", kernel_fit_tractions),
    patch.object(intrinsic, "kernel_commit_intrinsic_contacts", kernel_commit_intrinsic_contacts),
    patch.object(intrinsic, "kernel_scatter_tractions", kernel_scatter_tractions),
):
    if "--test" in sys.argv:
        status = pytest.main(
            [
                "-q",
                "-s",
                "-n",
                "0",
                "--tb=short",
                "--backend",
                sys.argv[sys.argv.index("--test") + 1],
                "tests/rigid/test_stress_nonconvex.py",
                "tests/rigid/test_stress_sources.py",
                "tests/rigid/test_stress_traction.py",
                "tests/rigid/test_stress_footprint.py",
                "tests/rigid/test_stress_sliding_footprint.py",
                "tests/rigid/test_stress_surface_fan.py",
                "tests/rigid/test_stress_shared_support.py",
                "tests/rigid/test_stress_shared_lifecycle.py",
                "tests/rigid/test_stress_compatibility.py",
            ],
            plugins=[CandidatePlugin()],
        )
        raise SystemExit(status)
    runpy.run_module("examples.speed_benchmark.rigid_stress", run_name="__main__")

output = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(output.read_text())
result["experimental_fit_scheduler"] = "cooperative_original_point_v10"
result["experimental_sample_scheduler"] = "cooperative_sample_v2"
for name in (
    "traction_original_point_candidate_v10.py",
    "traction_sample_cooperative_candidate_v2.py",
    Path(__file__).name,
):
    path = Path(__file__).with_name(name)
    result["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
output.write_text(json.dumps(result, indent=2) + "\n")
