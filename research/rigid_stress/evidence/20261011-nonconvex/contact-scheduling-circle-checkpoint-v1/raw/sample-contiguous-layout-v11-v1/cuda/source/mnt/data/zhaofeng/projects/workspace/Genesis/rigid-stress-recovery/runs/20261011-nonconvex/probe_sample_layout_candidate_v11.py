"""Evaluate contiguous sample storage with unchanged cooperative fitting and original budgets."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from traction_original_point_candidate_v10 import kernel_fit_tractions
from traction_sample_cooperative_candidate_v2 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

import genesis as gs
from genesis.engine.solvers.rigid.stress import intrinsic, traction
from genesis.utils.array_class import V


def sample_contiguous_tensor(*args, **kwargs):
    if gs.backend == gs.cuda and len(kwargs["shape"]) == 3:
        kwargs["layout"] = (2, 1, 0)
    return V(*args, **kwargs)


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
    patch.object(traction, "V", sample_contiguous_tensor),
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
result["experimental_sample_layout"] = "profile_and_indices_batch_contact_sample"
for name in (
    "traction_original_point_candidate_v10.py",
    "traction_sample_cooperative_candidate_v2.py",
    Path(__file__).name,
):
    path = Path(__file__).with_name(name)
    result["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
output.write_text(json.dumps(result, indent=2) + "\n")
