"""Use unchanged independent assertions to evaluate the experimental scheduler."""

import pytest
from traction_warp_candidate_v1 import kernel_fit_tractions


class CandidatePlugin:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            if "kernel_fit_tractions" in item.module.__dict__:
                item.module.kernel_fit_tractions = kernel_fit_tractions


raise SystemExit(
    pytest.main(
        [
            "-q",
            "-n", "0", "--tb=short",
            "tests/rigid/test_stress_surface_fan.py",
            "tests/rigid/test_stress_traction.py",
            "--backend",
            "cuda",
        ],
        plugins=[CandidatePlugin()],
    )
)
