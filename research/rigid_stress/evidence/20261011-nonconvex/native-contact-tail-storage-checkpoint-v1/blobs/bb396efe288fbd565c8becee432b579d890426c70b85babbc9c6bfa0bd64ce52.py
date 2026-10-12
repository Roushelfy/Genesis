"""Ablate sample-contiguous scalar profile/index storage on qualified counted-position sampling."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import genesis as gs
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import intrinsic, traction
from genesis.utils.array_class import V


def sample_contiguous_tensor(*args, **kwargs):
    if gs.backend == gs.cuda and len(kwargs["shape"]) == 3:
        kwargs["layout"] = (2, 1, 0)
    return V(*args, **kwargs)


mode = sys.argv.pop(1)
assert mode in ("off", "on")
factory = sample_contiguous_tensor if mode == "on" else V
with patch.object(traction, "V", factory), patch.object(intrinsic, "V", factory):
    if "--test" in sys.argv:
        backend = sys.argv[sys.argv.index("--test") + 1]
        report = sys.argv[sys.argv.index("--report") + 1]
        raise SystemExit(
            pytest.main(
                [
                    "-n",
                    "0",
                    "--backend",
                    backend,
                    "--tb=short",
                    "-q",
                    "--junitxml",
                    report,
                    "-o",
                    "junit_family=legacy",
                    "tests/rigid/test_stress_source_priors.py",
                    "tests/rigid/test_stress_sources.py",
                    "tests/rigid/test_stress_nonconvex.py::test_native_surface_paths_do_not_cross_gaps_or_holes",
                    "tests/rigid/test_stress_footprint.py",
                ]
            )
        )
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["sample_contiguous_scalar_storage"] = mode == "on"
result["sample_contiguous_scalar_storage_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
