"""Patch only profile scheduling for an isolated qualification or benchmark."""

import json
import runpy
import sys
from unittest.mock import patch

from fan_frontier_candidate_v1 import kernel_surface_fan_profiles, workspaces
from genesis.engine.solvers.rigid.stress import footprint, intrinsic

mode = sys.argv.pop(1)
with patch.object(footprint,"kernel_surface_fan_profiles",kernel_surface_fan_profiles), patch.object(intrinsic,"kernel_surface_fan_profiles",kernel_surface_fan_profiles):
    if mode == "tests":
        import pytest
        raise SystemExit(pytest.main(sys.argv[1:]))
    assert mode == "benchmark"
    runpy.run_module("examples.speed_benchmark.rigid_stress",run_name="__main__")
    print(json.dumps({"candidate_note": "Extra frontier allocation and maximum readback are within the measured step. The benchmark's native buffer enumeration excludes this isolated candidate's frontier;add the explicitly reported bytes.", "frontier_buffer_bytes": sum(4*(value.indices.shape[0]*value.indices.shape[1]*value.indices.shape[2]+value.count.shape[0]*value.count.shape[1]+1) for _reference,value in workspaces.values())}),flush=True)
