"""Exercise cached final scatter against the unchanged componentwise FP64 field oracles."""
import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.rigid.stress.traction import kernel_prepare_cached_tractions
from tests.rigid import test_stress_footprint as footprint_tests

anchor_factory = footprint_tests.create_anchor_fan_tractions
traction_factory = footprint_tests.create_tractions


def anchor_tractions(*args, **kwargs):
    kwargs["cached_acceptance"] = True
    return anchor_factory(*args, **kwargs)


def tractions(*args, **kwargs):
    kwargs["cached_acceptance"] = True
    return traction_factory(*args, **kwargs)


def scatter(*args, **kwargs):
    return kernel_prepare_cached_tractions(float(np.finfo(gs.np_float).eps), *args, **kwargs)


@pytest.fixture(autouse=True)
def fused_componentwise_scatter(monkeypatch):
    monkeypatch.setattr(footprint_tests, "create_anchor_fan_tractions", anchor_tractions)
    monkeypatch.setattr(footprint_tests, "create_tractions", tractions)
    monkeypatch.setattr(footprint_tests, "kernel_scatter_tractions", scatter)
