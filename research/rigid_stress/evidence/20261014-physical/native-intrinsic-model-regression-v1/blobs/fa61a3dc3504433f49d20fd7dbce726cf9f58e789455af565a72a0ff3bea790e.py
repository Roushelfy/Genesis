from pathlib import Path

import numpy as np
import pytest

from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.solve import kernel_peak_impl
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("cached", (False, True))
def test_corner_cancellation_against_decimal_polynomial(tmp_path, cached):
    """Keep small stress components accurate when large nodal terms cancel."""
    fixture = Path(__file__).with_name("data") / "stress_corner_cancellation.npz"
    mesh = tmp_path / "corner.npz"
    with np.load(fixture) as saved:
        np.savez(
            mesh,
            vertices=saved["vertices"],
            tetrahedra=saved["tetrahedra"],
            surface_triangles=saved["surface_triangles"],
        )
        displacement = saved["displacement_m"].copy()
        expected = saved["expected_corner_components_Pa"].copy()
        original = saved["original_native_components_Pa"].copy()
    budget = 1e-5 + 1e-9 * abs(expected)
    assert np.max(abs(original - expected) / budget) > 1
    options = RigidStressOptions(mesh=mesh, method="pcg", output_mode="full")
    model = StressModel(options)
    state = model.create_state(1, "full")
    state.displacement.from_numpy(displacement[:, None])
    state.valid.fill(True)
    kernel_peak_impl(options.young, options.poisson, state, model.info, cached, True)
    tensor = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0, 0]
    assert np.all(abs(tensor - expected) <= budget)
    expected_vm = np.sqrt(
        0.5
        * (
            (expected[:, 0] - expected[:, 1]) ** 2
            + (expected[:, 1] - expected[:, 2]) ** 2
            + (expected[:, 2] - expected[:, 0]) ** 2
        )
        + 3 * np.sum(expected[:, 3:] ** 2, axis=-1)
    )
    np.testing.assert_allclose(qd_to_numpy(state.peak)[0], expected_vm.max(), rtol=1e-4, atol=1e-3)
