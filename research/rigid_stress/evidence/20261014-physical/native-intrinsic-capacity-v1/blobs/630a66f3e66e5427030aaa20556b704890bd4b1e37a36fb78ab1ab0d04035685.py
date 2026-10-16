import numpy as np
import pytest
import torch

import genesis as gs


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("substeps", [1, 3])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_substep_contact_snapshots(substeps, n_envs):
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.009, substeps=substeps), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.011)))
    scene.build(n_envs=n_envs)
    observations = []

    def before(i_substep):
        observations.append(("pre", i_substep, box.get_pos().clone()))

    def after(i_substep):
        observations.append(("post", i_substep, box.get_pos().clone()))
        contacts = box.get_contacts(is_padded=True)
        assert torch.isfinite(contacts["force_a"]).all()
        assert torch.isfinite(contacts["force_b"]).all()

    scene.register_pre_substep_callback(before)
    scene.register_post_substep_callback(after)
    for _ in range(4):
        initial = box.get_pos().clone()
        observations.clear()
        scene.step(update_visualizer=False)
        assert [(kind, index) for kind, index, _ in observations] == [
            (kind, index) for index in range(substeps) for kind in ("pre", "post")
        ]
        torch.testing.assert_close(observations[0][2], initial, rtol=0, atol=0)
        for i_substep in range(substeps - 1):
            torch.testing.assert_close(observations[2 * i_substep + 1][2], observations[2 * i_substep + 2][2])
        torch.testing.assert_close(observations[-1][2], box.get_pos())
    assert np.isfinite(box.get_pos().cpu().numpy()).all()
    observations.clear()
    scene.register_pre_step_callback(lambda: True)
    scene.step(update_visualizer=False)
    assert observations == []
