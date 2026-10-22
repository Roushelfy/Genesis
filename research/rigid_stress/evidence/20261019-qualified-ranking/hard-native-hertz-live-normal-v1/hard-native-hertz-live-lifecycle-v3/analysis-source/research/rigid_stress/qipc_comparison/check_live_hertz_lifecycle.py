"""Check native normal stress substeps, public fields and selective invalidation."""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTet

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_live_hertz_normal import build_fixture, snapshot
from .check_native_hertz_recovery import array


class NativePeakAudit:
    """Observe original native results after each real recovery call."""

    def __init__(self, fixture):
        self.entry = fixture.entry
        self.sim = fixture.scene.sim
        self.values = []
        self.clocks = []

    def __call__(self, i_substep: int):
        assert self.entry is not None and self.entry.hertz is not None
        self.values.append(
            np.stack([array(self.entry.state.peak), array(self.entry.hertz.recovery.state.tensile_peak)])
        )
        self.clocks.append([i_substep, self.sim.cur_substep_global, self.sim.cur_substep_local])


def peaks(fixture):
    return np.stack(
        [
            tensor_to_array(fixture.target.base_link.get_max_stress()),
            tensor_to_array(fixture.target.base_link.get_max_principal_stress()),
        ]
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--steps", type=int, default=6)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"peaks": 1e-7, "trajectory_absolute_m": 1e-12, "field": 1e-7}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "targets": targets,
            "scope": "Actual centered normal contacts. Native substep aggregation, max/full, pure rigid trajectories, selected state mutation/restore, manual updates, unbatched output and release/reload. No general friction, curved contact or Franka throughput claim.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    mesh = MeshTet.init_tensor(np.array([-0.004, 0, 0.004]), np.array([-0.004, 0, 0.004]), np.array([-0.002, 0, 0.002]))
    path = args.output / "mesh.npz"
    np.savez_compressed(
        path, vertices=mesh.p.T, tetrahedra=mesh.t.T, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    full = build_fixture(path, dt=2e-4, substeps=2)
    half = build_fixture(path, dt=1e-4, substeps=1)
    maximum = build_fixture(path, dt=2e-4, substeps=2, output_mode="max")
    rigid = build_fixture(path, dt=2e-4, substeps=2, output_mode="rigid")
    assert full.entry is not None and full.entry.hertz is not None
    audit = NativePeakAudit(full)
    full.scene.register_post_substep_callback(audit)
    for fixture in [full, half, maximum, rigid]:
        fixture.target.set_dofs_velocity(np.tile([0, 0, -1e-4, 0, 0, 0], (4, 1)))
    rows, sampled = [], []
    for step in range(args.steps):
        position = np.zeros((4, 3))
        position[:, 2] = 1e-7 * np.sin(0.3 * step + np.arange(4) * 0.4)
        for fixture in [full, half, maximum, rigid]:
            fixture.sphere.set_pos(position)
        audit.values.clear()
        audit.clocks.clear()
        full.scene.step()
        assert len(audit.values) == 2
        maximum.scene.step()
        rigid.scene.step()
        half.scene.step()
        first = peaks(half)
        half.scene.step()
        second = peaks(half)
        observed, max_observed = peaks(full), peaks(maximum)
        expected = np.maximum.reduce(audit.values)
        np.savez_compressed(
            args.output / f"aggregation{step}.npz",
            observed=observed,
            expected=expected,
            raw_substeps=np.stack(audit.values),
            clocks=np.array(audit.clocks),
            full_force_n=array(full.entry.contacts.force),
            half_force_n=array(half.entry.contacts.force),
            half_peaks=np.stack([first, second]),
        )
        np.testing.assert_allclose(observed, expected, rtol=targets["peaks"], atol=1e-7)
        np.testing.assert_allclose(observed, np.maximum(first, second), rtol=targets["peaks"], atol=1e-7)
        np.testing.assert_allclose(observed, max_observed, rtol=targets["peaks"], atol=1e-7)
        pose = tensor_to_array(full.target.get_pos())
        for other in [half, maximum, rigid]:
            np.testing.assert_allclose(
                pose, tensor_to_array(other.target.get_pos()), rtol=0, atol=targets["trajectory_absolute_m"]
            )
        np.savez_compressed(
            args.output / f"dt-comparison{step}.npz",
            full_position_m=pose,
            half_position_m=tensor_to_array(half.target.get_pos()),
            full_force_n=array(full.entry.contacts.force),
            half_force_n=array(half.entry.contacts.force),
        )
        for actual, other in zip(full.target.base_link.get_stress_field(), half.target.base_link.get_stress_field()):
            np.testing.assert_allclose(
                tensor_to_array(actual), tensor_to_array(other), rtol=targets["field"], atol=1e-5
            )
        sampled.append(np.stack([observed, max_observed, *audit.values, first, second]))
        rows.append(
            {
                "step": step,
                "max_full_relative": float(np.linalg.norm(observed - max_observed) / max(np.linalg.norm(observed), 1)),
                "substep_aggregation_relative": float(
                    np.linalg.norm(observed - expected) / max(np.linalg.norm(observed), 1)
                ),
                "separate_outer_dt_comparison_relative": float(
                    np.linalg.norm(observed - np.maximum(first, second)) / max(np.linalg.norm(observed), 1)
                ),
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
    np.savez_compressed(args.output / "substep-peaks.npz", values=np.stack(sampled))
    assert any(np.any(values[2] > values[3] + 1e-7 * np.maximum(1, np.abs(values[2]))) for values in sampled)
    link, entry = full.target.base_link, full.entry
    assert entry is not None and entry.hertz is not None
    before = snapshot(entry)
    np.savez_compressed(args.output / "before-reset.npz", **before)
    checkpoint = full.scene.rigid_solver.__getstate__()
    assert checkpoint.configs["stress.0.options"].contact_model == "hertz"
    assert "stress.0.hertz.state.step_principal" in checkpoint.arrays
    borrowed, copied = link.get_stress_contact_probes(copy=False), link.get_stress_contact_probes(copy=True)
    full.target.set_dofs_velocity(np.zeros((1, 6)), envs_idx=[1])
    after = snapshot(entry)
    np.savez_compressed(args.output / "after-reset.npz", **after)
    for key in ["corner_tensor_pa", "vm_pa", "principal_pa", "contact_valid", "source_geom", "source_pair"]:
        np.testing.assert_array_equal(after[key][[0, 2, 3]], before[key][[0, 2, 3]])
    assert np.isnan(after["corner_tensor_pa"][1]).all() and np.isnan(after["principal_pa"][1])
    assert after["vm_pa"][1] == 0 and (after["source_pair"][1] == -1).all()
    assert np.isnan(tensor_to_array(borrowed.stress_tensor)[1]).all()
    np.testing.assert_array_equal(
        tensor_to_array(copied.stress_tensor), before["probe_tensor_pa"].reshape(-1, 4, 89, 6).transpose(1, 0, 2, 3)
    )
    selected = link.get_stress_contact_probes(envs_idx=[0, 2])
    np.testing.assert_array_equal(
        tensor_to_array(selected.stress_tensor), tensor_to_array(copied.stress_tensor)[[0, 2]]
    )
    full.scene.rigid_solver.__setstate__(checkpoint)
    restored = link.get_stress_contact_probes()
    assert not tensor_to_array(restored.active).any() and np.isnan(tensor_to_array(restored.stress_tensor)).all()
    assert np.isnan(tensor_to_array(link.get_max_principal_stress())).all()
    np.testing.assert_array_equal(tensor_to_array(link.get_max_stress()), 0.0)
    full.scene.step()
    assert np.isfinite(peaks(full)).all()
    manual = build_fixture(path, dt=2e-4, substeps=2, update_mode="manual")
    manual.target.base_link.request_stress_update()
    manual.scene.step()
    held = peaks(manual).copy()
    manual.scene.step()
    np.testing.assert_array_equal(peaks(manual), held)
    manual.target.base_link.request_stress_update()
    manual.target.set_dofs_velocity(np.zeros((1, 6)), envs_idx=[1])
    manual.scene.step()
    assert np.isnan(peaks(manual)[1, 1])
    np.testing.assert_array_equal(peaks(manual)[:, [0, 2, 3]], held[:, [0, 2, 3]])
    manual.target.base_link.request_stress_update()
    manual.scene.step()
    assert np.isfinite(peaks(manual)).all()
    unbatched = build_fixture(path, batch=0)
    unbatched.scene.step()
    probes = unbatched.target.base_link.get_stress_contact_probes()
    assert probes.positions.ndim == 3 and probes.stress_tensor.shape[-2:] == (89, 6)
    assert unbatched.target.base_link.get_max_principal_stress().ndim == 0
    assert unbatched.target.base_link.get_max_stress().ndim == 0
    rejected_radius = False
    try:
        unbatched.target.base_link.set_stress_contact_radius(0.001)
    except gs.GenesisException:
        rejected_radius = True
    assert rejected_radius
    unbatched.sphere.set_pos((0, 0, -0.01))
    unbatched.scene.step()
    released = unbatched.target.base_link.get_stress_contact_probes()
    assert not tensor_to_array(released.active).any()
    assert (tensor_to_array(released.elements) == -1).all()
    np.testing.assert_array_equal(tensor_to_array(unbatched.target.base_link.get_max_stress()), 0.0)
    unbatched.target.set_pos((0, 0, 0.004999))
    unbatched.target.set_dofs_velocity(np.zeros(6))
    unbatched.sphere.set_pos((0, 0, 0))
    unbatched.scene.step()
    assert tensor_to_array(unbatched.target.base_link.get_stress_contact_probes().active).any()
    np.savez_compressed(args.output / "final-full.npz", **snapshot(entry))
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "targets": targets,
            "mesh": file_id(path),
            "substep_max_full_and_rigid_trajectories_passed": True,
            "substep_reference_uses_the_same_scene_actual_native_calls": True,
            "public_post_substep_audit_does_not_produce_loads_or_fields_and_is_not_timed": True,
            "separate_outer_dt_comparison_also_passed": True,
            "partial_reset_and_snapshot_restore_passed": True,
            "copied_and_borrowed_probe_views_passed": True,
            "manual_request_cancellation_and_retention_passed": True,
            "unbatched_release_reload_and_predicted_radius_contract_passed": True,
            "general_friction_or_franka_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
