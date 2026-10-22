"""Check actual native friction contacts, held observations and physical restores."""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTet

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_live_hertz_normal import build_fixture, snapshot
from .check_native_hertz_recovery import array


def friction_snapshot(fixture):
    entry = fixture.entry
    assert entry is not None and entry.hertz is not None
    result = snapshot(entry)
    for name in [
        "line",
        "previous_radius",
        "previous_scale",
        "coefficients",
        "status",
        "residual",
        "traction_bound",
        "iterations",
    ]:
        result["friction_" + name] = array(vars(entry.hertz.friction.state)[name])
    for name in ["force", "normal", "friction", "angle", "previous_index", "is_reset", "is_valid"]:
        result["friction_input_" + name] = array(vars(entry.hertz.friction.input)[name])
    result["binding_lost"] = array(entry.hertz.binding.is_lost)
    result["position"] = tensor_to_array(fixture.target.get_pos())
    result["probe_active"] = tensor_to_array(fixture.target.base_link.get_stress_contact_probes().active)
    return result


def compare_history(left, right):
    for key in [
        "friction_line",
        "friction_previous_radius",
        "friction_previous_scale",
        "friction_coefficients",
        "position",
    ]:
        np.testing.assert_allclose(left[key], right[key], rtol=1e-7, atol=1e-14)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--rotated", action="store_true")
    cli.add_argument("--steps", type=int, default=18)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "scope": "Actual force-driven sphere/flat rigid contact. Compare physical history with every-step and manual observations, two substeps, selected velocity reset and checkpoint continuation. No independent pressure, moving-patch physical accuracy, curved-shell or Franka admission.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    mesh = MeshTet.init_tensor(np.array([-0.004, 0, 0.004]), np.array([-0.004, 0, 0.004]), np.array([-0.002, 0, 0.002]))
    path = args.output / "mesh.npz"
    np.savez_compressed(
        path, vertices=mesh.p.T, tetrahedra=mesh.t.T, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    full = build_fixture(path, dt=2e-4, substeps=2, friction=0.4, rotated=args.rotated)
    maximum = build_fixture(path, dt=2e-4, substeps=2, friction=0.4, output_mode="max", rotated=args.rotated)
    manual = build_fixture(path, dt=2e-4, substeps=2, friction=0.4, update_mode="manual", rotated=args.rotated)
    rigid = build_fixture(path, dt=2e-4, substeps=2, friction=0.4, output_mode="rigid", rotated=args.rotated)
    rows, saved = [], []
    held = None
    for step in range(args.steps):
        phase = 2 * np.pi * step / (args.steps - 1)
        force = np.zeros((4, 6))
        force[:, 0] = 0.0007 * np.cos(phase + np.arange(4) * 0.2)
        force[:, 1] = 0.0005 * np.sin(phase + np.arange(4) * 0.3)
        force[:, 2] = 0.0004 * np.sin(phase)
        force[:, :3] = force[:, :3] @ full.rotation.T
        is_observed = step % 4 == 0 or step == args.steps - 1
        if is_observed:
            manual.target.base_link.request_stress_update()
        for fixture in [full, manual, maximum, rigid]:
            fixture.target.control_dofs_force(force)
            try:
                fixture.scene.step()
            except gs.GenesisException:
                if fixture.entry is not None and fixture.entry.output_mode == "full":
                    np.savez_compressed(
                        args.output / f"failure-{step}-{fixture.entry.options.update_mode}.npz",
                        **friction_snapshot(fixture),
                    )
                raise
        current, sparse = friction_snapshot(full), friction_snapshot(manual)
        np.savez_compressed(args.output / f"full{step}.npz", **current)
        np.savez_compressed(args.output / f"manual{step}.npz", **sparse)
        compare_history(current, sparse)
        assert maximum.entry is not None and maximum.entry.hertz is not None
        np.testing.assert_allclose(
            array(maximum.entry.hertz.friction.state.line), current["friction_line"], rtol=1e-7, atol=1e-14
        )
        np.testing.assert_allclose(
            tensor_to_array(maximum.target.base_link.get_max_stress()), current["vm_pa"], rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            tensor_to_array(maximum.target.base_link.get_max_principal_stress()),
            current["principal_pa"],
            rtol=1e-7,
            atol=1e-7,
        )
        np.testing.assert_allclose(current["position"], tensor_to_array(rigid.target.get_pos()), rtol=0, atol=1e-12)
        assert current["step_valid"].all() and current["elastic_valid"].all()
        assert not current["binding_lost"].any()
        assert np.max(current["friction_traction_bound"]) <= 1 + 1e-10
        if is_observed:
            for key in ["corner_tensor_pa", "vm_pa", "principal_pa", "probe_tensor_pa", "probe_active"]:
                np.testing.assert_allclose(current[key], sparse[key], rtol=1e-7, atol=1e-7)
            held = {
                key: sparse[key].copy()
                for key in ["corner_tensor_pa", "vm_pa", "principal_pa", "probe_tensor_pa", "probe_active"]
            }
        else:
            assert held is not None
            for key, value in held.items():
                np.testing.assert_array_equal(sparse[key], value)
        fraction = np.linalg.norm(current["friction_input_force"], axis=-1) / np.maximum(
            current["friction_input_normal"], 1e-300
        )
        rows.append(
            {
                "step": step,
                "observed": is_observed,
                "maximum_actual_tangential_to_normal": float(fraction.max()),
                "maximum_solver_iterations": int(current["friction_iterations"].max()),
            }
        )
        saved.append(current)
        write_json(args.output / "progress.json", {"rows": rows})
    assert max(row["maximum_actual_tangential_to_normal"] for row in rows) > 1e-3
    assert not np.array_equal(saved[2]["friction_line"], saved[3]["friction_line"])
    entry = full.entry
    assert entry is not None and entry.hertz is not None
    checkpoint = full.scene.rigid_solver.__getstate__()
    assert any("friction.state.line" in name for name in checkpoint.arrays)
    before = friction_snapshot(full)
    full.scene.step()
    expected = friction_snapshot(full)
    full.scene.rigid_solver.__setstate__(checkpoint)
    restored = friction_snapshot(full)
    for key in ["friction_line", "friction_previous_radius", "friction_previous_scale"]:
        np.testing.assert_array_equal(restored[key], before[key])
    assert not restored["probe_active"].any() and np.isnan(restored["probe_tensor_pa"]).all()
    full.scene.step()
    replay = friction_snapshot(full)
    compare_history(expected, replay)
    for key in ["corner_tensor_pa", "vm_pa", "principal_pa", "probe_tensor_pa"]:
        np.testing.assert_allclose(replay[key], expected[key], rtol=1e-7, atol=1e-7)
    np.savez_compressed(args.output / "replay.npz", **replay)
    before_reset = friction_snapshot(full)
    velocity = tensor_to_array(full.target.get_dofs_velocity(envs_idx=[1]))
    full.target.set_dofs_velocity(velocity, envs_idx=[1])
    after_reset = friction_snapshot(full)
    n_regions = entry.contacts.valid.shape[0]
    before_lines = before_reset["friction_line"].reshape(n_regions, 4, 17, 2)
    after_lines = after_reset["friction_line"].reshape(n_regions, 4, 17, 2)
    np.testing.assert_array_equal(after_lines[:, [0, 2, 3]], before_lines[:, [0, 2, 3]])
    np.testing.assert_array_equal(after_lines[:, 1], 0)
    assert not after_reset["probe_active"][1].any()
    np.savez_compressed(args.output / "selected-reset.npz", **after_reset)
    full.scene.step()
    assert friction_snapshot(full)["step_valid"].all()
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "mesh": file_id(path),
            "native_force_driven_friction_and_history": True,
            "manual_history_and_held_observation_passed": True,
            "physical_checkpoint_replay_passed": True,
            "selected_reset_isolation_passed": True,
            "pure_rigid_trajectory_passed": True,
            "research_callback_used": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
