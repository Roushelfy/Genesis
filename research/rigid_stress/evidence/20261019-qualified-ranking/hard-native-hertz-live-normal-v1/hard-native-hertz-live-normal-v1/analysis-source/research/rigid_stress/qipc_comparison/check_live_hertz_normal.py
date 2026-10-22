"""Exercise the native Hertz route with actual moving rigid contacts."""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTet

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


def snapshot(entry):
    normal = entry.hertz
    assert normal is not None
    return {
        "contact_valid": array(entry.contacts.valid),
        "contact_force_n": array(entry.contacts.force),
        "contact_position_m": array(entry.contacts.position),
        "contact_normal": array(entry.contacts.normal),
        "contact_status": array(entry.contacts.status),
        "source_status": array(normal.sources.status),
        "source_geom": array(normal.sources.geom),
        "source_pair": array(normal.sources.pair),
        "source_witness_m": array(normal.sources.witness),
        "region_status": array(normal.regions.state.status),
        "region_source_count": array(normal.regions.state.source_count),
        "region_center_m": array(normal.regions.state.center),
        "region_force_n": array(normal.regions.state.force),
        "region_moment_nm": array(normal.regions.state.moment),
        "region_curvature_per_m": array(normal.regions.state.curvature),
        "law_force_n": array(normal.law.state.force),
        "law_axes_m": array(normal.law.state.axes),
        "law_pressure_pa": array(normal.law.state.peak_pressure),
        "law_status": array(normal.law.state.status),
        "input_status": array(normal.state.status),
        "input_force_error_n": array(normal.state.force_error),
        "input_moment_error_nm": array(normal.state.moment_error),
        "recovery_status": array(normal.recovery.state.status),
        "elastic_valid": array(entry.state.valid),
        "step_valid": array(entry.state.step_valid),
        "vm_pa": array(entry.state.step_peak),
        "principal_pa": array(normal.state.step_principal),
        "probe_position_m": array(normal.recovery.probes.position),
        "probe_elements": array(normal.recovery.probes.element),
        "probe_barycentric": array(normal.recovery.probes.barycentric),
        "probe_tensor_pa": array(normal.recovery.probes.tensor),
        "corner_tensor_pa": array(entry.state.stress_tensor),
        "regular_force_n": array(entry.state.force),
        "omega_per_s": array(entry.omega),
        "frame_position_m": array(entry.contacts.frame_position),
        "frame_quaternion": array(entry.contacts.frame_quaternion),
        "hertz_centers_m": array(normal.recovery.frames.center),
        "hertz_frames": array(normal.recovery.frames.basis),
        "hertz_anchors": array(normal.recovery.frames.anchor),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--steps", type=int, default=4)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "scope": "Actual native lifecycle and moving frictionless sphere/flat-link contact. No research recovery callback, prescribed force injection or pressure solve. This first control does not qualify curved material contacts, friction history or Franka throughput.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    mesh = MeshTet.init_tensor(np.array([-0.004, 0, 0.004]), np.array([-0.004, 0, 0.004]), np.array([-0.002, 0, 0.002]))
    path = args.output / "mesh.npz"
    np.savez_compressed(
        path, vertices=mesh.p.T, tetrahedra=mesh.t.T, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1e-4, substeps=1), show_viewer=False)
    sphere = scene.add_entity(gs.morphs.Sphere(radius=0.003, fixed=True), material=gs.materials.Rigid(friction=0.0))
    target = scene.add_entity(
        gs.morphs.Box(size=(0.008, 0.008, 0.004), pos=(0, 0, 0.004999)),
        material=gs.materials.Rigid(rho=2000, friction=0.0),
    )
    target.base_link.configure_stress_recovery(RigidStressOptions.hertz(mesh=path, output_mode="full"))
    scene.build(n_envs=4)
    entry = scene.rigid_solver.stress_recovery.links[0]
    initial = target.base_link.get_stress_contact_probes()
    assert not tensor_to_array(initial.active).any()
    assert np.isnan(tensor_to_array(initial.stress_tensor)).all()
    rows = []
    for step in range(args.steps):
        positions = np.zeros((4, 3))
        positions[:, 0] = 0.0001 * np.sin(0.3 * step + np.arange(4) * 0.4)
        sphere.set_pos(positions)
        try:
            scene.step()
        except gs.GenesisException:
            np.savez_compressed(args.output / f"failed-frame{step}.npz", **snapshot(entry))
            raise
        data = snapshot(entry)
        np.savez_compressed(args.output / f"frame{step}.npz", **data)
        probes = target.base_link.get_stress_contact_probes()
        vm = tensor_to_array(target.base_link.get_max_stress())
        principal = tensor_to_array(target.base_link.get_max_principal_stress())
        assert np.isfinite(vm).all() and np.isfinite(principal).all()
        assert tensor_to_array(probes.positions).shape[:3] == (4, entry.contacts.valid.shape[0], 89)
        assert data["step_valid"].all() and data["elastic_valid"].all()
        rows.append(
            {
                "step": step,
                "contact_counts": data["contact_valid"].sum(axis=1).tolist(),
                "normal_region_counts": (data["region_source_count"] > 0).sum(axis=1).tolist(),
                "vm_pa": vm.tolist(),
                "principal_pa": principal.tolist(),
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
    assert any(sum(row["normal_region_counts"]) > 0 for row in rows)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "mesh": file_id(path),
            "actual_native_contact_and_recovery_lifecycle": True,
            "research_callback_used": False,
            "friction_general_contact_or_scene_throughput_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
