"""Exercise the native Hertz route with actual moving rigid contacts."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from skfem import MeshTet

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions
from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
from genesis.engine.solvers.rigid.stress.recovery import StressLink
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


@dataclass(frozen=True)
class LiveHertzFixture:
    scene: gs.Scene
    sphere: RigidEntity
    target: RigidEntity
    rotation: np.ndarray
    entry: StressLink | None


def build_fixture(
    mesh,
    *,
    batch=4,
    dt=1e-4,
    substeps=1,
    output_mode="full",
    update_mode="every_step",
    quadrature=8,
    rotated=False,
    route="hertz",
):
    rotation = Rotation.from_rotvec([0.31, -0.47, 0.22]).as_matrix() if rotated else np.eye(3)
    quaternion = Rotation.from_matrix(rotation).as_quat()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=tuple(rotation @ np.array([0, 0, -9.81]))),
        show_viewer=False,
    )
    sphere = scene.add_entity(gs.morphs.Sphere(radius=0.003, fixed=True), material=gs.materials.Rigid(friction=0.01))
    target = scene.add_entity(
        gs.morphs.Box(
            size=(0.008, 0.008, 0.004),
            pos=tuple(rotation @ np.array([0, 0, 0.004999])),
            quat=tuple(np.r_[quaternion[3], quaternion[:3]]),
        ),
        material=gs.materials.Rigid(rho=2000, friction=0.01),
    )
    if output_mode != "rigid":
        if route == "hertz":
            options = RigidStressOptions.hertz(mesh=mesh, output_mode=output_mode, update_mode=update_mode)
            if quadrature is not None:
                options.quadrature = quadrature
        else:
            assert route == "dpr"
            options = RigidStressOptions.dpr(
                mesh=mesh, output_mode=output_mode, update_mode=update_mode, quadrature=quadrature, contact_radius=0.001
            )
        target.base_link.configure_stress_recovery(options)
    try:
        scene.build(n_envs=batch)
    except gs.GenesisException:
        recovery = scene.rigid_solver.stress_recovery
        if recovery is not None:
            np.savez_compressed(Path(mesh).parent / "failed-build.npz", **snapshot(recovery.links[0]))
        raise
    entry = scene.rigid_solver.stress_recovery.links[0] if output_mode != "rigid" else None
    target.set_dofs_velocity(np.zeros((batch, 6)) if batch else np.zeros(6))
    return LiveHertzFixture(scene, sphere, target, rotation, entry)


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
        "law_direction": array(normal.law.state.direction),
        "input_status": array(normal.state.status),
        "input_force_error_n": array(normal.state.force_error),
        "input_moment_error_nm": array(normal.state.moment_error),
        "recovery_status": array(normal.recovery.state.status),
        "recovery_wrench_error": array(normal.recovery.state.wrench_error),
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
        "probe_offsets": array(normal.recovery.probe_offsets),
        "material_nodes_m": array(entry.model.info.vertices),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--steps", type=int, default=4)
    cli.add_argument("--quadrature", type=int, default=8)
    cli.add_argument("--default-quadrature", action="store_true")
    cli.add_argument("--rotated", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "scope": "Actual native lifecycle and centered sphere/flat-link normal contact with changing vertical motion. Material friction is0.01; actual shear must satisfy the normal-only acceptance. No research recovery callback, prescribed force injection or pressure solve. This first control does not qualify curved material contacts, friction history or Franka throughput.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    mesh = MeshTet.init_tensor(np.array([-0.004, 0, 0.004]), np.array([-0.004, 0, 0.004]), np.array([-0.002, 0, 0.002]))
    path = args.output / "mesh.npz"
    np.savez_compressed(
        path, vertices=mesh.p.T, tetrahedra=mesh.t.T, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    fixture = build_fixture(path, quadrature=None if args.default_quadrature else args.quadrature, rotated=args.rotated)
    scene, target, sphere, entry = fixture.scene, fixture.target, fixture.sphere, fixture.entry
    assert entry is not None
    initial = target.base_link.get_stress_contact_probes()
    assert not tensor_to_array(initial.active).any()
    assert np.isnan(tensor_to_array(initial.stress_tensor)).all()
    rows = []
    for step in range(args.steps):
        positions = np.zeros((4, 3))
        positions[:, 2] = 1e-7 * np.sin(0.3 * step + np.arange(4) * 0.4)
        sphere.set_pos(positions @ fixture.rotation.T)
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
            "quadrature": entry.link.stress_options.quadrature,
            "default_quadrature_used": args.default_quadrature,
            "research_callback_used": False,
            "friction_general_contact_or_scene_throughput_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
