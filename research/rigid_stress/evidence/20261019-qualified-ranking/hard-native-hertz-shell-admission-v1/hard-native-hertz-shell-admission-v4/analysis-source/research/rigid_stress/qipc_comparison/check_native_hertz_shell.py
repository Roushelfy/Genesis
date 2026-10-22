"""Check native recovery on the unchanged level1 faceted hollow-egg volume."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array, discrete_particular, material_points, matrix
from .p2_elastic import FreeElasticP2


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--quadrature", type=int, default=8)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mesh = Path(gs.__file__).resolve().parent.parent / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "mesh": file_id(mesh),
            "quadrature": args.quadrature,
            "same_load_tensor_gate": 1e-7,
            "scope": "Four separate normal contacts at actual outer-facet centers of the unmodified level1 hollow egg, using prescribed relative curvatures50/80 per metre. Test native clearance, remote traction integration and complete regular/particular fields against independent same-load P2 elasticity. This isolates shell recovery from source association, curved pressure prediction and friction.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    with np.load(mesh) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    model = StressModel(RigidStressOptions(mesh=mesh))
    surface = StressSurface(args.quadrature, model.info)
    nodes = array(model.info.vertices)
    faces, normals = array(model.info.surface_nodes).T, array(surface.info.face_normal)
    triangles = nodes[faces[:, :3]]
    centers = triangles.mean(axis=1)
    is_outer = np.einsum("fi,fi->f", centers, normals) > 0
    directions = np.array([[0.0, 0, 1], [1.0, 0, 0], [0, 1.0, 0], [-1.0, 0.2, 0.3]])
    anchors = np.argmax(np.where(is_outer[None], directions @ normals.T, -np.inf), axis=1)
    assert len(np.unique(anchors)) == 4
    centers = centers[anchors]
    tangent = triangles[anchors, 1] - triangles[anchors, 0]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    inward = -normals[anchors]
    frames = np.stack([tangent, np.cross(inward, tangent), inward], axis=2)
    law = StressHertz(4)
    forces = np.array([0.2, 0.12, 0.18, 0.1])
    law.state.force.from_numpy(forces)
    law.state.curvature.from_numpy(np.broadcast_to(np.diag([50.0, 80.0]), (4, 2, 2)).copy())
    law.state.modulus.fill(1e10 / 0.91)
    law.update()
    axes, peaks = array(law.state.axes), array(law.state.peak_pressure)
    module = StressHertzRecovery(4, 1, output_mode="full", moving_probes=True)
    module.frames.center.from_numpy(centers)
    module.frames.basis.from_numpy(frames)
    module.frames.anchor.from_numpy(anchors.astype(gs.np_int))
    state = model.create_state(4, "full")
    omega, errno = V_VEC(3, dtype=gs.qd_float, shape=(4,)), V(dtype=gs.qd_int, shape=(4,))
    omega.fill(0)
    errno.fill(0)
    kernel_begin_step(state)
    module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
    qd.sync()
    status = array(module.state.status)
    path = args.output / "fields.npz"
    np.savez_compressed(
        path,
        centers_m=centers,
        frames=frames,
        anchors=anchors,
        axes_m=axes,
        pressure_pa=peaks,
        force_n=forces,
        status=status,
        wrench_error=array(module.state.wrench_error),
        errno=array(errno),
        material_nodes_m=nodes,
        regular_force_n=array(state.force),
        corner_tensor_pa=array(state.stress_tensor),
        probe_tensor_pa=array(module.probes.tensor),
        probe_position_m=array(module.probes.position),
        probe_elements=array(module.probes.element),
        probe_barycentric=array(module.probes.barycentric),
        probe_offsets=array(module.probe_offsets),
    )
    report = {
        "passed": False,
        "status": status.tolist(),
        "wrench_error": array(module.state.wrench_error).tolist(),
        "elastic_valid": array(state.valid).tolist(),
        "step_valid": array(state.step_valid).tolist(),
        "fields": file_id(path),
        "whole_goal_complete": False,
    }
    write_json(args.output / "result.json", report)
    assert np.all(status == 0) and np.all(array(state.step_valid)), report
    cpu = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000)
    permutation = cpu.reference.native_dof_permutation(nodes).reshape(-1, 3)[:, 0] // 3
    loads, native = array(state.force), matrix(array(state.stress_tensor))
    probe_field = matrix(array(module.probes.tensor))
    positions, elements, bary = array(module.probes.position), array(module.probes.element), array(module.probes.barycentric)
    errors, coordinate_controls = [], []
    for env in range(4):
        applied = np.empty_like(loads[env])
        applied[permutation] = loads[env]
        regular, _, _, _ = cpu.solve(applied)
        particular = discrete_particular(
            material_points(cpu.reference.corner_positions.reshape(-1, 3), centers[env], frames[env]),
            axes[env], peaks[env], forces[env], True,
        ).reshape(regular.shape)
        expected = regular + frames[env] @ particular @ frames[env].T
        corner_error = float(np.linalg.norm(native[env] - expected) / np.linalg.norm(expected))
        reconstructed_local = material_points(positions[env], centers[env], frames[env])
        local = array(module.probe_offsets) * np.r_[axes[env], axes[env, 0]]
        position_difference = float(np.linalg.norm(positions[env] - centers[env] - local @ frames[env].T, axis=1).max())
        assert position_difference < 1e-12
        # Moving probes declare their exact source-frame coordinates. The
        # surface edge is sensitive to roundoff from a world-coordinate return.
        particular = discrete_particular(local, axes[env], peaks[env], forces[env], True)
        reconstructed_particular = discrete_particular(
            reconstructed_local, axes[env], peaks[env], forces[env], True
        )
        coordinate_controls.append({
            "maximum_position_roundoff_m": position_difference,
            "reference_roundtrip_relative": float(
                np.linalg.norm(reconstructed_particular - particular) / np.linalg.norm(particular)
            ),
        })
        expected = np.einsum("qc,qcij->qij", bary[env], regular[elements[env]])
        expected += frames[env] @ particular @ frames[env].T
        probe_error = float(np.linalg.norm(probe_field[env] - expected) / np.linalg.norm(expected))
        errors.append({"corner_relative": corner_error, "probe_relative": probe_error})
    report.update(same_load_errors=errors, probe_coordinate_controls=coordinate_controls)
    write_json(args.output / "result.json", report)
    assert all(max(row.values()) < 1e-7 for row in errors), errors
    report["passed"] = True
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
