"""Check native curved Hertz transfer and material-following probes on captured input."""

import argparse
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd
from scipy.spatial import cKDTree
from skfem import Basis, ElementTetP2, MeshTet1

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from ..mechanics import shell_mesh
from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array, matrix
from .study_franka_contact_gap import von_mises
from .study_quadratic_rest_surface import quadratic_basis


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {
        "wrench": 1e-4,
        "refined_point_load": 1e-5,
        "reference_field": 1e-3,
        "reference_peak": 1e-3,
        "probe_location_m": 1e-12,
        "max_full": 1e-7,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "frame": file_id(args.frame),
            "geometry": file_id(args.geometry),
            "reference": file_id(args.reference),
            "targets": targets,
            "scope": "Native curved P2 point-boundary correction and 89 material-following probes for the captured first normal Franka source. Three environments include scaled and released loads. Compare retained independently assembled curved elasticity and distributed-Hertz corrections, then max/full, rejection isolation, release and recovery. This is a supplied normal-source check, not a live moving/friction scene or independent contact-pressure validation.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    with np.load(args.frame) as data:
        center, frame = data["hertz_centers_m"][0], data["hertz_frames"][0]
        direction, axes, pressure = data["law_direction"][0], data["law_axes_m"][0], data["law_pressure_pa"][0]
        force, anchor = data["law_force_n"][0], int(data["hertz_anchors"][0])
    principal = frame.copy()
    principal[:, :2] = frame[:, :2] @ np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
    xyz, tets, faces, _ = shell_mesh(1, 2, 0.0005)
    edges, inverse = np.unique(
        np.sort(tets[:, list(combinations(range(4), 2))].reshape(-1, 2), axis=1), axis=0, return_inverse=True
    )
    scalar = Basis(MeshTet1(xyz.T, tets.T, sort_t=False), ElementTetP2())
    rows = []
    for kind in ["analytic", "estimated"]:
        with np.load(args.geometry) as data:
            nodes = data[kind + "_volume_nodes_m"]
        edge_positions = np.empty((len(edges), 3))
        edge_positions[inverse] = nodes[:, 4:].reshape(-1, 3)
        asset = args.output / (kind + ".npz")
        np.savez_compressed(
            asset, vertices=xyz, tetrahedra=tets, surface_triangles=faces, edge_positions=edge_positions
        )
        model = StressModel(RigidStressOptions.hertz(mesh=asset, method="direct"))
        surface = StressSurface(16, model.info)
        law = StressHertz(3)
        scale = np.array([1.0, 0.5, 0.0])
        law.state.force.from_numpy(force * scale)
        law.state.axes.from_numpy(axes[None] * np.cbrt(scale)[:, None])
        law.state.peak_pressure.from_numpy(pressure * np.cbrt(scale))
        law.state.direction.from_numpy(np.tile(direction, (3, 1)))
        law.state.status.from_numpy(np.array([0, 0, 1], dtype=gs.np_int))
        omega, errno = V_VEC(3, dtype=gs.qd_float, shape=(3,)), V(dtype=gs.qd_int, shape=(3,))
        omega.fill(0)
        modules, states, fields = [], [], []
        for mode in ["full", "max"]:
            module = StressHertzRecovery(3, 1, output_mode=mode, moving_probes=True)
            module.frames.center.from_numpy(np.tile(center, (3, 1)))
            module.frames.basis.from_numpy(np.tile(frame, (3, 1, 1)))
            module.frames.anchor.fill(anchor)
            state = model.create_state(3, mode)
            kernel_begin_step(state)
            errno.fill(0)
            module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
            qd.sync()
            path = args.output / (kind + "-" + mode + ".npz")
            np.savez_compressed(
                path,
                status=array(module.state.status),
                errno=array(errno),
                wrench=array(module.state.wrench_error),
                forces_n=array(state.force),
                peak_pa=array(state.peak),
                tensile_peak_pa=array(module.state.tensile_peak),
                corners_pa=array(state.stress_tensor) if mode == "full" else np.empty(0),
                probes_pa=array(module.probes.tensor) if mode == "full" else np.empty(0),
                probe_position_m=array(module.probes.position),
                probe_element=array(module.probes.element),
                probe_barycentric=array(module.probes.barycentric),
                displacement_m=array(state.displacement),
            )
            write_json(
                args.output / "admission.json",
                {
                    "geometry": kind,
                    "mode": mode,
                    "status": array(module.state.status).tolist(),
                    "errno": array(errno).tolist(),
                    "wrench": array(module.state.wrench_error).tolist(),
                    "fields": file_id(path),
                },
            )
            np.testing.assert_array_equal(array(module.state.status), [0, 0, 1])
            np.testing.assert_array_equal(array(errno), 0)
            assert array(state.valid).all()
            assert array(module.state.wrench_error).max() < targets["wrench"]
            modules.append(module)
            states.append(state)
            if mode == "full":
                fields = [matrix(array(state.stress_tensor)), matrix(array(module.probes.tensor))]
        np.testing.assert_allclose(array(states[0].peak), array(states[1].peak), rtol=targets["max_full"], atol=1e-8)
        np.testing.assert_allclose(
            array(modules[0].state.tensile_peak),
            array(modules[1].state.tensile_peak),
            rtol=targets["max_full"],
            atol=1e-8,
        )
        np.testing.assert_array_equal(array(states[0].force)[2], 0)
        module, state = modules[0], states[0]
        native_nodes = array(model.info.vertices)
        reference_nodes = np.empty((scalar.N, 3))
        reference_nodes[scalar.element_dofs.T] = (nodes[:, [0, 1, 2, 3, 4, 7, 5, 6, 8, 9]] - center) @ principal
        distance, permutation = cKDTree(reference_nodes).query((native_nodes - center) @ principal)
        assert distance.max() < 1e-12 and len(np.unique(permutation)) == len(permutation)
        with np.load(args.reference) as data:
            reference_load = data[kind + "_last_loads_n"][0, permutation] @ principal.T
            reference_corners = data[kind + "_corners_pa"][-1]
            reference_probes = data[kind + "_probes_pa"][-1]
            reference_positions = data[kind + "_probe_positions_m"]
        load_error = np.linalg.norm(array(state.force)[0] - reference_load) / np.linalg.norm(reference_load)
        corners, probes = principal.T @ fields[0][0] @ principal, principal.T @ fields[1][0] @ principal
        errors = {
            "corner_relative": float(np.linalg.norm(corners - reference_corners) / np.linalg.norm(reference_corners)),
            "probe_relative": float(np.linalg.norm(probes - reference_probes) / np.linalg.norm(reference_probes)),
        }
        actual = np.concatenate([corners.reshape(-1, 3, 3), probes])
        expected = np.concatenate([reference_corners.reshape(-1, 3, 3), reference_probes])
        peak_error = abs(von_mises(actual).max() / von_mises(expected).max() - 1)
        tensile_error = abs(np.linalg.eigvalsh(actual)[:, -1].max() / np.linalg.eigvalsh(expected)[:, -1].max() - 1)
        local = (array(module.probes.position)[0] - center) @ principal
        position_error = np.linalg.norm(local - reference_positions, axis=1).max()
        chosen, bary = array(module.probes.element)[:2], array(module.probes.barycentric)[:2]
        values, _, _ = quadratic_basis(bary.reshape(-1, 4))
        located = np.einsum("qn,qni->qi", values, nodes[chosen.reshape(-1)]).reshape(2, -1, 3)
        inverse_error = np.linalg.norm(located - array(module.probes.position)[:2], axis=2).max()
        report = {
            "geometry": kind,
            "refined_point_load_relative": float(load_error),
            **errors,
            "von_mises_peak_relative": float(peak_error),
            "tensile_peak_relative": float(tensile_error),
            "probe_reference_position_error_m": float(position_error),
            "probe_inverse_error_m": float(inverse_error),
            "wrench_relative": array(module.state.wrench_error).tolist(),
        }
        rows.append(report)
        write_json(args.output / "progress.json", {"rows": rows})
        assert load_error < targets["refined_point_load"] and max(errors.values()) < targets["reference_field"], report
        assert max(peak_error, tensile_error) < targets["reference_peak"]
        assert max(position_error, inverse_error) < targets["probe_location_m"]
        original_corner, original_probe = fields
        centers = np.tile(center, (3, 1))
        centers[1] += frame[:, 2] * 1e-4
        module.frames.center.from_numpy(centers)
        kernel_begin_step(state)
        errno.fill(0)
        module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
        np.testing.assert_array_equal(array(module.state.status), [0, 2, 1])
        assert array(errno)[1] != 0 and np.all(array(errno)[[0, 2]] == 0)
        np.testing.assert_allclose(matrix(array(state.stress_tensor))[0], original_corner[0], rtol=1e-7, atol=1e-5)
        law.state.status.fill(1)
        law.state.force.fill(0)
        kernel_begin_step(state)
        errno.fill(0)
        module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
        np.testing.assert_array_equal(array(module.state.status), 1)
        np.testing.assert_array_equal(array(state.peak), 0)
        np.testing.assert_array_equal(array(state.force), 0)
        law.state.status.from_numpy(np.array([0, 0, 1], dtype=gs.np_int))
        law.state.force.from_numpy(force * scale)
        module.frames.center.from_numpy(np.tile(center, (3, 1)))
        kernel_begin_step(state)
        errno.fill(0)
        module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
        np.testing.assert_array_equal(array(module.state.status), [0, 0, 1])
        np.testing.assert_array_equal(array(errno), 0)
        np.testing.assert_allclose(matrix(array(state.stress_tensor)), original_corner, rtol=1e-7, atol=1e-5)
        np.testing.assert_allclose(matrix(array(module.probes.tensor))[:2], original_probe[:2], rtol=1e-7, atol=1e-5)
        qd.sync()
        samples = []
        for _ in range(3):
            started = perf_counter()
            kernel_begin_step(state)
            module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
            qd.sync()
            samples.append(perf_counter() - started)
        report.update(
            rejection_isolation_passed=True,
            release_and_recovery_passed=True,
            B3_normal_module_inclusive_seconds=samples,
        )
        write_json(args.output / "progress.json", {"rows": rows})
    write_json(
        args.output / "result.json",
        {"passed": True, "targets": targets, "rows": rows, "live_scene_qualified": False, "whole_goal_complete": False},
    )


if __name__ == "__main__":
    main()
