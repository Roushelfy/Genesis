"""Compare full and complete-boundary native inverse application at fixed FP64 accuracy."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.solve import kernel_balance, kernel_full_residual, kernel_peak
from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverse
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.peak_cpu import P2Peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    mesh = Path("examples/rigid/assets/hollow_egg/level1/elastic.npz")
    model = StressModel(RigidStressOptions(mesh=mesh))
    nodes = np.unique(qd_to_numpy(model.info.surface_nodes))
    candidate = StressSurfaceInverse(model.info, model.factor, model.create_state, nodes)
    state = model.create_state(args.envs)
    random = np.random.default_rng(10917)
    force = np.zeros((model.info.vertices.shape[0], args.envs, 3))
    force[nodes] = random.normal(size=(len(nodes), args.envs, 3)) * 0.01
    force[:, 0] = 0
    angular_velocity = random.normal(size=(args.envs, 3)) * 2
    angular_velocity[0] = 0
    omega = V_VEC(3, dtype=gs.qd_float, shape=(args.envs,))
    omega.from_numpy(angular_velocity)
    state.force.from_numpy(force)
    kernel_balance(omega, state, model.info)
    model.inverse.apply(model.options.young, state)
    kernel_peak(model.options.young, model.options.poisson, state, model.info)
    full_displacement = qd_to_numpy(state.displacement, transpose=True, copy=True)
    full_peak = qd_to_numpy(state.peak, copy=True)
    candidate.apply(model.options.young, omega, state)
    kernel_full_residual(
        model.options.young,
        model.options.tolerance,
        model.options.absolute_tolerance,
        state,
        model.info,
        only_active=False,
    )
    kernel_peak(model.options.young, model.options.poisson, state, model.info)
    candidate_displacement = qd_to_numpy(state.displacement, transpose=True, copy=True)
    candidate_peak = qd_to_numpy(state.peak, copy=True)
    assert qd_to_numpy(state.valid).all()
    np.testing.assert_allclose(candidate_peak, full_peak, rtol=1e-4, atol=1e-3)
    with np.load(mesh) as asset:
        oracle = P2Shell(
            asset["vertices"],
            asset["tetrahedra"],
            asset["surface_triangles"],
            1e10,
            0.3,
            2000.0,
            2,
            factor_backend="none",
        )
    sample_count = min(args.envs, 5)
    loads = force[:, :sample_count].transpose(1, 0, 2).reshape(sample_count, -1).T
    centrifugal = -np.cross(
        angular_velocity[:sample_count, None],
        np.cross(angular_velocity[:sample_count, None], (oracle.xyz - oracle.com)[None]),
    )
    raw = loads + oracle.m @ centrifugal.reshape(sample_count, -1).T
    rhs = raw - oracle.mr @ np.linalg.solve(oracle.gram, oracle.r.T @ raw)
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins))
    displacement = np.zeros_like(rhs)
    displacement[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    actual = candidate_displacement[:sample_count].reshape(sample_count, -1).T
    residual = np.linalg.norm(oracle.k @ actual - rhs, axis=0)
    assert (residual <= np.maximum(1e-11, 1e-8 * np.linalg.norm(rhs, axis=0))).all()
    scan = P2Peak(oracle.glambda, oracle.elements, 1e10 / 2.6)
    cpu_peak = np.array([scan(displacement[:, i])[0] for i in range(sample_count)])
    np.testing.assert_allclose(candidate_peak[:sample_count], cpu_peak, rtol=1e-4, atol=1e-3)
    timings = {}
    for name, operation in (
        ("full", lambda: model.inverse.apply(model.options.young, state)),
        ("surface", lambda: candidate.apply(model.options.young, omega, state)),
    ):
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        timings[name] = 1e3 * (time.perf_counter() - started) / args.repetitions
    report = {
        "envs": args.envs,
        "nodes": len(force),
        "exterior_nodes": len(nodes),
        "application_ms": timings,
        "peak_difference_Pa": float(np.max(abs(candidate_peak - full_peak))),
        "displacement_relative_error": float(
            np.linalg.norm(candidate_displacement - full_displacement) / np.linalg.norm(full_displacement)
        ),
        "complete_residual_max_N": float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max()),
        "independent_cpu_residual_N": residual.tolist(),
        "independent_cpu_peak_error_Pa": (candidate_peak[:sample_count] - cpu_peak).tolist(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
