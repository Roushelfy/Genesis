"""Check visible half-space/clearance refusal on a connected nonconvex link."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
from skfem import MeshTet

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import runtime_identity, write_json
from .check_native_hertz_recovery import array
from .hertz_reference import ellipse


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "scope": "Connected L-shaped material. A valid global outer flat contact is retained; a real reentrant ledge source with material across its tangent plane and an oversized footprint are refused without changing their identities. Test same-step failure retention, next-step release/recovery, source motion and invalid normal input. No nonconvex physical stress qualification.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    axes, _, _, _ = ellipse(0.2, 1e10 / 0.91, np.diag([50.0, 80.0]))
    a = axes[0]
    block = MeshTet.init_tensor(a * np.array([-20, 0, 20]), a * np.array([-20, 0, 20]), a * np.array([0, 20, 40]))
    vertices, tets = block.p.T, block.t.T
    centers = vertices[tets].mean(axis=1)
    tets = tets[~((centers[:, 0] > 0) & (centers[:, 2] > 20 * a))]
    used, inverse = np.unique(tets, return_inverse=True)
    vertices, tets = vertices[used], inverse.reshape(-1, 4)
    mesh = MeshTet(vertices.T, tets.T, sort_t=False)
    path = args.output / "mesh.npz"
    np.savez_compressed(
        path, vertices=vertices, tetrahedra=tets, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    model = StressModel(RigidStressOptions(mesh=path))
    surface = StressSurface(4, model.info)
    law = StressHertz(3)
    law.state.force.from_numpy(np.array([0.2, 0.2, 50.0]))
    law.state.curvature.from_numpy(np.broadcast_to(np.diag([50.0, 80.0]), (3, 2, 2)).copy())
    law.state.modulus.fill(1e10 / 0.91)
    law.update()
    module = StressHertzRecovery(3, 1, 0, "full")
    center = a * np.array([[-5, 3, 0], [10, 3, 20], [-5, 3, 0]])
    frames = np.broadcast_to(np.eye(3), (3, 3, 3)).copy()
    frames[1] = np.diag([1.0, -1.0, -1.0])
    origin, dual, normal = (
        array(surface.info.face_origin),
        array(surface.info.face_dual),
        array(surface.info.face_normal),
    )
    local = np.einsum("fij,pfj->pfi", dual, center[:, None] - origin)
    bary = np.concatenate([1 - local.sum(axis=2, keepdims=True), local], axis=2)
    valid = (bary.min(axis=2) >= -1e-10) & (np.abs(np.einsum("pfj,fj->pf", center[:, None] - origin, normal)) < 1e-12)
    valid &= np.einsum("pj,fj->pf", frames[:, :, 2], normal) < -1 + 1e-10
    assert valid.any(axis=1).all()
    anchor = valid.argmax(axis=1)
    module.frames.center.from_numpy(center)
    module.frames.basis.from_numpy(frames)
    module.frames.anchor.from_numpy(anchor.astype(gs.np_int))
    state = model.create_state(3, "full")
    omega, errno = V_VEC(3, dtype=gs.qd_float, shape=(3,)), V(dtype=gs.qd_int, shape=(3,))
    omega.fill(0)
    errno.fill(0)
    kernel_begin_step(state)
    module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
    qd.sync()
    status = array(module.state.status)
    np.testing.assert_array_equal(status, [0, 3, 3])
    np.testing.assert_array_equal(array(state.invalid_steps), [0, 1, 1])
    assert np.isnan(array(state.stress_tensor)[1:]).all() and np.all(array(state.force)[1:] == 0)
    np.testing.assert_array_equal(array(module.frames.center), center)
    np.testing.assert_array_equal(array(module.frames.anchor), anchor)
    first = array(state.stress_tensor).copy()
    np.savez_compressed(
        args.output / "refusal.npz",
        center_m=center,
        anchor=anchor,
        status=status,
        errno=array(errno),
        tensor_pa=first,
        force_n=array(state.force),
    )
    law.state.force.from_numpy(np.array([0.2, 0, 0.2]))
    law.update()
    module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
    qd.sync()
    np.testing.assert_array_equal(array(module.state.status), [0, 1, 0])
    np.testing.assert_array_equal(array(state.step_valid), [True, False, False])
    np.testing.assert_array_equal(array(state.invalid_steps), [0, 1, 1])
    kernel_begin_step(state)
    errno.fill(0)
    module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
    qd.sync()
    assert np.all(array(state.step_valid)) and not np.any(array(errno))
    np.testing.assert_array_equal(array(state.stress_tensor)[1], 0)
    np.testing.assert_allclose(array(state.stress_tensor)[0], first[0], rtol=1e-7, atol=1e-5)
    center[0, 0] += a
    module.frames.center.from_numpy(center)
    kernel_begin_step(state)
    module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
    qd.sync()
    np.testing.assert_array_equal(array(module.state.status), [0, 1, 0])
    assert np.all(array(state.step_valid)) and not np.any(array(errno))
    assert not np.allclose(array(state.stress_tensor)[0], first[0], rtol=1e-5, atol=1e-5)
    law.state.force.from_numpy(np.array([0.2, 0, float("nan")]))
    law.update()
    kernel_begin_step(state)
    module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
    qd.sync()
    np.testing.assert_array_equal(array(module.state.status), [0, 1, 2])
    np.testing.assert_array_equal(array(state.invalid_steps), [0, 1, 2])
    assert array(errno)[2] != 0 and np.isnan(array(state.stress_tensor)[2]).all()
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "initial_status": status.tolist(),
            "final_status": array(module.state.status).tolist(),
            "final_invalid_step_counts": array(state.invalid_steps).tolist(),
            "source_identity_preserved_on_refusal": True,
            "same_step_invalidity_next_step_release_recovery_and_motion_passed": True,
            "general_nonconvex_physics_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
