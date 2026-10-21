"""Check the scene bridge against independent elasticity, frame covariance and partial failures."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegions
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step, kernel_reset
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import runtime_identity, write_json
from .check_hertz_transfer import voxel_mesh
from .live_hertz_inputs import array, kernel_estimate_inputs
from .live_hertz_scene import (
    SceneObservationState,
    kernel_accept_surface_load,
    kernel_finish_observation,
    kernel_prepare_patches,
)
from .native_hertz import NativeHertz
from .native_hertz_transfer import HertzSurfaceTransfer
from .p2_elastic import FreeElasticP2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu")})
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    epsilon = float(np.finfo(gs.np_float).eps)
    vertices, tets, triangles = voxel_mesh("face")
    mesh = args.output / "mesh.npz"
    np.savez_compressed(mesh, vertices=vertices, tetrahedra=tets, surface_triangles=triangles)
    model = StressModel(RigidStressOptions(mesh=mesh, method="direct"))
    surface = StressSurface(3, model.info)
    contacts, sources = create_contacts(1, 3, 1e-4), create_sources(1, 3)
    regions = StressHertzRegions(contacts).state
    native, transfer = NativeHertz(3), HertzSurfaceTransfer(3, 3)
    observation = SceneObservationState(
        status=V(dtype=gs.qd_int, shape=(3,)),
        is_loaded=V(dtype=gs.qd_bool, shape=(3,)),
        friction=V(dtype=gs.qd_float, shape=(3,)),
        required_friction=V(dtype=gs.qd_float, shape=(3,)),
        counts=V_VEC(9, dtype=gs.qd_int, shape=(3,)),
    )
    states = [model.create_state(3, mode) for mode in ("max", "full")]
    omega = V_VEC(3, dtype=gs.qd_float, shape=(3,))
    omega.fill(0)
    rotations = Rotation.from_euler("xyz", [[0, 0, 0], [21, -37, 19], [-45, 13, 68]], degrees=True)
    origins = np.array([[0.0, 0, 0], [0.03, -0.02, 0.01], [-0.02, 0.04, -0.01]])
    center = np.broadcast_to([0.0003, 0.0012, 0.001], (3, 3)).copy()
    normal, tangent = np.broadcast_to([0.0, 0, 1], (3, 3)), np.broadcast_to([1.0, 0, 0], (3, 3))
    force = np.array([[0.0, 0, -0.1], [0.02, 0, -0.1], [0, -0.02, -0.1]])
    contacts.frame_position.from_numpy(origins)
    contacts.frame_quaternion.from_numpy(rotations.as_quat()[:, [3, 0, 1, 2]])
    contacts.position.from_numpy(center[None])
    contacts.normal.from_numpy(-normal[None].copy())
    contacts.friction.fill(0.5)
    sources.geom.fill(0)
    sources.pair.fill(1)
    regions.center.from_numpy((rotations.apply(center) + origins)[None])
    regions.normal.from_numpy(rotations.apply(normal)[None])
    regions.tangent.from_numpy(rotations.apply(tangent)[None])
    regions.curvature.from_numpy(np.broadcast_to([[50.0, 0], [0, 80.0]], (1, 3, 2, 2)).copy())
    regions.leader.fill(0)
    reference = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000)
    coordinates = array(model.info.vertices)
    permutation = reference.reference.native_dof_permutation(coordinates).reshape(-1, 3)[:, 0] // 3
    controls = []
    baseline = None
    for name in ("loaded", "release", "source_failure", "moment_failure", "cone_failure", "recovered"):
        valid = np.ones((1, 3), dtype=gs.np_bool)
        counts = np.ones((1, 3), dtype=gs.np_int)
        applied, witnesses = force.copy(), center.copy()
        moments = np.zeros((1, 3, 3))
        expected = np.zeros(3, dtype=gs.np_int)
        if name == "release":
            valid[0, 1], counts[0, 1], applied[1] = False, 0, 0
        elif name == "source_failure":
            witnesses[1, 2] += 0.01
            expected[1] = 2 | 4
        elif name == "moment_failure":
            moments[0, 1, 0] = 1e-7
            expected[1] = 4
        elif name == "cone_failure":
            applied[1, 0] = 0.06
            expected[1] = 8
        contacts.valid.from_numpy(valid)
        contacts.force.from_numpy(applied[None])
        sources.witness.from_numpy(witnesses[None])
        regions.source_count.from_numpy(counts)
        regions.is_valid.fill(True)
        regions.force.from_numpy(rotations.apply(applied)[None])
        regions.moment.from_numpy(moments)
        kernel_resolve_sources(1e-4, epsilon, contacts, sources, model.info, surface.info)
        kernel_estimate_inputs(regions, native.state)
        native.update()
        for state in states:
            kernel_begin_step(state)
            kernel_prepare_patches(
                contacts, sources, regions, transfer.patches, observation, model.info, surface.info, epsilon
            )
            transfer.scatter(native.state, state, model.info, surface.info, epsilon)
            kernel_accept_surface_load(transfer.patches, transfer.samples, observation, state, surface.info, epsilon)
            model.recover(omega, state, surface_load=True)
            kernel_finish_observation(observation, state, state.stress_tensor.shape[0] > 0)
            np.testing.assert_array_equal(array(observation.status), expected)
            np.testing.assert_array_equal(array(state.step_valid), expected == 0)
        np.testing.assert_allclose(array(states[0].step_peak), array(states[1].step_peak), rtol=1e-11, atol=1e-8)
        nodal, field = array(states[1].force), array(states[1].stress_tensor)
        np.testing.assert_array_equal(nodal[expected != 0], 0)
        assert np.isnan(field[expected != 0]).all()
        errors = []
        for env in np.flatnonzero(expected == 0):
            np.testing.assert_allclose(nodal[env].sum(axis=0), applied[env], rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(np.cross(coordinates - center[env], nodal[env]).sum(axis=0), 0, atol=1e-15)
            load = np.empty_like(nodal[env])
            load[permutation] = nodal[env]
            expected_tensor, _, _, residual = reference.solve(load)
            actual = field[env][..., [[0, 3, 4], [3, 1, 5], [4, 5, 2]]]
            error = np.linalg.norm(actual - expected_tensor) / max(np.linalg.norm(expected_tensor), 1)
            assert error < 1e-8 and np.linalg.norm(residual) < 1e-7 * max(np.linalg.norm(load), 1e-10)
            errors.append(float(error))
        if name == "loaded":
            baseline = field.copy()
            peak_before = array(states[1].step_peak)
            kernel_reset(np.array([1], dtype=gs.np_int), states[1])
            np.testing.assert_array_equal(array(states[1].stress_tensor)[[0, 2]], baseline[[0, 2]])
            np.testing.assert_array_equal(array(states[1].step_peak)[[0, 2]], peak_before[[0, 2]])
            assert np.isnan(array(states[1].stress_tensor)[1]).all()
        else:
            np.testing.assert_allclose(field[[0, 2]], baseline[[0, 2]], rtol=1e-12, atol=1e-9)
        if name == "recovered":
            np.testing.assert_allclose(field, baseline, rtol=1e-12, atol=1e-9)
        controls.append({"name": name, "status": expected.tolist(), "independent_field_relative_errors": errors})
    np.savez_compressed(args.output / "fields.npz", baseline=baseline, recovered=field)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "controls": controls,
            "scope": "Three material frames, normal and two-direction proportional shear, original force/moment, same-load independent complete P2 fields, max/full reduction, release, source/moment/cone failure isolation, partial reset and recovery. This verifies the scene bridge; Hertz pressure prediction, continuum peak resolution and friction memory require separate physical evidence.",
        },
    )


if __name__ == "__main__":
    main()
