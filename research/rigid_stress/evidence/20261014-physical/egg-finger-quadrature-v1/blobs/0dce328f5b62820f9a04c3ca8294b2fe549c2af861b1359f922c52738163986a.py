"""Restore an exact archived spatial mesh and test wide-coordinate inertial recovery."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .franka_contact_reference import RigidFingerContactSystem, build_from_capture
from .franka_rigid_mode_coordinates import coordinate_map
from .refined_franka_contact import refine_contact
from .skfem_reference import SkfemElasticReference
from .sparse_franka_inertia import SparseInertialRigidFingerContactSystem
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk


def restore_static(run):
    closed_source(run)
    config = json.loads((run / "results/configuration.json").read_text())
    closure_config = json.loads((Path(config["closure_run"]) / "results/configuration.json").read_text())
    seed, _ = build_from_capture(Path(closure_config["capture"]), closure_config["dhat"])
    assert not config.get("freeze_contact", False)
    with np.load(run / "results/reference_geometry.npz") as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        fixed = saved["fixed_initializer_points_m"].copy()
        components = saved["fixed_initializer_components"].copy()
    distance, nodes = cKDTree(body.coordinates).query(fixed)
    assert distance.max() < 1e-12
    body.gauge_dofs = body.component_dofs[nodes, components]
    assert len(np.unique(body.gauge_dofs)) == 6
    body.free_dofs = np.setdiff1d(np.arange(body.basis.N), body.gauge_dofs)
    vertices, faces, centres = [], [], []
    for index, indices in enumerate(seed.finger_nodes):
        local = indices[:-1]
        selected = seed.faces[seed.groups[seed.faces[:, 0]] == index + 1]
        assert np.all(np.isin(selected, local))
        vertices.append(seed.rest[local])
        faces.append(selected - local[0])
        centres.append(seed.rest[indices[-1]])
    system = RigidFingerContactSystem(body, vertices, faces, seed.axes, seed.masses, centres, seed.dhat, seed.friction)
    system.world_coordinate_origin_m = seed.world_coordinate_origin_m.copy()
    system.fixed_initializer_points_m = fixed
    system.fixed_initializer_components = components
    system = refine_contact(system, config.get("contact_refinement", 0), config.get("finger_refinement", 0))
    with np.load(run / "results/contact_geometry.npz") as saved:
        for key, actual in [
            ("rest_positions", system.contact_surface_rest),
            ("faces", system.contact_surface_faces),
            ("groups", system.contact_surface_groups),
        ]:
            np.testing.assert_array_equal(saved[key], actual)
    mapping = sparse.load_npz(run / "results/contact_node_map.npz")
    assert mapping.shape == system.contact_node_map.shape
    for key in ["data", "indices", "indptr"]:
        np.testing.assert_array_equal(getattr(mapping, key), getattr(system.contact_node_map, key))
    return system, config


class WideSpatialInertia(SparseInertialRigidFingerContactSystem):
    def __init__(self, static, dt):
        super().__init__(static, dt)
        self.bulk = WidePiolaBulk(static.body)

    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        energy, gradient, matrix, normal, contact = super().evaluate(
            displacement, previous, tangential, hessian, project_hessian, external_force
        )
        n = self.body.basis.N
        permutation = self.parts[0].permutation
        vector = np.empty(n, dtype=np.longdouble)
        vector[permutation] = displacement[:n]
        body_gradient = self.bulk.evaluate(vector, False)[1][permutation] + contact[:n]
        if external_force is not None:
            body_gradient -= external_force[:n]
        increment = np.asarray(displacement, dtype=np.longdouble) - self.predicted_displacement
        body_gradient = body_gradient + (self.inertial_mass @ increment)[:n] / self.dt**2
        gradient = np.asarray(gradient, dtype=float)
        gradient[:n] = body_gradient
        return energy, gradient, matrix, normal, contact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert int((args.run / "returncode.txt").read_text()) != 0
    args.output.mkdir(parents=True, exist_ok=False)
    static, config = restore_static(args.run)
    system = WideSpatialInertia(static, config["dt"])
    system.q_target[:] = 0.018
    dt = config["dt"]
    paths = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
    assert paths
    with np.load(paths[-1]) as saved:
        previous = np.asarray(saved["displacement"], dtype=np.longdouble)
        velocity = np.asarray(saved["velocity"], dtype=np.longdouble)
    with np.load(args.run / "results/failure.npz") as failed:
        np.testing.assert_array_equal(failed["rest_positions"], static.rest)
        np.testing.assert_array_equal(failed["previous_displacement"], previous)
        displacement = np.asarray(failed["displacement"], dtype=np.longdouble)
    system.predicted_displacement = previous + dt * velocity
    mapping = coordinate_map(system)
    external = system.gravity_force([0, 0, -9.81])
    initial = system.evaluate_updated_friction(displacement, previous, external, False)
    result = {
        "scope": __doc__,
        "source_run": str(args.run),
        "original_accepted_steps": len(paths),
        "body_tetrahedra": system.body.mesh.nelements,
        "initial_complete_residual_N": float(np.linalg.norm(mapping.T @ initial[1])),
        "original_external_budget_N": 1e-9,
        "internal_target_N": 1e-10,
        "goal_completion": False,
    }
    try:
        u, gradient, force, _energy, _iterations = system.solve_updated_friction(
            np.array([], dtype=int),
            np.array([]),
            1,
            tolerance=1e-10,
            initial_displacement=displacement,
            initial_previous_displacement=previous,
            coordinate_map=mapping,
            applied_force=external,
            initial_force=external,
            phase="same_spatial_failed_state",
            failure_path=args.output / "failure.npz",
        )
        result["recovered"] = True
        result["recovered_complete_residual_N"] = float(np.linalg.norm(mapping.T @ gradient))
        result["recovered_body_residual_N"] = float(np.linalg.norm(gradient[: system.body.basis.N]))
        assert result["recovered_body_residual_N"] <= 1e-9
        np.savez_compressed(
            args.output / "recovered.npz",
            displacement=u,
            contact_force_N=force,
            previous_displacement=previous,
            previous_velocity=velocity,
        )
    except RuntimeError as error:
        result["recovered"] = False
        result["recovery_error"] = str(error)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
