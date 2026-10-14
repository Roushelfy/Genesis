"""Objective deformable egg with actual rigid Panda fingers and jaw PD potential."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse

from .audit_franka_reference import rotation
from .contact_reference import ContactElasticPart, IndependentContactSystem, quadratic_boundary_triangles
from .skfem_reference import SkfemElasticReference
from .stable_neohookean_reference import StableNeoHookeanBulk
from .summarize_fine_references import closed_source


class RigidFingerContactSystem(IndependentContactSystem):
    """Use exact rigid finger coordinates; introduce no fictitious pad modulus."""

    def __init__(
        self, body, finger_vertices, finger_faces, axes, masses, centres, dhat, friction, barrier_stiffness=1e6
    ):
        self.parts = [ContactElasticPart("body", body, 0, body.component_dofs.reshape(-1))]
        self.body = body
        self.bulk = StableNeoHookeanBulk(body)
        positions, faces = [body.coordinates], [quadratic_boundary_triangles(body)]
        groups = [np.zeros(len(body.coordinates), dtype=int)]
        self.finger_nodes, self.finger_com_nodes = [], []
        offset = len(body.coordinates)
        for index, (vertices, triangles, centre) in enumerate(zip(finger_vertices, finger_faces, centres, strict=True)):
            self.finger_nodes.append(np.arange(offset, offset + len(vertices) + 1))
            self.finger_com_nodes.append(offset + len(vertices))
            positions.extend((vertices, np.array(centre)[None]))
            faces.append(triangles + offset)
            groups.append(np.full(len(vertices) + 1, index + 1))
            offset += len(vertices) + 1
        self.rest, self.faces, self.groups = np.concatenate(positions), np.concatenate(faces), np.concatenate(groups)
        edges = np.unique(np.sort(self.faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
        self.mesh = ipctk.CollisionMesh.build_from_full_mesh(self.rest, edges, self.faces)
        self.mesh.can_collide = ipctk.make_vertex_patches_filter(self.groups[self.mesh.to_full_vertex_id()])
        self.stiffness = sparse.block_diag(
            (
                body.stiffness[self.parts[0].permutation][:, self.parts[0].permutation],
                sparse.csc_matrix((3 * (offset - len(body.coordinates)),) * 2),
            ),
            format="csc",
        )
        self.axes, self.masses = np.array(axes), np.array(masses)
        np.testing.assert_allclose(np.linalg.norm(self.axes, axis=1), 1, rtol=0, atol=1e-12)
        self.q_reference = np.array([0.04, 0.04])
        self.q_target = self.q_reference.copy()
        self.kp, self.force_cap = 100.0, 4.0
        self.dhat, self.friction = dhat, friction
        self.barrier = ipctk.BarrierPotential(dhat, barrier_stiffness, use_physical_barrier=True)
        self.friction_potential = ipctk.FrictionPotential(1e-8)
        self.accepted_steps = []
        self.reuse_empty_factor = False
        self.supported_factorizations = self.empty_factor_reuses = 0

    def elastic_displacement(self, displacement):
        elastic = np.zeros_like(displacement)
        count = self.body.basis.N
        vector = np.empty(count)
        vector[self.parts[0].permutation] = displacement[:count]
        elastic[:count] = self.body.align_rigid_gauge(vector)[self.parts[0].permutation]
        return elastic

    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        energy, gradient, matrix, normal, contact_gradient = super().evaluate(
            displacement, previous, tangential, hessian, project_hessian, external_force
        )
        count = self.body.basis.N
        vector = np.empty(count)
        vector[self.parts[0].permutation] = displacement[:count]
        linear = self.body.align_rigid_gauge(vector)
        old_gradient = self.body.stiffness @ linear
        try:
            new_energy, new_gradient, new_matrix = self.bulk.evaluate(vector, hessian)
        except ValueError:
            if hessian:
                raise
            return float("inf"), np.full_like(gradient, np.inf), None, normal, contact_gradient
        energy += new_energy - 0.5 * linear @ old_gradient
        gradient[:count] += (new_gradient - old_gradient)[self.parts[0].permutation]
        if hessian:
            delta = new_matrix - self.body.stiffness
            matrix += sparse.block_diag(
                (
                    delta[self.parts[0].permutation][:, self.parts[0].permutation],
                    sparse.csc_matrix((len(displacement) - count,) * 2),
                ),
                format="csc",
            )
        actuator_hessian = sparse.lil_matrix(self.stiffness.shape) if hessian else None
        for index, node in enumerate(self.finger_com_nodes):
            dofs = np.arange(3 * node, 3 * node + 3)
            q = self.q_reference[index] + displacement[dofs] @ self.axes[index]
            error = q - self.q_target[index]
            transition = self.force_cap / self.kp
            force = np.clip(self.kp * error, -self.force_cap, self.force_cap)
            energy += (
                0.5 * self.kp * error**2
                if abs(error) <= transition
                else self.force_cap * (abs(error) - 0.5 * transition)
            )
            gradient[dofs] += force * self.axes[index]
            if hessian and abs(error) < transition:
                actuator_hessian[np.ix_(dofs, dofs)] = self.kp * np.outer(self.axes[index], self.axes[index])
        if hessian:
            matrix += actuator_hessian.tocsc()
        return energy, gradient, matrix, normal, contact_gradient

    def coordinate_map(self, hold_body):
        count = self.body.basis.N
        omitted = (
            np.flatnonzero(np.isin(self.parts[0].permutation, self.body.gauge_dofs))
            if hold_body
            else np.array([], dtype=int)
        )
        independent = np.setdiff1d(np.arange(count), omitted)
        rows, cols, data = list(independent), list(range(len(independent))), [1.0] * len(independent)
        for index, nodes in enumerate(self.finger_nodes):
            for axis in range(3):
                rows.extend((3 * nodes + axis).tolist())
                cols.extend([len(independent) + index] * len(nodes))
                data.extend([self.axes[index, axis]] * len(nodes))
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(self.rest) * 3, len(independent) + 2)), omitted

    def gravity_force(self, gravity, body_fraction=1.0):
        force = np.zeros(self.stiffness.shape[0])
        acceleration = np.zeros(self.body.basis.N)
        acceleration[self.body.component_dofs] = gravity
        force[: self.body.basis.N] = body_fraction * (self.body.mass @ acceleration)[self.parts[0].permutation]
        for node, mass in zip(self.finger_com_nodes, self.masses, strict=True):
            force[3 * node : 3 * node + 3] = mass * np.array(gravity)
        return force


def build_from_capture(run, dhat):
    closed_source(run)
    assert int((run / "returncode.txt").read_text()) == 0
    report = json.loads((run / "results/summary.json").read_text())
    row = [r for r in report["captures"] if r["mode"] == "stationary_lift_hold"][-1]
    body_rotation = rotation(row["egg_quaternion_wxyz"])
    path = run / "source/examples/rigid/assets/hollow_egg/level1/elastic.npz"
    with np.load(path) as saved:
        vertices = saved["vertices"] @ body_rotation.T + row["egg_position_m"]
        body = SkfemElasticReference(vertices, saved["tetrahedra"], 1e10, 0.3, 2000)
    np.testing.assert_allclose(body.total_mass, row["egg_mass_kg"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(body.com, row["egg_world_com_m"], rtol=0, atol=1e-12)
    axes, positions, faces, centres = [], [], [], []
    with np.load(run / "results" / row["geometry_snapshot"]) as world:
        for index, finger in enumerate(row["fingers"]):
            axis = rotation(finger["world_quaternion_wxyz"]) @ [0, 1, 0]
            axes.append(axis)
            shift = (0.04 - row["finger_position_m"][index]) * axis
            group_vertices, group_faces, offset = [], [], 0
            for geom in report["geometry"]:
                if geom["link"] != finger["link"]:
                    continue
                vertices = world[f"geom{geom['geom']}_world_vertices_m"] + shift
                with np.load(run / "results" / geom["mesh"]) as local:
                    triangles = local["triangles"]
                assert triangles.max() < len(vertices)
                group_vertices.append(vertices)
                group_faces.append(triangles + offset)
                offset += len(vertices)
            assert len(group_vertices) == 6
            positions.append(np.concatenate(group_vertices))
            faces.append(np.concatenate(group_faces))
            centres.append(np.array(finger["world_position_m"]) + shift)
    mu = {contact["friction"] for contact in row["contacts"]}
    assert len(mu) == 1
    return RigidFingerContactSystem(body, positions, faces, axes, row["finger_masses_kg"], centres, dhat, mu.pop()), row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--dhat", type=float, default=2.5e-6)
    parser.add_argument("--control-steps", type=int, default=64)
    parser.add_argument("--gravity-steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    system, _row = build_from_capture(args.capture, args.dhat)
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    np.savez_compressed(
        args.output / "geometry.npz",
        rest_positions=system.rest,
        faces=system.faces,
        groups=system.groups,
        body_vertices=system.body.mesh.p.T,
        body_tetrahedra=system.body.mesh.t.T,
        axes_world=system.axes,
    )
    initial = system.mesh.vertices(system.rest)
    assert not ipctk.has_intersections(system.mesh, initial)
    u = np.zeros(system.stiffness.shape[0])
    held_map, held = system.coordinate_map(True)
    external = system.gravity_force([0, 0, -9.81])
    for step in range(1, args.control_steps + 1):
        system.q_target[:] = 0.04 + (0.018 - 0.04) * step / args.control_steps
        (args.output / "control-progress.json").write_text(
            json.dumps({"step": step, "q_target_m": system.q_target.tolist()}) + "\n"
        )
        u, _, _, _, _ = system.solve_updated_friction(
            held,
            np.zeros(len(held)),
            1,
            initial_displacement=u,
            coordinate_map=held_map,
            applied_force=external,
            initial_force=external,
            phase="held_body_actual_pd_closure",
            failure_path=args.output / "closure_failure.npz",
        )
    free_map, _ = system.coordinate_map(False)
    # Release the temporary initializer with the same actuation and full gravity.
    u, g, f, energy, records = system.solve_updated_friction(
        np.array([], dtype=int),
        np.array([]),
        args.gravity_steps,
        initial_displacement=u,
        coordinate_map=free_map,
        applied_force=external,
        initial_force=external,
        phase="free_egg_fixed_hand_pd",
        failure_path=args.output / "force_failure.npz",
    )
    residual = float(np.linalg.norm(free_map.T @ g))
    assert residual <= 1e-9 and np.linalg.norm(g[: system.body.basis.N]) <= 1e-9
    vector = np.empty(system.body.basis.N)
    vector[system.parts[0].permutation] = u[: system.body.basis.N]
    tensor, vm, upper = system.bulk.stress_bounds(vector)
    q = [
        0.04 + u[3 * node : 3 * node + 3] @ axis
        for node, axis in zip(system.finger_com_nodes, system.axes, strict=True)
    ]
    np.savez_compressed(
        args.output / "fields.npz",
        displacement=u,
        contact_force_N=f,
        applied_force_N=external,
        body_tensor_world_Pa=tensor,
        body_vm_Pa=vm,
        body_component_permutation=system.parts[0].permutation,
    )
    output = {
        "scope": __doc__,
        "capture": str(args.capture),
        "complete_constrained_residual_N": residual,
        "body_residual_N": float(np.linalg.norm(g[: system.body.basis.N])),
        "jaw_position_m": q,
        "jaw_pd_force_N": [float(np.clip(100 * (0.018 - value), -4, 4)) for value in q],
        "body_raw_peak_lower_Pa": float(vm.max()),
        "body_global_peak_upper_Pa": float(upper.max()),
        "body_whole_volume_determinant_lower_bound": system.bulk.determinant_lower_bound(vector),
        "potential_J": float(energy),
        "accepted_steps": [asdict(r) for r in system.accepted_steps],
        "iterations": [asdict(r) for r in records],
        "goal_completion": False,
        "limitations": "Full actual rigid finger collision geometry, measured mass, slide axes, friction and18mm/100N-per-m/4N jaw actuation are retained. Hand pose is fixed to the selected actual lifted frame, and a temporary held-egg initializer is removed before final acceptance. This controlled quasistatic reference requires a corresponding rigid fixed-hand fixture and history checks; it is not yet full seven-arm-joint PD/dynamic equivalence or converged physical application acceptance.",
    }
    (args.output / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ("accepted_steps", "iterations")}), flush=True)


if __name__ == "__main__":
    main()
