"""Independent quasistatic quadratic FEM coupled to IPC Toolkit contact and friction."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import ipctk
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from skfem import MeshTet

from .skfem_reference import SkfemElasticReference


@dataclass(frozen=True)
class ContactElasticPart:
    name: str
    reference: SkfemElasticReference
    first_node: int
    permutation: np.ndarray


@dataclass(frozen=True)
class ContactStepAcceptance:
    phase: str
    step: int
    friction_iterations: int
    complete_free_residual_n: float
    normal_collisions: int


@dataclass(frozen=True)
class ContactIteration:
    step: int
    friction_iteration: int
    newton_iteration: int
    energy_j: float
    free_residual_n: float
    normal_collisions: int
    collision_free_step: float
    accepted_step: float


def quadratic_boundary_triangles(reference):
    mesh = reference.mesh
    faces = mesh.facets[:, mesh.boundary_facets()].T
    edges = np.sort(mesh.edges.T, axis=1)
    edge_keys = edges[:, 0] * mesh.nvertices + edges[:, 1]
    order = np.argsort(edge_keys)
    local_edges = np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]], axis=2)
    keys = local_edges[..., 0] * mesh.nvertices + local_edges[..., 1]
    indices = order[np.searchsorted(edge_keys[order], keys)]
    mids = reference.scalar_basis.dofs.edge_dofs[0, indices]
    corners = reference.scalar_basis.dofs.nodal_dofs[0, faces]
    a, b, c = corners.T
    ab, ac, bc = mids.T
    return np.stack(
        (
            np.stack((a, ab, ac), axis=-1),
            np.stack((ab, b, bc), axis=-1),
            np.stack((ac, bc, c), axis=-1),
            np.stack((ab, bc, ac), axis=-1),
        ),
        axis=1,
    ).reshape(-1, 3)


class IndependentContactSystem:
    """Minimize independently assembled bulk elasticity plus contact potentials."""

    def __init__(self, references, dhat, barrier_stiffness, friction, friction_mollifier):
        self.parts = []
        stiffness, positions, faces, group_ids = [], [], [], []
        offset = 0
        for name, reference in references:
            permutation = reference.component_dofs.reshape(-1)
            self.parts.append(ContactElasticPart(name, reference, offset, permutation))
            stiffness.append(reference.stiffness[permutation][:, permutation])
            positions.append(reference.coordinates)
            faces.append(quadratic_boundary_triangles(reference) + offset)
            group_ids.append(np.full(len(reference.coordinates), len(self.parts) - 1))
            offset += len(reference.coordinates)
        self.stiffness = sparse.block_diag(stiffness, format="csc")
        self.rest = np.concatenate(positions)
        self.faces = np.concatenate(faces)
        edges = np.unique(np.sort(self.faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
        self.mesh = ipctk.CollisionMesh.build_from_full_mesh(self.rest, edges, self.faces)
        self.groups = np.concatenate(group_ids)
        collision_groups = self.groups[self.mesh.to_full_vertex_id()]
        self.mesh.can_collide = ipctk.make_vertex_patches_filter(collision_groups)
        self.dhat = dhat
        self.barrier = ipctk.BarrierPotential(dhat, barrier_stiffness, use_physical_barrier=True)
        self.friction = friction
        self.friction_potential = ipctk.FrictionPotential(friction_mollifier)
        self.accepted_steps = []

    def elastic_displacement(self, displacement):
        elastic = np.empty_like(displacement)
        for part in self.parts:
            count = part.reference.basis.N
            section = slice(part.first_node * 3, part.first_node * 3 + count)
            vector = np.empty(count)
            vector[part.permutation] = displacement[section]
            elastic[section] = part.reference.align_rigid_gauge(vector)[part.permutation]
        return elastic

    def save_failure(self, path, displacement, previous, lagged, boundary_dofs, records):
        if path is not None:
            np.savez_compressed(
                path,
                rest_positions=self.rest,
                displacement=displacement,
                previous_displacement=previous,
                lagged_displacement=lagged,
                boundary_dofs=boundary_dofs,
                surface_triangles=self.faces,
            )
            path.with_suffix(".json").write_text(
                json.dumps({"iterations": [record.__dict__ for record in records]}, indent=2) + "\n"
            )

    def normal_collisions(self, displacement):
        current = self.mesh.vertices(self.rest + displacement.reshape(-1, 3))
        collisions = ipctk.NormalCollisions()
        collisions.use_area_weighting = True
        collisions.collision_set_type = ipctk.NormalCollisions.IMPROVED_MAX_APPROX
        collisions.build(self.mesh, current, self.dhat)
        return collisions, current

    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        normal, current = self.normal_collisions(displacement)
        # Rigid translation energy cancels in the bulk quadratic form. Remove each part's rigid modes before evaluating it.
        elastic = self.elastic_displacement(displacement)
        gradient = self.stiffness @ elastic
        energy = 0.5 * elastic @ gradient + self.barrier(normal, self.mesh, current)
        contact_gradient = self.mesh.to_full_dof(self.barrier.gradient(normal, self.mesh, current))
        gradient = gradient + contact_gradient
        tangent_velocity = self.mesh.vertices((displacement - previous).reshape(-1, 3))
        if self.friction > 0:
            energy += self.friction_potential(tangential, self.mesh, tangent_velocity)
            shear = self.mesh.to_full_dof(self.friction_potential.gradient(tangential, self.mesh, tangent_velocity))
            gradient += shear
            contact_gradient += shear
        if external_force is not None:
            energy -= external_force @ displacement
            gradient -= external_force
        matrix = None
        if hessian:
            projection = ipctk.PSDProjectionMethod.CLAMP if project_hessian else ipctk.PSDProjectionMethod.NONE
            matrix = self.stiffness + self.mesh.to_full_dof(
                self.barrier.hessian(normal, self.mesh, current, projection)
            )
            if self.friction > 0:
                matrix += self.mesh.to_full_dof(
                    self.friction_potential.hessian(tangential, self.mesh, tangent_velocity)
                )
        return energy, gradient, matrix, normal, contact_gradient

    def reduced_gradient(self, gradient, free, coordinate_map):
        return gradient[free] if coordinate_map is None else coordinate_map.T @ gradient

    def reduced_hessian(self, matrix, free, coordinate_map):
        if coordinate_map is None:
            return matrix[free][:, free].tocsc()
        return (coordinate_map.T @ matrix @ coordinate_map).tocsc()

    def coordinate_energy_scale(self, displacement, gradient, contact_gradient, external_force):
        """Bound cancellation in coordinate-based contact work before force terms cancel."""
        applied = np.zeros_like(gradient) if external_force is None else external_force
        bulk_gradient = gradient + applied - contact_gradient
        return (abs(bulk_gradient) + abs(contact_gradient) + abs(applied)) @ abs(self.rest.reshape(-1) + displacement)

    def evaluate_updated_friction(self, displacement, previous, external_force, hessian):
        """Differentiate the current friction force through both geometry and relative motion."""
        normal, current = self.normal_collisions(displacement)
        tangential = ipctk.TangentialCollisions()
        if self.friction > 0:
            tangential.build(self.mesh, current, normal, self.barrier, self.friction)
        evaluated = self.evaluate(displacement, previous, tangential, hessian=hessian, external_force=external_force)
        if not hessian or self.friction == 0:
            return evaluated
        energy, gradient, matrix, normal, contact_gradient = evaluated
        rest = self.mesh.vertices(self.rest)
        current_displacement = self.mesh.vertices(displacement.reshape(-1, 3))
        velocity = self.mesh.vertices((displacement - previous).reshape(-1, 3))
        jacobians = [
            self.friction_potential.force_jacobian(
                tangential, self.mesh, rest, current_displacement, velocity, self.barrier, wrt
            )
            for wrt in (
                ipctk.TangentialPotential.DiffWRT.VELOCITIES,
                ipctk.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS,
            )
        ]
        frozen_shear = self.mesh.to_full_dof(self.friction_potential.hessian(tangential, self.mesh, velocity))
        matrix = matrix - frozen_shear - self.mesh.to_full_dof(jacobians[0] + jacobians[1])
        return energy, gradient, matrix, normal, contact_gradient

    def solve_updated_friction(
        self,
        boundary_dofs,
        boundary_values,
        load_steps,
        tolerance=1e-9,
        failure_path=None,
        initial_displacement=None,
        phase="updated_friction",
        coordinate_map=None,
        applied_force=None,
        initial_force=None,
    ):
        """Solve current-force equilibrium with CCD and a residual-based Newton line search.

        Current Coulomb friction is not a geometry-independent conservative
        potential. The merit function is the complete constrained force residual;
        the diagnostic potential is retained without treating it as its integral.
        """
        free = np.setdiff1d(np.arange(self.stiffness.shape[0]), boundary_dofs)
        displacement = (
            np.zeros(self.stiffness.shape[0]) if initial_displacement is None else initial_displacement.copy()
        )
        boundary_start = displacement[boundary_dofs].copy()
        records = []
        for step in range(1, load_steps + 1):
            previous = displacement.copy()
            displacement[boundary_dofs] = boundary_start + (boundary_values - boundary_start) * step / load_steps
            force_current = None
            if applied_force is not None:
                starting_force = np.zeros_like(applied_force) if initial_force is None else initial_force
                force_current = starting_force + (applied_force - starting_force) * step / load_steps
            for iteration in range(100):
                final = self.evaluate_updated_friction(displacement, previous, force_current, hessian=True)
                energy, gradient, matrix, normal, contact_gradient = final
                reduced_gradient = self.reduced_gradient(gradient, free, coordinate_map)
                residual = float(np.linalg.norm(reduced_gradient))
                print(
                    json.dumps({"step": step, "newton_iteration": iteration, "updated_residual_N": residual}),
                    flush=True,
                )
                if residual <= tolerance:
                    self.accepted_steps.append(ContactStepAcceptance(phase, step, 1, residual, len(normal)))
                    break
                reduced_matrix = self.reduced_hessian(matrix, free, coordinate_map)
                reduced_direction = splu(reduced_matrix).solve(-reduced_gradient)
                direction = np.zeros_like(displacement)
                if coordinate_map is None:
                    direction[free] = reduced_direction
                else:
                    direction = coordinate_map @ reduced_direction
                current = self.mesh.vertices(self.rest + displacement.reshape(-1, 3))
                proposed = self.mesh.vertices(self.rest + (displacement + direction).reshape(-1, 3))
                safe = ipctk.compute_collision_free_stepsize(self.mesh, current, proposed)
                alpha = min(1.0, safe)
                for _ in range(60):
                    trial = displacement + alpha * direction
                    evaluated = self.evaluate_updated_friction(trial, previous, force_current, hessian=False)
                    trial_residual = np.linalg.norm(self.reduced_gradient(evaluated[1], free, coordinate_map))
                    if trial_residual < (1 - 1e-4 * alpha) * residual:
                        break
                    alpha *= 0.5
                else:
                    self.save_failure(failure_path, displacement, previous, displacement, boundary_dofs, records)
                    raise RuntimeError(f"Updated friction residual line search did not converge at step{step}")
                records.append(ContactIteration(step, 0, iteration, energy, residual, len(normal), safe, alpha))
                displacement = trial
            else:
                self.save_failure(failure_path, displacement, previous, displacement, boundary_dofs, records)
                raise RuntimeError(f"Updated friction Newton solve did not converge at step{step}")
        energy, gradient, _, normal, contact_gradient = final
        return displacement, gradient, -contact_gradient, energy, records

    def solve_supported(
        self,
        boundary_dofs,
        boundary_values,
        load_steps,
        tolerance=1e-9,
        failure_path=None,
        initial_displacement=None,
        phase="load",
        coordinate_map=None,
        applied_force=None,
        initial_force=None,
    ):
        free = np.setdiff1d(np.arange(self.stiffness.shape[0]), boundary_dofs)
        displacement = (
            np.zeros(self.stiffness.shape[0]) if initial_displacement is None else initial_displacement.copy()
        )
        boundary_start = displacement[boundary_dofs].copy()
        records = []
        for step in range(1, load_steps + 1):
            previous = displacement.copy()
            displacement[boundary_dofs] = boundary_start + (boundary_values - boundary_start) * step / load_steps
            force_current = None
            if applied_force is not None:
                if initial_force is None:
                    force_current = applied_force * step / load_steps
                else:
                    force_current = initial_force + (applied_force - initial_force) * step / load_steps
            for friction_iteration in range(12 if self.friction > 0 else 1):
                lagged = displacement.copy()
                normal, current = self.normal_collisions(displacement)
                tangential = ipctk.TangentialCollisions()
                if self.friction > 0:
                    tangential.build(self.mesh, current, normal, self.barrier, self.friction)
                for iteration in range(100):
                    energy, gradient, matrix, normal, contact_gradient = self.evaluate(
                        displacement, previous, tangential, hessian=True, external_force=force_current
                    )
                    reduced_gradient = self.reduced_gradient(gradient, free, coordinate_map)
                    residual = np.linalg.norm(reduced_gradient)
                    if residual <= tolerance:
                        break
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "friction_iteration": friction_iteration,
                                "newton_iteration": iteration,
                                "free_residual_N": residual,
                                "normal_collisions": len(normal),
                            }
                        ),
                        flush=True,
                    )
                    reduced_matrix = self.reduced_hessian(matrix, free, coordinate_map)
                    reduced_direction = splu(reduced_matrix).solve(-reduced_gradient)
                    direction = np.zeros_like(displacement)
                    if coordinate_map is None:
                        direction[free] = reduced_direction
                    else:
                        direction = coordinate_map @ reduced_direction
                    slope = gradient @ direction
                    if slope >= 0:
                        scale = np.max(np.abs(matrix.diagonal()))
                        for relative_shift in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1):
                            shifted = reduced_matrix + relative_shift * scale * sparse.eye(
                                reduced_matrix.shape[0], format="csc"
                            )
                            reduced_direction = splu(shifted.tocsc()).solve(-reduced_gradient)
                            if coordinate_map is None:
                                direction[free] = reduced_direction
                            else:
                                direction = coordinate_map @ reduced_direction
                            slope = gradient @ direction
                            if slope < 0:
                                break
                        if slope >= 0:
                            self.save_failure(failure_path, displacement, previous, lagged, boundary_dofs, records)
                            raise RuntimeError(f"Contact Newton direction is not descending: {slope}")
                    start = self.mesh.vertices(self.rest + displacement.reshape(-1, 3))
                    end = self.mesh.vertices(self.rest + (displacement + direction).reshape(-1, 3))
                    safe = ipctk.compute_collision_free_stepsize(self.mesh, start, end)
                    alpha = min(1.0, safe)
                    for _ in range(60):
                        trial = displacement + alpha * direction
                        evaluated = self.evaluate(
                            trial, previous, tangential, hessian=False, external_force=force_current
                        )
                        candidate = evaluated[0]
                        coordinate_scale = max(
                            self.coordinate_energy_scale(displacement, gradient, contact_gradient, force_current),
                            self.coordinate_energy_scale(trial, evaluated[1], evaluated[4], force_current),
                        )
                        roundoff = 64 * np.finfo(float).eps * max(abs(energy), abs(candidate), coordinate_scale)
                        residual_reduced = (
                            np.linalg.norm(self.reduced_gradient(evaluated[1], free, coordinate_map))
                            < (1 - 1e-3 * alpha) * residual
                        )
                        if candidate <= energy + 1e-4 * alpha * slope or (
                            candidate <= energy + roundoff and residual_reduced
                        ):
                            break
                        alpha *= 0.5
                    else:
                        self.save_failure(failure_path, displacement, previous, lagged, boundary_dofs, records)
                        raise RuntimeError("Contact line search did not converge")
                    records.append(
                        ContactIteration(
                            step, friction_iteration, iteration, energy, residual, len(normal), safe, alpha
                        )
                    )
                    displacement = trial
                else:
                    self.save_failure(failure_path, displacement, previous, lagged, boundary_dofs, records)
                    raise RuntimeError(f"Contact Newton solve did not converge at load step{step}")
                updated_normal, updated_current = self.normal_collisions(displacement)
                updated = ipctk.TangentialCollisions()
                if self.friction > 0:
                    updated.build(self.mesh, updated_current, updated_normal, self.barrier, self.friction)
                final = self.evaluate(displacement, previous, updated, hessian=False, external_force=force_current)
                complete_residual = np.linalg.norm(self.reduced_gradient(final[1], free, coordinate_map))
                if complete_residual <= tolerance:
                    self.accepted_steps.append(
                        ContactStepAcceptance(
                            phase,
                            step,
                            friction_iteration + 1,
                            float(complete_residual),
                            len(updated_normal),
                        )
                    )
                    break
            else:
                self.save_failure(failure_path, displacement, previous, lagged, boundary_dofs, records)
                raise RuntimeError(f"Lagged friction did not converge at load step{step}")
        energy, gradient, _, normal, contact_gradient = final
        return displacement, gradient, -contact_gradient, energy, records


def block_reference(lower, upper, resolution, young, poisson):
    axes = [np.linspace(a, b, resolution) for a, b in zip(lower, upper)]
    mesh = MeshTet.init_tensor(*axes)
    return SkfemElasticReference(mesh.p.T, mesh.t.T, young, poisson, 1000)


def validate_blocks(resolution, dhat, friction, output, shear_after_close=0.0, contact_load_increment=None):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = perf_counter()
    young, poisson, gap, closing = 1e6, 0.0, 0.001, 0.00005
    lower = block_reference((-0.005, -0.005, -0.005), (0.005, 0.005, 0), resolution, young, poisson)
    upper = block_reference((-0.005, -0.005, gap), (0.005, 0.005, gap + 0.005), resolution, young, poisson)
    system = IndependentContactSystem((("lower", lower), ("upper", upper)), dhat, 1e6, friction, 1e-8)
    bottom = np.flatnonzero(np.isclose(system.rest[:, 2], -0.005))
    top = np.flatnonzero(np.isclose(system.rest[:, 2], gap + 0.005))
    nodes = np.r_[bottom, top]
    dofs = (nodes[:, None] * 3 + np.arange(3)).reshape(-1)
    values = np.zeros((len(nodes), 3))
    values[len(bottom) :, 2] = -(gap + closing)
    if contact_load_increment is None:
        contact_steps = 12
        displacement, gradient, contact_force, energy, records = system.solve_supported(
            dofs, values.reshape(-1), contact_steps, failure_path=output / "failure.npz", phase="compression"
        )
    else:
        if not 0 < contact_load_increment <= dhat:
            raise ValueError("Contact displacement increment must be positive and at most the activation distance")
        before_touch = values.copy()
        before_touch[len(bottom) :, 2] = -(gap - 4 * dhat)
        initial, _, _, _, approach_records = system.solve_supported(
            dofs, before_touch.reshape(-1), 12, failure_path=output / "approach_failure.npz", phase="approach"
        )
        contact_steps = int(np.ceil((closing + 4 * dhat) / contact_load_increment))
        displacement, gradient, contact_force, energy, records = system.solve_supported(
            dofs,
            values.reshape(-1),
            contact_steps,
            failure_path=output / "failure.npz",
            initial_displacement=initial,
            phase="compression",
        )
        (output / "approach_iterations.json").write_text(
            json.dumps([asdict(row) for row in approach_records], indent=2) + "\n"
        )
    if shear_after_close != 0:
        values[len(bottom) :, 0] = shear_after_close
        displacement, gradient, contact_force, energy, shear_records = system.solve_supported(
            dofs,
            values.reshape(-1),
            8,
            failure_path=output / "shear_failure.npz",
            initial_displacement=displacement,
            phase="shear",
        )
        records.extend(shear_records)
    fields, rows = [], []
    for part in system.parts:
        count = part.reference.basis.N
        section = slice(part.first_node * 3, part.first_node * 3 + count)
        vector = np.empty(count)
        vector[part.permutation] = displacement[section]
        tensor, vm = part.reference.corner_stress(vector)
        gradients = part.reference.corner_basis.interpolate(vector).grad.transpose(2, 3, 0, 1)
        force = contact_force[section].reshape(-1, 3)
        positions = part.reference.coordinates + displacement[section].reshape(-1, 3)
        rows.append(
            {
                "name": part.name,
                "contact_force_N": force.sum(axis=0).tolist(),
                "contact_moment_Nm": np.cross(positions - part.reference.com, force).sum(axis=0).tolist(),
                "peak_Pa": float(vm.max()),
                "tetrahedra": part.reference.mesh.nelements,
                "maximum_displacement_gradient": float(np.linalg.norm(gradients, axis=(-2, -1)).max()),
                "minimum_deformation_jacobian": float(np.linalg.det(gradients + np.eye(3)).min()),
                "maximum_elastic_displacement_m": float(
                    np.linalg.norm(
                        part.reference.align_rigid_gauge(vector)[part.reference.component_dofs], axis=1
                    ).max()
                ),
            }
        )
        fields.append((tensor, vm))
    free = np.setdiff1d(np.arange(len(gradient)), dofs)
    result = {
        "scope": "supported full3D deformable-block compression contact formulation control",
        "young_Pa": young,
        "poisson": poisson,
        "gap_m": gap,
        "closing_after_touch_m": closing,
        "shear_after_close_m": shear_after_close,
        "shear_steps": 8 if shear_after_close != 0 else 0,
        "resolution": resolution,
        "dhat_m": dhat,
        "barrier_stiffness": 1e6,
        "physical_area_weighted_barrier": True,
        "friction": friction,
        "friction_mollifier_m_per_unit_step": 1e-8,
        "contact_load_increment_m": contact_load_increment,
        "approach_steps": 0 if contact_load_increment is None else 12,
        "load_steps": contact_steps,
        "accepted_steps": [asdict(row) for row in system.accepted_steps],
        "complete_free_residual_N": float(np.linalg.norm(gradient[free])),
        "total_energy_J": float(energy),
        "parts": rows,
        "analytic_frictionless_zero_gap_force_N": young * 1e-4 * closing / 0.01,
        "elapsed_seconds": perf_counter() - started,
        "iterations": [record.__dict__ for record in records],
    }
    np.savez_compressed(
        output / "fields.npz",
        rest_positions=system.rest,
        surface_triangles=system.faces,
        displacement=displacement,
        contact_force_N=contact_force,
        reactions_N=gradient,
        lower_tensor_Pa=fields[0][0],
        lower_vm_Pa=fields[0][1],
        upper_tensor_Pa=fields[1][0],
        upper_vm_Pa=fields[1][1],
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=3)
    parser.add_argument("--dhat", type=float, default=1e-5)
    parser.add_argument("--friction", type=float, default=0)
    parser.add_argument("--contact-load-increment", type=float)
    parser.add_argument("--shear-after-close", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_blocks(
        args.resolution, args.dhat, args.friction, args.output, args.shear_after_close, args.contact_load_increment
    )


if __name__ == "__main__":
    main()
