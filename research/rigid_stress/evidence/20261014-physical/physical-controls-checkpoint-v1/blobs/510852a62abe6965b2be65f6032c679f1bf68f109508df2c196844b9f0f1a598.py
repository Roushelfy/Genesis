"""Independent quasistatic quadratic FEM coupled to IPC Toolkit contact and friction."""

import argparse
import json
from dataclasses import dataclass
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

    def evaluate(self, displacement, previous, tangential, hessian):
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
        matrix = None
        if hessian:
            matrix = self.stiffness + self.mesh.to_full_dof(
                self.barrier.hessian(normal, self.mesh, current, ipctk.PSDProjectionMethod.NONE)
            )
            if self.friction > 0:
                matrix += self.mesh.to_full_dof(
                    self.friction_potential.hessian(tangential, self.mesh, tangent_velocity)
                )
        return energy, gradient, matrix, normal, contact_gradient

    def solve_supported(self, boundary_dofs, boundary_values, load_steps, tolerance=1e-9, failure_path=None):
        free = np.setdiff1d(np.arange(self.stiffness.shape[0]), boundary_dofs)
        displacement = np.zeros(self.stiffness.shape[0])
        records = []
        for step in range(1, load_steps + 1):
            previous = displacement.copy()
            displacement[boundary_dofs] = boundary_values * step / load_steps
            for friction_iteration in range(12 if self.friction > 0 else 1):
                lagged = displacement.copy()
                normal, current = self.normal_collisions(displacement)
                tangential = ipctk.TangentialCollisions()
                if self.friction > 0:
                    tangential.build(self.mesh, current, normal, self.barrier, self.friction)
                for iteration in range(512):
                    energy, gradient, matrix, normal, _ = self.evaluate(
                        displacement, previous, tangential, hessian=True
                    )
                    residual = np.linalg.norm(gradient[free])
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
                    direction = np.zeros_like(displacement)
                    direction[free] = splu(matrix[free][:, free].tocsc()).solve(-gradient[free])
                    slope = gradient @ direction
                    if slope >= 0:
                        scale = np.max(np.abs(matrix.diagonal()))
                        for relative_shift in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1):
                            shifted = matrix[free][:, free] + relative_shift * scale * sparse.eye(
                                len(free), format="csc"
                            )
                            direction[free] = splu(shifted.tocsc()).solve(-gradient[free])
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
                        evaluated = self.evaluate(trial, previous, tangential, hessian=False)
                        candidate = evaluated[0]
                        roundoff = 64 * np.finfo(float).eps * max(abs(energy), abs(candidate))
                        residual_reduced = np.linalg.norm(evaluated[1][free]) < (1 - 1e-3 * alpha) * residual
                        # An Armijo decrease smaller than energy roundoff cannot distinguish a two-cycle.
                        # At that scale require equilibrium improvement as well as bounded energy change.
                        resolved_decrease = -1e-4 * alpha * slope > roundoff
                        accepted = (
                            candidate <= energy + 1e-4 * alpha * slope
                            if resolved_decrease
                            else candidate <= energy + roundoff and residual_reduced
                        )
                        if accepted:
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
                final = self.evaluate(displacement, previous, updated, hessian=False)
                if np.linalg.norm(final[1][free]) <= tolerance:
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


def validate_blocks(resolution, dhat, friction, output):
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
    displacement, gradient, contact_force, energy, records = system.solve_supported(
        dofs, values.reshape(-1), 12, failure_path=output / "failure.npz"
    )
    fields, rows = [], []
    for part in system.parts:
        count = part.reference.basis.N
        section = slice(part.first_node * 3, part.first_node * 3 + count)
        vector = np.empty(count)
        vector[part.permutation] = displacement[section]
        tensor, vm = part.reference.corner_stress(vector)
        force = contact_force[section].reshape(-1, 3)
        positions = part.reference.coordinates + displacement[section].reshape(-1, 3)
        rows.append(
            {
                "name": part.name,
                "contact_force_N": force.sum(axis=0).tolist(),
                "contact_moment_Nm": np.cross(positions - part.reference.com, force).sum(axis=0).tolist(),
                "peak_Pa": float(vm.max()),
                "tetrahedra": part.reference.mesh.nelements,
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
        "resolution": resolution,
        "dhat_m": dhat,
        "barrier_stiffness": 1e6,
        "physical_area_weighted_barrier": True,
        "friction": friction,
        "friction_mollifier_m_per_unit_step": 1e-8,
        "load_steps": 12,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_blocks(args.resolution, args.dhat, args.friction, args.output)


if __name__ == "__main__":
    main()
