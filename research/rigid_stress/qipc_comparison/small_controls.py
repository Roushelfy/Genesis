"""Compare actual QIPC P1 dynamics with independently assembled CPU linear elasticity."""

import argparse
from dataclasses import asdict
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

import qipc
import qipc._src.native.solver.qipc_solver_ext as extension
from qipc import tetmesh

from .artifacts import file_id, write_json
from .reference import ReferenceConfig, SolidObserver, solver_statistics


def beam_mesh():
    shape = (5, 3, 3)
    vertices = np.stack(
        np.meshgrid(
            np.linspace(0, 0.004, shape[0]),
            np.linspace(0, 0.002, shape[1]),
            np.linspace(0, 0.0005, shape[2]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    cells = np.indices(tuple(value - 1 for value in shape)).reshape(3, -1).T
    tetrahedra = []
    for axes in permutations(range(3)):
        path = np.concatenate(
            (np.zeros((1, 3), dtype=np.int64), np.cumsum(np.eye(3, dtype=np.int64)[list(axes)], axis=0))
        )
        tetrahedra.append(np.ravel_multi_index((cells[:, None] + path).transpose(2, 0, 1), shape))
    tetrahedra = np.concatenate(tetrahedra)
    signed = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
    negative = signed < 0
    tetrahedra[negative, :2] = tetrahedra[negative, 1::-1]
    return vertices, tetrahedra


def independent_operator(vertices, tetrahedra, young, poisson, density):
    inverse = np.linalg.inv((vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]]).transpose(0, 2, 1))
    gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
    volume = 1 / np.linalg.det(inverse) / 6
    mu = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    local = (
        lame * np.einsum("tai,tbj->taibj", gradients, gradients)
        + mu * np.einsum("tak,tbk,ij->taibj", gradients, gradients, np.eye(3))
        + mu * np.einsum("taj,tbi->taibj", gradients, gradients)
    ) * volume[:, None, None, None, None]
    dofs = (tetrahedra[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
    rows = np.broadcast_to(dofs[:, :, None], local.reshape(-1, 12, 12).shape)
    columns = np.broadcast_to(dofs[:, None, :], rows.shape)
    matrix = sparse.coo_matrix(
        (local.reshape(-1), (rows.reshape(-1), columns.reshape(-1))), shape=(len(vertices) * 3, len(vertices) * 3)
    ).tocsr()
    masses = np.bincount(tetrahedra.reshape(-1), weights=np.repeat(volume * density / 4, 4), minlength=len(vertices))
    return matrix, masses, gradients, volume


def linear_stress(displacement, tetrahedra, gradients, young, poisson):
    gradient = np.einsum("tai,taj->tij", displacement[tetrahedra], gradients)
    strain = (gradient + gradient.transpose(0, 2, 1)) / 2
    mu = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    return 2 * mu * strain + lame * np.trace(strain, axis1=-2, axis2=-1)[:, None, None] * np.eye(3)


def run_case(config, amplitude, output):
    output.mkdir()
    vertices, tetrahedra = beam_mesh()
    matrix, masses, gradients, volumes = independent_operator(
        vertices, tetrahedra, config.young, config.poisson, config.density
    )
    fixed = vertices[:, 0] == 0
    free = np.repeat(~fixed, 3)
    # Trapezoidal weights integrate a uniform end traction exactly on the
    # structured planar end face. A shear component exercises bending too.
    end = vertices[:, 0] == vertices[:, 0].max()
    weights = np.ones(len(vertices))
    for axis in (1, 2):
        edge = (vertices[:, axis] == vertices[:, axis].min()) | (vertices[:, axis] == vertices[:, axis].max())
        weights[edge] *= 0.5
    weights *= end
    force = weights[:, None] / weights.sum() * np.array([0.01, 0.0005, 0.0002]) * amplitude
    scene = config.scene(gravity=(0, 0, 0), contact=False)
    geometry = tetmesh(vertices, tetrahedra)
    config.material().apply_to(
        geometry,
        youngs_modulus=config.young,
        poissons_ratio=config.poisson,
        mass_density=config.density,
        is_fixed=fixed,
    )
    scene.geometries.create("beam", geometry)
    scene.init()
    with torch.cuda.stream(scene.solver.cuda_stream):
        observer = SolidObserver.allocate(scene, config)
        scene.finite_element.external_acc[:] = torch.as_tensor(force / masses[:, None], device=observer.rest.device)
    torch.cuda.synchronize()
    np.testing.assert_allclose(observer.masses.cpu().numpy(), masses, rtol=2e-13, atol=1e-22)
    system = matrix + sparse.diags(np.repeat(masses / config.dt**2, 3))
    direct = splu(system[free][:, free].tocsc())
    displacement = np.zeros_like(vertices)
    velocity = np.zeros_like(vertices)
    history, native_displacements, cpu_displacements, native_fields, cpu_fields = [], [], [], [], []
    for step in range(6):
        predictor = displacement + config.dt * velocity
        rhs = force + masses[:, None] * predictor / config.dt**2
        next_displacement = np.zeros_like(displacement)
        next_displacement.reshape(-1)[free] = direct.solve(rhs.reshape(-1)[free])
        velocity = (next_displacement - displacement) / config.dt
        displacement = next_displacement
        cpu_stress = linear_stress(displacement, tetrahedra, gradients, config.young, config.poisson)
        with torch.cuda.stream(scene.solver.cuda_stream):
            previous_velocity = scene.finite_element.velocities.clone()
        scene.step()
        with torch.cuda.stream(scene.solver.cuda_stream):
            field = observer.update()
            elastic_gradient = observer.elastic_gradient(field)
            residual = (
                observer.masses[:, None]
                * (
                    (scene.finite_element.velocities - previous_velocity) / config.dt
                    - scene.finite_element.external_acc
                )
                + elastic_gradient
            )
        torch.cuda.synchronize()
        actual = scene.finite_element.x.cpu().numpy() - vertices
        actual_stress = field.cauchy_pa.cpu().numpy()
        actual_linear_stress = linear_stress(actual, tetrahedra, gradients, config.young, config.poisson)
        weighted_error = np.sum(volumes[:, None, None] * (actual_stress - cpu_stress) ** 2)
        weighted_reference = np.sum(volumes[:, None, None] * cpu_stress**2)
        balance = residual.cpu().numpy()
        row = {
            "step": step,
            **solver_statistics(scene),
            "minimum_det_F": field.determinant.min().item(),
            "maximum_displacement_m": float(np.abs(actual).max()),
            "displacement_relative_l2": float(np.linalg.norm(actual - displacement) / np.linalg.norm(displacement)),
            "volume_weighted_tensor_relative_l2": float(np.sqrt(weighted_error / weighted_reference)),
            "actual_state_linear_constitutive_difference": float(
                np.linalg.norm(actual_stress - actual_linear_stress) / np.linalg.norm(actual_linear_stress)
            ),
            "free_force_residual_relative": float(np.linalg.norm(balance[~fixed]) / np.linalg.norm(force)),
            "clamp_reaction_n": (-elastic_gradient[torch.as_tensor(fixed, device=observer.rest.device)])
            .sum(dim=0)
            .tolist(),
            "raw_vm_peak_pa": field.von_mises_pa.max().item(),
        }
        history.append(row)
        native_displacements.append(actual)
        cpu_displacements.append(displacement.copy())
        native_fields.append(actual_stress)
        cpu_fields.append(cpu_stress)
        write_json(output / "progress.json", {"config": asdict(config), "amplitude": amplitude, "history": history})
    np.savez_compressed(
        output / "fields.npz",
        vertices=vertices,
        tetrahedra=tetrahedra,
        force_n=force,
        native_displacements=native_displacements,
        cpu_displacements=cpu_displacements,
        native_cauchy_pa=native_fields,
        cpu_linear_stress_pa=cpu_fields,
    )
    return {
        "config": asdict(config),
        "amplitude": amplitude,
        "history": history,
        "physical_scope": "P1 lumped-mass implicit Euler vs independent CPU isotropic P1 stiffness; finite-strain constitutive differences retained",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"qipc_import": qipc.__file__, "native": file_id(extension.__file__)})
    cases = []
    for energy in ("snh", "hencky"):
        for amplitude in (1.0, 0.1):
            config = ReferenceConfig(
                dt=1e-4,
                mas_precision="double",
                energy=energy,
                velocity_tolerance=args.velocity_tolerance,
                linear_tolerance=1e-9,
            )
            cases.append(run_case(config, amplitude, args.output / f"{energy}-{amplitude}"))
    write_json(args.output / "result.json", {"kind": "independent-small-strain-dynamic-controls", "cases": cases})


if __name__ == "__main__":
    main()
