"""Advance the full hollow FEM egg over a plane and audit original contact forces and elastic stress."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import qipc
import qipc._src.native.solver.qipc_solver_ext as extension
from qipc import tetmesh
from qipc.geometry import ground

from .artifacts import file_id, write_json
from .linear_diagnostics import linear_diagnostics
from .reference import ReferenceConfig, SolidObserver, solver_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--velocity-tolerance", type=float, default=1e-5)
    parser.add_argument("--linear-tolerance", type=float, default=1e-7)
    parser.add_argument("--mas-precision", choices=("float", "double"), default="float")
    parser.add_argument("--linear-solver", choices=("linear_pcg", "partition_pcg"), default="partition_pcg")
    parser.add_argument("--linear-diagnostics", action="store_true")
    parser.add_argument("--energy", choices=("snh", "hencky"), default="snh")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = ReferenceConfig(
        dt=args.dt,
        velocity_tolerance=args.velocity_tolerance,
        linear_tolerance=args.linear_tolerance,
        energy=args.energy,
        mas_precision=args.mas_precision,
        linear_solver=args.linear_solver,
    )
    write_json(
        args.output / "start.json",
        {
            "qipc_import": qipc.__file__,
            "native": file_id(extension.__file__),
            "input": file_id(args.mesh),
            "config": asdict(config),
            "scope": "intrusive small workflow/force diagnostic",
        },
    )
    with np.load(args.mesh) as mesh:
        vertices, tets = mesh["vertices"].copy(), mesh["tetrahedra"]
    vertices[:, 2] += 0.031
    scene = config.scene(diagnostics=args.linear_diagnostics)
    geo = tetmesh(vertices, tets)
    config.material().apply_to(
        geo, youngs_modulus=config.young, poissons_ratio=config.poisson, mass_density=config.density
    )
    scene.geometries.create("egg", geo)
    scene.geometries.create("ground", ground(height=0.0, N=(0.0, 0.0, 1.0)))
    started = perf_counter()
    scene.init()
    torch.cuda.synchronize()
    setup = perf_counter() - started
    with torch.cuda.stream(scene.solver.cuda_stream):
        observation = SolidObserver.allocate(scene, config)
    torch.cuda.synchronize()
    history, positions, fields, contact_forces = [], [], [], []
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for step in range(args.steps):
            with torch.cuda.stream(scene.solver.cuda_stream):
                previous_velocity = scene.finite_element.velocities.clone()
            scene.step()
            with torch.cuda.stream(scene.solver.cuda_stream):
                field = observation.update()
            torch.cuda.synchronize()
            contact = observation.contact()
            fe = scene.finite_element
            gradient = observation.elastic_gradient(field)
            inertial = observation.masses[:, None] * ((fe.velocities - previous_velocity) / config.dt - fe.gravity)
            residual = inertial + gradient - contact.fem_forces_n
            reference_norm = torch.linalg.norm(contact.fem_forces_n).item()
            kinetic = 0.5 * (observation.masses[:, None] * fe.velocities**2).sum()
            gravitational = -(observation.masses[:, None] * fe.gravity * fe.x).sum()
            elastic = (observation.volumes * field.energy_density_relative_rest_j_per_m3).sum()
            row = {
                "step": step,
                "time_s": (step + 1) * config.dt,
                **solver_statistics(scene),
                "minimum_det_F": field.determinant.min().item(),
                "raw_vm_peak_pa": field.von_mises_pa.max().item(),
                "raw_maximum_principal_pa": field.maximum_principal_pa.max().item(),
                "minimum_surface_height_m": fe.x[:, 2].min().item(),
                "contact_resultant_n": contact.fem_forces_n.sum(dim=0).tolist(),
                "actual_contact_vertex_count": int((torch.linalg.norm(contact.fem_forces_n, dim=1) > 0).sum()),
                "force_residual_l2_n": torch.linalg.norm(residual).item(),
                "force_residual_resultant_n": residual.sum(dim=0).tolist(),
                "force_residual_relative_contact": torch.linalg.norm(residual).item() / reference_norm
                if reference_norm > 1e-12
                else None,
                "kinetic_j": kinetic.item(),
                "gravitational_j": gravitational.item(),
                "elastic_j": elastic.item(),
            }
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            history.append(row)
            positions.append(fe.x.cpu().numpy())
            fields.append(field.cauchy_pa.cpu().numpy())
            contact_forces.append(contact.fem_forces_n.cpu().numpy())
            if row["minimum_det_F"] <= 0 or not torch.isfinite(field.cauchy_pa).all():
                raise RuntimeError(
                    "Reference elastic state is not physically admissible; retained raw state is unqualified"
                )
    np.savez_compressed(
        args.output / "trajectory.npz",
        positions_m=positions,
        cauchy_pa=fields,
        contact_forces_n=contact_forces,
        rest_positions_m=observation.rest.cpu().numpy(),
        tetrahedra=tets,
    )
    write_json(
        args.output / "result.json",
        {
            "kind": "original-QIPC-hollow-egg-workflow",
            "setup_seconds": setup,
            "config": asdict(config),
            "history": history,
            "reference_qualified": False,
            "force_timing": "Original last-assembled contact gradient including friction, divided by -dt^2; force residual compares it with final accepted elastic/inertial state and therefore exposes stopping/cadence error.",
            "linear_diagnostics": linear_diagnostics(scene, args.output) if args.linear_diagnostics else None,
        },
    )


if __name__ == "__main__":
    main()
