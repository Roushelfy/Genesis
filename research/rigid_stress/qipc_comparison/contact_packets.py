"""Read-only normal-barrier queries and wrench-preserving compression of recorded spatial contact loads."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from qipc import tetmesh
from qipc._src.native.solver import GlobalVertexManager
from qipc.geometry import ground

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import corotation
from .pose_robot import PoseRobot, pose_contact_table
from .reference import ReferenceConfig, SolidObserver


def barrier_forces(scene, positions, dt):
    result = scene.contact_system.recompute(positions, positions)
    force = torch.zeros_like(positions)
    normal_bound = positions.new_zeros(len(positions))
    for channel, width in ((result.pt, 4), (result.ee, 4), (result.pe, 3), (result.pp, 2), (result.ph, 1)):
        active = channel.active != 0
        indices = channel.stencil[active, :width].reshape(-1).to(torch.int64)
        values = -channel.grad[active].reshape(-1, 3) / dt**2
        assert torch.isfinite(values).all().item()
        force.index_add_(0, indices, values)
        normal_bound.index_add_(0, indices, torch.linalg.norm(values, dim=1))
    return force, normal_bound


def compress(points, forces, adjacency, length):
    active = np.flatnonzero(np.linalg.norm(forces, axis=1) > 1e-12)
    if not len(active):
        return active, np.empty(0), np.empty(0, dtype=np.int64), 0.0
    count, labels = connected_components(adjacency[active][:, active], directed=False)
    selected, coefficients, groups = [], [], []
    error = 0.0
    for group in range(count):
        nodes = active[labels == group]
        magnitude = np.linalg.norm(forces[nodes], axis=1)
        directions = forces[nodes] / magnitude[:, None]
        center = points[nodes].mean(axis=0)
        matrix = np.vstack((directions.T, np.cross(points[nodes] - center, directions).T / length))
        target = matrix @ magnitude
        result = linprog(
            np.ones(len(nodes)),
            A_eq=matrix,
            b_eq=target,
            bounds=(0, None),
            method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
        if not result.success:
            raise RuntimeError(f"Nonnegative contact-wrench compression failed: {result.message}")
        chosen = result.x > 1e-12
        defect = matrix[:, chosen] @ result.x[chosen] - target
        error = max(error, np.linalg.norm(defect).item())
        assert np.linalg.norm(defect) <= 1e-8 * max(np.linalg.norm(target), 1e-8) + 1e-10
        selected.append(nodes[chosen])
        coefficients.append(result.x[chosen] / magnitude[chosen])
        groups.append(np.full(chosen.sum(), group))
    return np.concatenate(selected), np.concatenate(coefficients), np.concatenate(groups), error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    receipt = json.loads((args.reference / "result.json").read_text())
    inputs = json.loads((args.reference / "inputs.json").read_text())
    config = ReferenceConfig(**receipt["config"])
    with np.load(args.reference / "rest.npz") as saved:
        authored, world, tets, masses = (
            saved["authored_vertices_m"],
            saved["vertices_world_m"],
            saved["tetrahedra"],
            saved["masses_kg"],
        )
    with np.load(inputs["conditions"]["path"]) as saved:
        base_radius, delay = saved["radius"][0].item(), saved["delays"][0].item()
    pairs = np.concatenate([tets[:, [left, right]] for left in range(4) for right in range(left + 1, 4)])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    adjacency = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(authored), len(authored))
    ).tocsr()
    adjacency = adjacency + adjacency.T
    length = np.linalg.norm(np.ptp(authored, axis=0)).item()
    scene = config.scene()
    descriptor_scene = config.scene(contact=False)
    descriptor = descriptor_scene.add_mjcf(args.reference / "panda-bank-initial.xml", link_type="rigid")
    robot = PoseRobot.create(scene, descriptor, inputs["dbc_stiffness_ratio"])
    geometry = tetmesh(world, tets)
    config.material().apply_to(
        geometry, youngs_modulus=config.young, poissons_ratio=config.poisson, mass_density=config.density
    )
    scene.geometries.create("egg", geometry)
    floor = ground(height=0, N=(0, 0, 1))
    scene.geometries.create("floor", floor)
    pose_contact_table(scene, robot, floor, config.resistance)
    scene.init()
    with torch.cuda.stream(scene.solver.cuda_stream):
        observer = SolidObserver.allocate(scene, config)
    torch.cuda.synchronize()
    manager = scene.sim_systems[GlobalVertexManager]
    base_positions = manager.positions_view().cpu().numpy()
    owners = scene.rigid_body.body_id.cpu().numpy()
    assert scene.affine_body.n_bodies == 0
    np.testing.assert_array_equal(manager.body_id_view()[: len(owners)].cpu().numpy(), owners)
    local = scene.rigid_body.x_bar.cpu().numpy() - scene.rigid_body.com_local.cpu().numpy()[owners]
    fem_global = observer.fem_global.cpu().numpy()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for path in sorted((args.reference / "frames").glob("step*.npz")):
            with np.load(path) as saved:
                position, force = saved["positions_m"], saved["contact_forces_n"]
                velocity, inertial = saved["velocities_m_per_s"], saved["inertial_reaction_n"]
                rotation = corotation(authored, position, masses)
                material_position = (
                    position - masses @ position / masses.sum()
                ) @ rotation + masses @ authored / masses.sum()
                material_force, material_inertial = force @ rotation, inertial @ rotation
                quaternion = saved["actual_robot_material_quaternion_wxyz"]
                rigid_rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
                query = base_positions.copy()
                query[: len(owners)] = (
                    np.einsum("vij,vj->vi", rigid_rotation[owners], local) + saved["actual_robot_com_world_m"][owners]
                )
                query[fem_global] = position
                time, step = saved["time_s"].item(), saved["step"].item()
            query = torch.as_tensor(query, device="cuda")
            normal_force, normal_bound = barrier_forces(scene, query, config.dt)
            normal_force = normal_force[observer.fem_global].cpu().numpy() @ rotation
            normal_bound = normal_bound[observer.fem_global].cpu().numpy()
            magnitude = np.linalg.norm(normal_force, axis=1)
            normal = np.divide(
                normal_force, magnitude[:, None], out=np.zeros_like(normal_force), where=magnitude[:, None] > 1e-12
            )
            selected, coefficient, component, defect = compress(material_position, material_force, adjacency, length)
            current_relative = position - masses @ position / masses.sum()
            inertia = np.eye(3) * np.einsum("v,vi,vi->", masses, current_relative, current_relative) - np.einsum(
                "v,vi,vj->ij", masses, current_relative, current_relative
            )
            omega = (
                np.linalg.solve(inertia, np.einsum("v,vi->i", masses, np.cross(current_relative, velocity))) @ rotation
            )
            tick = step // round(0.01 / config.dt)
            phase = ((tick - delay) % 600) / 599 if tick >= delay else 0.0
            np.savez_compressed(
                args.output / "frames" / path.name,
                time_s=time,
                authored_vertices_m=authored,
                physical_contact_positions_material_m=material_position,
                contact_forces_material_n=material_force,
                inertial_reaction_material_n=material_inertial,
                normal_barrier_forces_material_n=normal_force,
                normal_force_sum_of_magnitudes_n=normal_bound,
                effective_nodal_normal=normal,
                selected_nodes=selected,
                positive_coefficients=coefficient,
                support_components=component,
                compressed_positions_material_m=material_position[selected],
                compressed_forces_material_n=material_force[selected] * coefficient[:, None],
                compressed_normals=normal[selected],
                omega_material_rad_per_s=omega,
                friction=config.friction,
                radius_m=base_radius * (1 + 0.15 * np.sin(2 * np.pi * phase)),
            )
            journal.write(
                json.dumps(
                    {
                        "time_s": time,
                        "source_frame": file_id(path),
                        "active_force_nodes": int((np.linalg.norm(force, axis=1) > 1e-12).sum()),
                        "compressed_nodes": len(selected),
                        "scaled_wrench_compression_defect_n": defect,
                        "selected_effective_normals_valid": bool((magnitude[selected] > 1e-12).all()),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    write_json(
        args.output / "result.json",
        {
            "kind": "resolved-and-compressed-historical-contact-packets",
            "identity": runtime_identity(),
            "reference": file_id(args.reference / "result.json"),
            "reference_qualified": False,
            "scope": "Read-only original normal-barrier recomputation at recorded actual robot/FEM positions; no time step or replacement friction history. Total contact and inertia remain the actual recorded quantities. Effective nodal normals aggregate original barrier gradients; they are not claimed to be individual rigid collider normals. Nonnegative linear programming compresses each connected active mesh-support component while preserving its physical force/moment, without a radius or footprint fit. These are derived packets for an interface ablation, not observations from the native rigid collider.",
        },
    )


if __name__ == "__main__":
    main()
