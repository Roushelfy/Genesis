"""Replay actual spatial contact minus measured inertia into independent P1 and P2 linear elasticity."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from ..skfem_reference import SkfemElasticReference
from .artifacts import file_id, runtime_identity, write_json
from .common_metric import CommonColumns, TetSampler, corotation, field_error, tangent
from .p1_elastic import FreeElasticP1


def wrench(positions, forces):
    return np.r_[forces.sum(axis=0), np.cross(positions, forces).sum(axis=0)]


def transfer_loads(reference, points, output):
    sampler = TetSampler.allocate(reference.mesh.p.T, reference.mesh.t.T)
    relative = points[:, None] - sampler.vertices[sampler.tetrahedra[:, 0]][None]
    free = np.einsum("tij,ptj->pti", sampler.inverse, relative)
    barycentric = np.concatenate((1 - free.sum(axis=2, keepdims=True), free), axis=2)
    cells = barycentric.min(axis=2).argmax(axis=1)
    barycentric = barycentric[np.arange(len(points)), cells]
    projected = np.maximum(barycentric, 0)
    projected /= projected.sum(axis=1, keepdims=True)
    selected_vertices = sampler.vertices[sampler.tetrahedra[cells]]
    distances = np.linalg.norm(np.einsum("pc,pcj->pj", projected, selected_vertices) - points, axis=1)
    write_json(
        output / "transfer-diagnostics.json",
        {
            "maximum_selected_tet_projection_distance_m": distances.max().item(),
            "minimum_selected_tet_barycentric": barycentric.min().item(),
            "points_with_projection_distance_over_0_1nm": int((distances > 1e-10).sum()),
            "scope": "Complete material-point locator over the native P1 partition; no coordinate clipping is used in the virtual-work transfer. Distance is an upper bound to the selected tet, not a nearest-surface claim.",
        },
    )
    assert distances.max() < 1e-10, (
        "Reference material nodes do not lie in the native continuum; retain mismatch diagnostics"
    )
    weights = np.stack(
        [
            reference.scalar_basis.elem.lbasis(barycentric[:, 1:].T, index)[0]
            for index in range(reference.scalar_basis.Nbfun)
        ]
    )
    scalar_nodes = reference.scalar_basis.element_dofs[:, cells]
    rows, columns, values = [], [], []
    for component in range(3):
        rows.append(reference.component_dofs[scalar_nodes, component].reshape(-1))
        columns.append(np.broadcast_to(np.arange(len(points))[None] * 3 + component, weights.shape).reshape(-1))
        values.append(weights.reshape(-1))
    return sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(reference.basis.N, len(points) * 3),
    ).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native-mesh", type=Path, required=True)
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    with np.load(args.reference / "rest.npz") as saved:
        authored, tets, mass = saved["authored_vertices_m"], saved["tetrahedra"], saved["masses_kg"]
    with np.load(args.native_mesh) as saved:
        native_vertices, native_tets = saved["vertices"], saved["tetrahedra"]
    with np.load(args.columns) as saved:
        columns = CommonColumns(**{name: saved[name] for name in saved.files})
    p1 = FreeElasticP1(authored, tets, 1e10, 0.3, 2000)
    p2 = SkfemElasticReference(native_vertices, native_tets, 1e10, 0.3, 2000)
    np.testing.assert_allclose(p1.masses, mass, rtol=2e-12, atol=1e-20)
    affine = authored @ np.array([[0.003, 0.001, 0], [0.001, -0.002, 0.0004], [0, 0.0004, 0.001]])
    analytic_load = (p1.stiffness @ affine.reshape(-1)).reshape(-1, 3)
    recovered, _, correction, residual = p1.solve(analytic_load)
    from_gradient = p1.gradients
    gradient = np.einsum("tai,taj->tij", affine[tets], from_gradient)
    strain = (gradient + gradient.transpose(0, 2, 1)) / 2
    mu, lame = 1e10 / 2.6, 1e10 * 0.3 / (1.3 * 0.4)
    analytic = 2 * mu * strain + lame * np.trace(strain, axis1=-2, axis2=-1)[:, None, None] * np.eye(3)
    np.testing.assert_allclose(recovered, analytic, rtol=1e-7, atol=0.1)
    assert np.linalg.norm(correction) / np.linalg.norm(analytic_load) < 1e-10
    assert np.linalg.norm(residual) / np.linalg.norm(analytic_load) < 1e-8
    transfer = transfer_loads(p2, authored, args.output)
    np.savez_compressed(
        args.output / "operator.npz",
        p2_coordinates_m=p2.coordinates,
        p2_component_dofs=p2.component_dofs,
        transfer_data=transfer.data,
        transfer_indices=transfer.indices,
        transfer_indptr=transfer.indptr,
        transfer_shape=transfer.shape,
    )
    receipt = json.loads((args.reference / "result.json").read_text())
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for index, path in enumerate(sorted((args.reference / "frames").glob("step*.npz"))):
            with np.load(path) as saved:
                position = saved["positions_m"]
                rotation = corotation(authored, position, mass)
                contact = saved["contact_forces_n"] @ rotation
                inertial = saved["inertial_reaction_n"] @ rotation
                actual_tensor = rotation.T @ saved["cauchy_pa"] @ rotation
                time = saved["time_s"].item()
            load = contact - inertial
            physical_material_position = (
                position - mass @ position / mass.sum()
            ) @ rotation + mass @ authored / mass.sum()
            tensor, displacement, correction, residual = p1.solve(load)
            mapped = transfer @ load.reshape(-1)
            p2_solution = p2.solve(mapped)
            mapped_force = mapped[p2.component_dofs]
            np.testing.assert_allclose(
                wrench(p2.coordinates, mapped_force), wrench(authored, load), rtol=1e-8, atol=1e-10
            )
            sampled_actual = tangent(actual_tensor[columns.reference_elements], columns)
            sampled_p1 = tangent(tensor[columns.reference_elements], columns)
            sampled_p2 = tangent(
                np.einsum(
                    "qc,qcij->qij",
                    columns.native_barycentric,
                    p2_solution.tensor_pa[columns.native_elements],
                ),
                columns,
            )
            row = {
                "time_s": time,
                "frame": file_id(path),
                "reference_qualified": False,
                "contact_wrench_material_origin": wrench(authored, contact).tolist(),
                "inertial_wrench_material_origin": wrench(authored, inertial).tolist(),
                "physical_contact_wrench_material_origin": wrench(physical_material_position, contact).tolist(),
                "physical_inertial_wrench_material_origin": wrench(physical_material_position, inertial).tolist(),
                "balance_correction_l2_n": np.linalg.norm(correction).item(),
                "balance_correction_relative_load": np.linalg.norm(correction).item()
                / max(np.linalg.norm(load).item(), 1e-12),
                "independent_p1_force_residual_l2_n": np.linalg.norm(residual).item(),
                "independent_p2_force_residual_l2_n": p2_solution.residual_n,
                "physical_versus_undeformed_contact_moment_difference_nm": np.linalg.norm(
                    np.cross((position - (mass @ position / mass.sum())) @ rotation, contact).sum(axis=0)
                    - np.cross(authored - (mass @ authored / mass.sum()), contact).sum(axis=0)
                ).item(),
                "p1_common_tangent_error": field_error(sampled_p1, sampled_actual, columns),
                "p2_common_tangent_error": field_error(sampled_p2, sampled_actual, columns),
            }
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            np.savez_compressed(
                args.output / "frames" / path.name,
                time_s=time,
                physical_contact_positions_world_m=position,
                physical_contact_positions_material_m=physical_material_position,
                material_rotation=rotation,
                authored_contact_n=contact,
                authored_inertial_reaction_n=inertial,
                p1_displacement_m=displacement,
                p1_linear_stress_pa=tensor,
                p2_linear_corner_stress_pa=p2_solution.tensor_pa,
                p2_force_n=mapped_force,
                p2_balanced_force_n=p2_solution.rhs[p2.component_dofs],
                reference_corotated_cauchy_pa=actual_tensor,
            )
    write_json(
        args.output / "result.json",
        {
            "kind": "independent-spatial-reference-load-replay",
            "identity": runtime_identity(),
            "reference": file_id(args.reference / "result.json"),
            "source_reference_completed": receipt["completed_requested_history"],
            "native_mesh": file_id(args.native_mesh),
            "columns": file_id(args.columns),
            "reference_qualified": False,
            "scope": "Actual discrete contact minus measured acceleration/gravity loads, in the fitted material frame. Spatial nodal forces are transferred by P2 virtual work, conserving force and undeformed moment; no footprint reconstruction or radius fit. Measured rigid-balance correction and deformed-coordinate moment differences are retained. This isolates linear elastic/discretization response and does not qualify contact or a failed source history.",
        },
    )


if __name__ == "__main__":
    main()
