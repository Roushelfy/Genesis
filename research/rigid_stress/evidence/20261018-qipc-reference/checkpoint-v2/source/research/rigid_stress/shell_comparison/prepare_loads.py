"""Integrate one finite, balanced egg compression for independent volume and shell mechanics.

The same quadrature forces on the outer surface feed quadratic volume FEM and
linear midpoint-shell vertices. Their explicit position/moment difference is
retained; this separates a load-transfer approximation from the elastic solve.
"""

import argparse
from pathlib import Path

import numpy as np

from ..finite_traction_reference import assemble_finite_patch, physical_grip_patches, triangle_closest_points
from ..skfem_reference import SkfemElasticReference
from .common import file_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for level in (1, 2):
        path = args.inputs / f"egg-level{level}.npz"
        with np.load(path) as saved:
            vertices, tets = saved["vertices"], saved["tetrahedra"]
            midpoint, faces = saved["mid_vertices"], saved["mid_faces"]
        count = len(midpoint)
        reference = SkfemElasticReference(vertices, tets, 1e10, 0.3, 2000.0)
        patches = [
            assemble_finite_patch(reference, specification, 3)
            for specification in physical_grip_patches("egg", "opposing", 0.006)
        ]
        force = sum((patch.load_vector_n for patch in patches), np.zeros(reference.basis.N))
        solution = reference.solve(force)
        assert solution.residual_budget_fraction <= 1
        shell_force = np.zeros_like(midpoint)
        cloud_points, cloud_forces, cloud_normals, cloud_areas = [], [], [], []
        for patch in patches:
            indices = reference.mesh.facets[:, patch.boundary_facets].T
            assert (indices < count).all(), "The prescribed footprint must stay on the outer connected sheet."
            triangles = vertices[indices]
            # The reference packet retains only active quadrature points, not
            # a rectangular face-by-point array. Recover their containing face.
            closest = [triangle_closest_points(triangles, point)[1] for point in patch.positions_m]
            distances = np.array(closest)
            selected = distances.argmin(axis=1)
            assert distances.min(axis=1).max() <= 1e-20
            point_triangles = triangles[selected]
            point_indices = indices[selected]
            edges = point_triangles[:, 1:] - point_triangles[:, :1]
            gram = np.einsum("qai,qbi->qab", edges, edges)
            local = np.linalg.solve(
                gram, np.einsum("qai,qi->qa", edges, patch.positions_m - point_triangles[:, 0])[..., None]
            )[..., 0]
            barycentric = np.column_stack((1 - local.sum(axis=1), local))
            weighted = patch.tractions_pa * patch.area_weights_m2[:, None]
            for corner in range(3):
                np.add.at(shell_force, point_indices[:, corner], barycentric[:, corner, None] * weighted)
            cloud_points.append(patch.positions_m)
            cloud_forces.append(patch.tractions_pa * patch.area_weights_m2[:, None])
            cloud_normals.append(patch.normals)
            cloud_areas.append(patch.area_weights_m2)
        # Sample volume tensors at the same face-centroid fibers. The tiny inset
        # prevents a boundary point from being classified outside by roundoff.
        outer = vertices[:count][faces].mean(axis=1)
        inner = vertices[2 * count : 3 * count][faces].mean(axis=1)
        centers = midpoint[faces].mean(axis=1)
        samples = np.stack((inner, centers, outer), axis=1)
        samples = samples * (1 - 1e-6) + centers[:, None] * 1e-6
        tensor, vm = reference.stress_at_points(solution.displacement, samples.reshape(-1, 3))
        tensor = tensor.reshape(len(faces), 3, 3, 3)
        vm = vm.reshape(len(faces), 3)
        quadrature_points, quadrature_forces = np.concatenate(cloud_points), np.concatenate(cloud_forces)
        np.savez_compressed(
            args.output / f"level{level}.npz",
            mid_vertices=midpoint,
            mid_faces=faces,
            shell_force_n=shell_force,
            volume_force_n=force,
            volume_displacement_m=solution.displacement,
            reference_tensor_pa=tensor,
            reference_vm_pa=vm,
            sample_positions_m=samples,
            cloud_points_m=quadrature_points,
            cloud_forces_n=quadrature_forces,
            cloud_normals=np.concatenate(cloud_normals),
            cloud_areas_m2=np.concatenate(cloud_areas),
        )
        rows.append(
            {
                "level": level,
                "source": file_identity(path),
                "reference_residual_budget_fraction": float(solution.residual_budget_fraction),
                "reference_peak_pa": float(solution.peak_pa),
                "reference_energy_j": float(solution.energy_j),
                "cloud_force_n": quadrature_forces.sum(axis=0).tolist(),
                "shell_force_n": shell_force.sum(axis=0).tolist(),
                "cloud_moment_origin_nm": np.cross(quadrature_points, quadrature_forces).sum(axis=0).tolist(),
                "shell_moment_origin_nm": np.cross(midpoint, shell_force).sum(axis=0).tolist(),
                "cloud_absolute_force_n": float(np.linalg.norm(quadrature_forces, axis=1).sum()),
                "fiber_sampling_relative_inset": 1e-6,
            }
        )
    write_json(args.output / "result.json", {"kind": "matched-finite-compression", "cases": rows})


if __name__ == "__main__":
    main()
