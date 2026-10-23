"""Check curved P2 shell operators and material probes before a native geometry change."""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import qr
from scipy.sparse.linalg import splu
from skfem import Basis, ElementTetP2, ElementVector, MeshTet1, MeshTet2, asm
from skfem.helpers import sym_grad
from skfem.models.elasticity import lame_parameters, linear_elasticity, linear_stress
from skfem.quadrature import get_quadrature_tri

from ..mechanics import shell_mesh
from ..skfem_reference import vector_mass
from .artifacts import file_id, runtime_identity, write_json
from .study_curved_contact_gap import curved_depth
from .study_quadratic_rest_surface import quadratic_basis


def locate_curved(nodes, points):
    inverse = np.linalg.inv((nodes[:, 1:4] - nodes[:, :1]).transpose(0, 2, 1))
    coordinates = np.einsum("tij,tqj->tqi", inverse, points[None] - nodes[:, None, 0])
    for _ in range(10):
        bary = np.concatenate([1 - coordinates.sum(axis=2, keepdims=True), coordinates], axis=2)
        values, gradient, _ = quadratic_basis(bary.reshape(-1, 4))
        location = np.einsum("tqn,tni->tqi", values.reshape(*bary.shape[:2], 10), nodes)
        jacobian = np.einsum("tqna,tni->tqia", gradient.reshape(*bary.shape[:2], 10, 3), nodes)
        coordinates -= np.linalg.solve(jacobian, (location - points[None])[..., None])[..., 0]
    bary = np.concatenate([1 - coordinates.sum(axis=2, keepdims=True), coordinates], axis=2)
    values, _, _ = quadratic_basis(bary.reshape(-1, 4))
    location = np.einsum("tqn,tni->tqi", values.reshape(*bary.shape[:2], 10), nodes)
    error = np.linalg.norm(location - points[None], axis=2)
    inside = (bary.min(axis=2) >= -1e-9) & (bary.max(axis=2) <= 1 + 1e-9) & (error < 1e-12)
    selected = np.argmax(inside, axis=0)
    return selected, bary[selected, np.arange(len(points))], inside.any(axis=0), error[selected, np.arange(len(points))]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {
        "uniform_stress_relative": 1e-7,
        "patch_load_relative": 1e-9,
        "rigid_mode_relative": 1e-12,
        "inverse_map_m": 1e-12,
        "quadrature_mass_relative": 1e-8,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "frame": file_id(args.frame),
            "geometry": file_id(args.geometry),
            "targets": targets,
            "scope": "Independent isoparametric P2 stiffness, consistent mass, all-boundary uniform-traction patch test and curved-material probe containment on the captured coarse shell. No native curved geometry, contact-pressure solve or physical contact-stress accuracy is claimed.",
        },
    )
    xyz, tets, _, _ = shell_mesh(1, 2, 0.0005)
    affine = MeshTet1(xyz.T, tets.T, sort_t=False)
    scalar = Basis(affine, ElementTetP2())
    element = ElementVector(ElementTetP2())
    lame = lame_parameters(1e10, 0.3)
    tensor = np.array([[1.2e6, 0.2e6, -0.15e6], [0.2e6, -0.6e6, 0.1e6], [-0.15e6, 0.1e6, 0.3e6]])
    strain = ((1 + 0.3) * tensor - 0.3 * np.trace(tensor) * np.eye(3)) / 1e10
    with np.load(args.frame) as saved:
        center, basis = saved["hertz_centers_m"][0], saved["hertz_frames"][0]
        direction, axes, offsets = saved["law_direction"][0], saved["law_axes_m"][0], saved["probe_offsets"]
        basis[:, :2] = basis[:, :2] @ np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
    rows, arrays = [], {}
    masses = {}
    for kind in ["analytic", "estimated"]:
        with np.load(args.geometry) as saved:
            nodes = saved[kind + "_volume_nodes_m"]
        coordinates = np.empty((scalar.N, 3))
        # scikit-fem orders tetrahedron edges01,12,02,03,13,23.
        reordered = nodes[:, [0, 1, 2, 3, 4, 7, 5, 6, 8, 9]]
        coordinates[scalar.element_dofs.T] = reordered
        np.testing.assert_allclose(coordinates[scalar.element_dofs.T], reordered, atol=1e-14, rtol=0)
        mesh = MeshTet2(coordinates.T, tets.T, sort_t=False)
        facets = mesh.facets[:, mesh.boundary_facets()].T
        owners = mesh.f2t[0, mesh.boundary_facets()]
        edge_lookup = {tuple(edge): node for edge, node in zip(affine.edges.T, scalar.edge_dofs[0])}
        mids = np.array(
            [[edge_lookup[tuple(sorted((face[i], face[j])))] for i, j in [(0, 1), (0, 2), (1, 2)]] for face in facets]
        )
        boundary_nodes = np.column_stack([facets, mids])
        boundary_coordinates = coordinates[boundary_nodes]
        original_normal = np.cross(
            boundary_coordinates[:, 1] - boundary_coordinates[:, 0],
            boundary_coordinates[:, 2] - boundary_coordinates[:, 0],
        )
        orientation = np.sign(
            np.einsum(
                "fi,fi->f", original_normal, boundary_coordinates[:, :3].mean(axis=1) - xyz[tets[owners]].mean(axis=1)
            )
        )
        local = offsets * [axes[0], axes[1], axes[0]]
        surface = (nodes[:240:3, [0, 1, 2, 4, 5, 7]] - center) @ basis
        depth = curved_depth(local[:, :2], surface, -center @ basis)
        corrected = local.copy()
        corrected[:, 2] += depth
        points = center + corrected @ basis.T
        owners, bary, valid, error = locate_curved(nodes, points)
        assert valid.all(), (kind, np.flatnonzero(~valid))
        assert error.max() < targets["inverse_map_m"]
        arrays[kind + "_probe_points_m"] = points
        arrays[kind + "_probe_elements"] = owners
        arrays[kind + "_probe_barycentric"] = bary
        for order in [8, 9]:
            field_basis = Basis(mesh, element, intorder=order)
            dofs = np.stack(field_basis.split_indices(), axis=1)
            stiffness = asm(linear_elasticity(*lame), field_basis).tocsr()
            mass = asm(vector_mass, field_basis, density=2000).tocsr()
            # Integrate directly in each boundary chart; global inverse mapping
            # from a generic initial guess is unreliable for the thin curved cells.
            knots, weights = get_quadrature_tri(order)
            face_bary = np.column_stack([1 - knots.sum(axis=0), knots.T])
            shape, gradients, _ = quadratic_basis(face_bary)
            tangents = np.einsum("qna,fni->fqia", gradients, boundary_coordinates)
            normal_area = np.cross(tangents[..., 0], tangents[..., 1]) * orientation[:, None, None]
            traction = normal_area @ tensor.T
            nodal = np.einsum("qn,fqi,q->fni", shape, traction, weights)
            load = np.zeros(field_basis.N)
            np.add.at(load, dofs[boundary_nodes].reshape(-1), nodal.reshape(-1))
            exact = np.empty(field_basis.N)
            exact[dofs] = coordinates @ strain.T
            patch_error = np.linalg.norm(stiffness @ exact - load) / np.linalg.norm(load)
            modes = np.zeros((field_basis.N, 6))
            modes[dofs, np.arange(3)[None]] = 1
            for axis in range(3):
                modes[dofs, axis + 3] = np.cross(np.eye(3)[axis], coordinates)
            rigid_error = np.linalg.norm(stiffness @ modes) / np.linalg.norm(stiffness.data) / np.linalg.norm(modes)
            scaled = modes / np.linalg.norm(modes, axis=0)
            _, _, pivot = qr(scaled.T, mode="economic", pivoting=True)
            free = np.setdiff1d(np.arange(field_basis.N), pivot[:6])
            mass_modes = mass @ modes
            gram = modes.T @ mass_modes
            rhs = load - mass_modes @ np.linalg.solve(gram, modes.T @ load)
            displacement = np.zeros(field_basis.N)
            displacement[free] = splu((stiffness[free][:, free] / 1e10).tocsc()).solve(rhs[free] / 1e10)
            recovered = linear_stress(*lame)(sym_grad(field_basis.interpolate(displacement))).transpose(2, 3, 0, 1)
            stress_error = (
                np.linalg.norm(recovered - tensor)
                / np.sqrt(recovered.shape[0] * recovered.shape[1])
                / np.linalg.norm(tensor)
            )
            scalar_mass = mass[dofs[:, 0]][:, dofs[:, 0]].sum()
            masses[kind, order] = scalar_mass
            ratio = field_basis.mapping.detDF(field_basis.X) / affine.mapping().detDF(field_basis.X)
            assert ratio.min() > 0.2
            row = {
                "normals": kind,
                "quadrature_order": order,
                "nodes": len(coordinates),
                "dofs": int(field_basis.N),
                "minimum_jacobian_ratio": float(ratio.min()),
                "uniform_stress_relative": float(stress_error),
                "patch_load_relative": float(patch_error),
                "rigid_mode_relative": float(rigid_error),
                "mass_kg": float(scalar_mass),
                "material_probes": len(points),
                "maximum_probe_surface_shift_m": float(depth.max()),
                "maximum_probe_inverse_error_m": float(error.max()),
            }
            rows.append(row)
            write_json(args.output / "progress.json", {"rows": rows})
            assert stress_error < targets["uniform_stress_relative"] and patch_error < targets["patch_load_relative"]
            assert rigid_error < targets["rigid_mode_relative"]
    changes = {kind: float(abs(masses[kind, 8] / masses[kind, 9] - 1)) for kind in ["analytic", "estimated"]}
    assert max(changes.values()) < targets["quadrature_mass_relative"]
    np.savez_compressed(args.output / "probes.npz", **arrays)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "targets": targets,
            "quadrature_mass_changes": changes,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
