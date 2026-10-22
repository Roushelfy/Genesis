"""Compare point and distributed Hertz boundary corrections on captured curved material."""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import qr
from scipy.sparse.linalg import splu
from skfem import Basis, ElementTetP2, ElementVector, MeshTet1, MeshTet2, asm
from skfem.helpers import sym_grad
from skfem.models.elasticity import lame_parameters, linear_elasticity, linear_stress

from ..mechanics import shell_mesh
from ..skfem_reference import vector_mass
from .artifacts import file_id, runtime_identity, write_json
from .bilinear_normal_stress import point_tensor
from .check_curved_shell_operators import locate_curved
from .study_curved_contact_gap import curved_depth
from .study_franka_contact_gap import von_mises
from .study_hertz_residual import elliptic_stress
from .study_hertz_shell_quadrature import closest_barycentric
from .study_quadratic_rest_surface import quadratic_basis


def boundary_correction(coordinates, face_nodes, orientation, is_outer, axes, force, peak, order):
    """Integrate both particular tractions in quadratic face charts."""
    knots, weights = np.polynomial.legendre.leggauss(order)
    knots, weights = (knots + 1) / 2, weights / 2
    u, v = np.meshgrid(knots, knots, indexing="ij")
    quadrature_weights = weights[:, None] * weights[None]
    loads = np.zeros((2, len(coordinates), 3))
    samples, focused = [0, 0], 0
    for face, sign, outer_face in zip(face_nodes, orientation, is_outer):
        nodes = coordinates[face]
        triangle = nodes[:3]
        anchor = closest_barycentric(triangle, np.zeros(3))
        anchor_shape, _, _ = quadratic_basis(anchor[None])
        distance = np.linalg.norm(anchor_shape[0] @ nodes)
        diameter = max(np.linalg.norm(triangle[a] - triangle[b]) for a, b in [(0, 1), (0, 2), (1, 2)])
        rules = []
        if distance < diameter / 4:
            focused += 1
            for edge in range(3):
                endpoints = np.eye(3)[[edge, (edge + 1) % 3]]
                chart = endpoints[:, 1:] - anchor[1:]
                chart_area = abs(np.linalg.det(chart))
                if chart_area < 1e-14:
                    continue
                first, second = (endpoints - anchor) @ triangle
                for i_u in range(2):
                    for i_v in range(2):
                        radial, angular = (u + i_u) / 2, (v + i_v) / 2
                        length = np.linalg.norm((1 - angular)[..., None] * first + angular[..., None] * second, axis=2)
                        scale = max(distance, axes[0])
                        denominator = length * (1 - radial) + scale * radial
                        radius = scale * radial / denominator
                        derivative = scale * length / denominator**2
                        edge_bary = (1 - angular)[..., None] * endpoints[0] + angular[..., None] * endpoints[1]
                        bary = anchor + radius[..., None] * (edge_bary - anchor)
                        methods = (0, 1)
                        if distance < 1e-12:
                            # Surface tangency cancels the point singularity against
                            # the fan Jacobian; a linear radial rule resolves its
                            # smooth remaining integrand across the entire face.
                            point_bary = anchor + radial[..., None] * (edge_bary - anchor)
                            rules.append((point_bary, chart_area * radial * quadrature_weights / 4, (0,)))
                            methods = (1,)
                        rules.append((bary, chart_area * radius * derivative * quadrature_weights / 4, methods))
        else:
            rules.append((np.stack([1 - u, u * (1 - v), u * v], axis=2), u * quadrature_weights, (0, 1)))
        for bary, measure, methods in rules:
            bary, measure = bary.reshape(-1, 3), measure.reshape(-1)
            shape, gradients, _ = quadratic_basis(bary)
            positions = shape @ nodes
            tangents = np.einsum("qna,ni->qia", gradients, nodes - nodes[0])
            area = sign * np.cross(tangents[..., 0], tangents[..., 1])
            assert positions[:, 2].min() >= -1e-12
            positions[:, 2] = np.maximum(positions[:, 2], 0)
            for index in methods:
                tensor = (
                    force * point_tensor(positions)
                    if index == 0
                    else elliptic_stress(positions, axes, peak=peak, order=96)
                )
                if index == 1:
                    tensor -= force * point_tensor(positions)
                traction = -np.einsum("qij,qj->qi", tensor, area)
                if index == 1 and outer_face:
                    # Integrate the distributed residual jointly: its two large
                    # near-surface pressure terms cancel before quadrature.
                    pressure = peak * np.sqrt(np.maximum(1 - np.sum((positions[:, :2] / axes) ** 2, axis=1), 0))
                    traction[:, 2] += pressure * np.maximum(-area[:, 2], 0)
                loads[index, face] += shape.T @ (measure[:, None] * traction)
                samples[index] += len(measure)
    # The smooth distributed-minus-point correction is localized around the
    # pressure footprint; the broad point term uses its own converged rule.
    loads[1] += loads[0]
    return loads, {
        "point_quadrature_samples": samples[0],
        "distributed_quadrature_samples": samples[1],
        "focused_faces": focused,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {
        "wrench": 1e-4,
        "complete_field": 0.05,
        "sample_peak": 0.10,
        "reference_field_change": 0.01,
        "reference_peak_change": 0.02,
        "reference_wrench": 1e-4,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "frame": file_id(args.frame),
            "geometry": file_id(args.geometry),
            "targets": targets,
            "scope": "Captured first-loaded normal source on the coarse curved shell. Compare the inexpensive all-boundary point-traction correction with distributed prescribed-Hertz traction cancellation and the same independent curved P2 elasticity. This is a decomposition and quadrature check; it does not independently validate contact pressure or admit a native Franka update. Tangential loading is excluded explicitly.",
        },
    )
    with np.load(args.frame) as data:
        center, frame = data["hertz_centers_m"][0], data["hertz_frames"][0]
        direction = data["law_direction"][0]
        frame[:, :2] = frame[:, :2] @ np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
        axes, force, peak = data["law_axes_m"][0], data["law_force_n"][0], data["law_pressure_pa"][0]
        offsets = data["probe_offsets"]
    xyz, tets, _, metadata = shell_mesh(1, 2, 0.0005)
    affine = MeshTet1(xyz.T, tets.T, sort_t=False)
    scalar = Basis(affine, ElementTetP2())
    element, lame = ElementVector(ElementTetP2()), lame_parameters(1e10, 0.3)
    rows, arrays = [], {}
    for kind in ["analytic", "estimated"]:
        with np.load(args.geometry) as data:
            volume_nodes = (data[kind + "_volume_nodes_m"] - center) @ frame
        coordinates = np.empty((scalar.N, 3))
        coordinates[scalar.element_dofs.T] = volume_nodes[:, [0, 1, 2, 3, 4, 7, 5, 6, 8, 9]]
        mesh = MeshTet2(coordinates.T, tets.T, sort_t=False)
        facets = mesh.boundary_facets()
        faces, owners = mesh.facets[:, facets].T, mesh.f2t[0, facets]
        edge_lookup = {tuple(edge): node for edge, node in zip(affine.edges.T, scalar.edge_dofs[0])}
        mids = np.array(
            [[edge_lookup[tuple(sorted((face[a], face[b])))] for a, b in [(0, 1), (0, 2), (1, 2)]] for face in faces]
        )
        face_nodes = np.column_stack([faces, mids])
        triangle = coordinates[faces]
        area = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        orientation = np.sign(
            np.einsum("fi,fi->f", area, triangle.mean(axis=1) - coordinates[tets[owners]].mean(axis=1))
        )
        knots, weights = np.polynomial.legendre.leggauss(7)
        knots, weights = (knots + 1) / 2, weights / 2
        u, v, w = np.array(np.meshgrid(knots, knots, knots, indexing="ij")).reshape(3, -1)
        wu, wv, ww = np.array(np.meshgrid(weights, weights, weights, indexing="ij")).reshape(3, -1)
        basis = Basis(
            mesh,
            element,
            quadrature=(np.array([u, (1 - u) * v, (1 - u) * (1 - v) * w]), wu * wv * ww * (1 - u) ** 2 * (1 - v)),
        )
        stiffness, mass = asm(linear_elasticity(*lame), basis).tocsr(), asm(vector_mass, basis, density=2000).tocsr()
        dofs = np.stack(basis.split_indices(), axis=1)
        modes = np.zeros((basis.N, 6))
        modes[dofs, np.arange(3)[None]] = 1
        for axis in range(3):
            modes[dofs, axis + 3] = np.cross(np.eye(3)[axis], coordinates)
        mass_modes, gram = mass @ modes, modes.T @ mass @ modes
        _, _, pivot = qr((modes / np.linalg.norm(modes, axis=0)).T, mode="economic", pivoting=True)
        free = np.setdiff1d(np.arange(basis.N), pivot[:6])
        factor = splu((stiffness[free][:, free] / 1e10).tocsc())
        # The polar pressure rule integrates the projected ellipse exactly in force.
        radial, radial_weight = np.polynomial.legendre.leggauss(24)
        radial, radial_weight = (radial + 1) / 2, radial_weight / 2
        theta = 2 * np.pi * (np.arange(192) + 0.5) / 192
        xy = (
            np.sqrt(1 - radial[:, None] ** 2)[..., None] * (np.column_stack([np.cos(theta), np.sin(theta)]) * axes)
        ).reshape(-1, 2)
        outer = volume_nodes[:240:3, [0, 1, 2, 4, 5, 7]]
        depths = curved_depth(xy, outer, -center @ frame)
        locations = np.column_stack([xy, depths])
        weights_force = np.repeat(3 * radial**2 * radial_weight * force / 192, 192)
        physical = np.zeros_like(coordinates)
        for start in range(0, len(locations), 256):
            chosen, bary, valid, _ = locate_curved(volume_nodes, locations[start : start + 256])
            assert valid.all()
            values, _, _ = quadratic_basis(bary)
            # Native geometry order differs from the independent basis edge order.
            local_nodes = scalar.element_dofs.T[chosen][:, [0, 1, 2, 3, 4, 6, 7, 5, 8, 9]]
            np.add.at(
                physical[:, 2], local_nodes.reshape(-1), (values * weights_force[start : start + 256, None]).reshape(-1)
            )
        np.testing.assert_allclose(physical.sum(axis=0), [0, 0, force], atol=1e-13)
        assert np.linalg.norm(np.cross(coordinates, physical).sum(axis=0)) / (force * axes[0]) < 1e-10
        corner_bary = np.eye(4)
        field_basis = Basis(mesh, element, quadrature=(corner_bary[:, 1:].T, np.ones(4)))
        corners = field_basis.mapping.F(field_basis.X).transpose(1, 2, 0)
        assert corners[..., 2].min() >= -1e-12
        particular_corners = elliptic_stress(
            np.maximum(corners.reshape(-1, 3), [-np.inf, -np.inf, 0]), axes, peak=peak, order=96
        ).reshape(len(tets), 4, 3, 3)
        local = offsets * [axes[0], axes[1], axes[0]]
        local[:, 2] += curved_depth(local[:, :2], outer, -center @ frame)
        selected, probe_bary, valid, _ = locate_curved(volume_nodes, local)
        assert valid.all()
        _, gradients, _ = quadratic_basis(probe_bary)
        jacobian = np.einsum("qna,qni->qia", gradients, volume_nodes[selected] - volume_nodes[selected, :1])
        gradient = gradients @ np.linalg.inv(jacobian)
        probe_nodes = scalar.element_dofs.T[selected][:, [0, 1, 2, 3, 4, 6, 7, 5, 8, 9]]
        particular_probes = elliptic_stress(local, axes, peak=peak, order=96)
        fields, probes = [], []
        for order in [16, 24, 32]:
            loads, counts = boundary_correction(
                coordinates,
                face_nodes,
                orientation,
                np.all(faces < metadata.outer_surface_vertices, axis=1),
                axes,
                force,
                peak,
                order,
            )
            for method, load in zip(["point", "distributed"], loads):
                rhs = load.reshape(-1)
                rhs = rhs - mass_modes @ np.linalg.solve(gram, modes.T @ rhs)
                displacement = np.zeros(basis.N)
                displacement[free] = factor.solve(rhs[free] / 1e10)
                combined = (
                    linear_stress(*lame)(sym_grad(field_basis.interpolate(displacement))).transpose(2, 3, 0, 1)
                    + particular_corners
                )
                nodal = displacement.reshape(-1, 3)[probe_nodes]
                derivative = np.einsum("qni,qnj->qij", nodal - nodal[:, :1], gradient)
                tensor = lame[1] * (derivative + derivative.transpose(0, 2, 1))
                tensor += lame[0] * np.trace(derivative, axis1=1, axis2=2)[:, None, None] * np.eye(3)
                tensor += particular_probes
                complete = np.concatenate([combined.reshape(-1, 3, 3), tensor])
                wrench = max(
                    np.linalg.norm(load.sum(axis=0) - [0, 0, force]) / force,
                    np.linalg.norm(np.cross(coordinates, load).sum(axis=0)) / (force * axes[0]),
                )
                row = {
                    "geometry": kind,
                    "order": order,
                    "method": method,
                    **counts,
                    "wrench_relative": float(wrench),
                    "von_mises_peak_pa": float(von_mises(complete).max()),
                    "tensile_peak_pa": float(np.linalg.eigvalsh(complete)[:, -1].max()),
                }
                rows.append(row)
                fields.append(combined)
                probes.append(tensor)
            write_json(args.output / "progress.json", {"rows": rows})
        reference, reference_probes = fields[-1], probes[-1]
        final_peaks = rows[-1]
        for row, field, probe in zip(rows[-6:], fields, probes):
            row["corner_field_relative"] = float(np.linalg.norm(field - reference) / np.linalg.norm(reference))
            row["probe_field_relative"] = float(
                np.linalg.norm(probe - reference_probes) / np.linalg.norm(reference_probes)
            )
            row["vm_peak_relative"] = abs(row["von_mises_peak_pa"] / final_peaks["von_mises_peak_pa"] - 1)
            row["tensile_peak_relative"] = abs(row["tensile_peak_pa"] / final_peaks["tensile_peak_pa"] - 1)
        arrays[kind + "_corners_pa"] = np.array(fields)
        arrays[kind + "_probes_pa"] = np.array(probes)
        arrays[kind + "_probe_positions_m"] = local
        arrays[kind + "_last_loads_n"] = loads
    np.savez_compressed(args.output / "fields.npz", **arrays)
    converged = all(
        max(row["corner_field_relative"], row["probe_field_relative"]) < targets["reference_field_change"]
        and max(row["vm_peak_relative"], row["tensile_peak_relative"]) < targets["reference_peak_change"]
        for row in rows
        if row["order"] == 24 and row["method"] == "distributed"
    )
    converged = converged and all(
        row["wrench_relative"] < targets["reference_wrench"]
        for row in rows
        if row["order"] == 32 and row["method"] == "distributed"
    )
    candidates = {
        kind: all(
            row["wrench_relative"] < targets["wrench"]
            and max(row["corner_field_relative"], row["probe_field_relative"]) < targets["complete_field"]
            and max(row["vm_peak_relative"], row["tensile_peak_relative"]) < targets["sample_peak"]
            for row in rows
            if row["geometry"] == kind and row["method"] == "point" and row["order"] == 16
        )
        for kind in ["analytic", "estimated"]
    }
    write_json(
        args.output / "result.json",
        {
            "passed": converged,
            "targets": targets,
            "rows": rows,
            "point16_candidates": candidates,
            "whole_goal_complete": False,
        },
    )
    assert converged


if __name__ == "__main__":
    main()
