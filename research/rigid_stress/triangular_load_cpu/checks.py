"""Independent analytical and work controls for triangular shell loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from research.rigid_stress.shell_cpu.contacts import Contact, ShellDPRTransfer
from research.rigid_stress.triangular_load_cpu.proxy import TriangularLoadTransfer
from research.rigid_stress.triangular_load_cpu.reference import Q4LoadTransfer, inverse_bilinear


@dataclass(frozen=True)
class SurfaceGeometry:
    """Supply physical shell geometry to load-only controls."""

    vertices: np.ndarray
    quads: np.ndarray
    directors: np.ndarray
    h: float = 0.0005
    E: float = 1e10
    nu: float = 0.3

    @property
    def ndof(self):
        return 6 * len(self.vertices)


def _shape(coordinates):
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    values = np.atleast_2d(coordinates)
    return 0.25 * (1 + values[:, :1] * signs[:, 0]) * (1 + values[:, 1:] * signs[:, 1])


def _cross_matrix(vector):
    first, second, third = vector
    return np.array([[0.0, -third, second], [third, 0.0, -first], [-second, first, 0.0]])


def _relative(value, reference):
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-30))


def _wrench(model, load, origin):
    grouped = load.reshape((-1, 6))
    force = grouped[:, :3].sum(axis=0)
    moment = np.cross(model.vertices - origin, grouped[:, :3]).sum(axis=0) + grouped[:, 3:].sum(axis=0)
    return force, moment


def _kinematic_map(model, face, coordinates, side):
    """Build the independent physical-displacement Jacobian at fixed material points."""
    nodes = model.quads[face]
    weights = _shape(coordinates)
    mapping = np.zeros((3 * len(weights), model.ndof))
    offsets = side * 0.5 * model.h * model.directors[nodes]
    for i_point, row in enumerate(weights):
        for i_local, node in enumerate(nodes):
            mapping[3 * i_point:3 * i_point + 3, 6 * node:6 * node + 3] = row[i_local] * np.eye(3)
            mapping[3 * i_point:3 * i_point + 3, 6 * node + 3:6 * node + 6] = -row[i_local] * _cross_matrix(offsets[i_local])
    positions = weights @ (model.vertices[nodes] + offsets)
    return mapping, positions, weights


def _work_controls(model):
    generator = np.random.default_rng(702134)
    coordinates = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-0.17, 0.49]])
    direction = generator.normal(size=model.ndof)
    force = generator.normal(size=(len(coordinates), 3))
    translation = np.array([0.013, -0.007, 0.021])
    rotation = np.array([0.3, -0.2, 0.1])
    rigid = np.empty((len(model.vertices), 6))
    rigid[:, :3] = translation + np.cross(rotation, model.vertices)
    rigid[:, 3:] = rotation
    rows = []
    for side in (-1, 1):
        mapping, positions, weights = _kinematic_map(model, 0, coordinates, side)
        load = mapping.T @ force.reshape(-1)
        displacement = (mapping @ direction).reshape((-1, 3))
        work_first = direction @ load
        work_second = np.sum(displacement * force)
        scale = max(abs(work_first), abs(work_second), np.linalg.norm(direction) * np.linalg.norm(load), 1e-30)
        work_error = abs(work_first - work_second) / scale
        rigid_expected = translation + np.cross(rotation, positions)
        rigid_error = _relative((mapping @ rigid.reshape(-1)).reshape((-1, 3)), rigid_expected)
        total_force, total_moment = _wrench(model, load, np.zeros(3))
        force_error = _relative(total_force, force.sum(axis=0))
        expected_moment = np.cross(positions, force).sum(axis=0)
        moment_error = _relative(total_moment, expected_moment)
        # The vertex interpolation is extended linearly over one proxy triangle.
        barycentric = np.array([0.21, 0.34, 0.45])
        triangle_map = np.einsum("i,ijk->jk", barycentric, mapping.reshape((-1, 3, model.ndof))[:3])
        triangle_position = barycentric @ positions[:3]
        triangle_error = _relative(triangle_map @ rigid.reshape(-1), translation + np.cross(rotation, triangle_position))
        rows.append(dict(name=f"work_rigid_wrench_side_{side}", passed=max(work_error, rigid_error, force_error, moment_error, triangle_error) < 2e-13,
                         virtual_work_relative=work_error, rigid_reproduction_relative=rigid_error,
                         total_force_relative=force_error, total_moment_relative=moment_error,
                         triangle_rigid_reproduction_relative=triangle_error,
                         partition_unity_error=float(np.max(abs(weights.sum(axis=1) - 1)))))
    return rows


def _affine_model():
    first = np.array([0.018, 0.002, 0.0])
    second = np.array([0.008, 0.014, 0.0])
    vertices = np.array([-first - second, first - second, first + second, -first + second])
    return SurfaceGeometry(vertices, np.array([[0, 1, 2, 3]]), np.tile([0.0, 0.0, 1.0], (4, 1)))


def _affine_disk_load(model, contact):
    """Integrate bilinear Q4 loading with independent closed Hertz moments."""
    physical = model.vertices + contact.side * 0.5 * model.h * model.directors
    origin = physical.mean(axis=0)
    first = (-physical[0] + physical[1] + physical[2] - physical[3]) / 4
    second = (-physical[0] - physical[1] + physical[2] + physical[3]) / 4
    inverse = np.linalg.pinv(np.column_stack((first, second)))
    coordinate = inverse @ (contact.point - origin)
    covariance_cross = contact.radius**2 / 5 * np.dot(inverse[0], inverse[1])
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    integrated_shape = 0.25 * (1 + signs[:, 0] * coordinate[0] + signs[:, 1] * coordinate[1]
                              + signs[:, 0] * signs[:, 1] * (coordinate[0] * coordinate[1] + covariance_cross))
    load = np.zeros((4, 6))
    load[:, :3] = integrated_shape[:, None] * contact.force
    load[:, 3:] = contact.side * 0.5 * model.h * np.cross(model.directors, load[:, :3])
    return load.reshape(-1), float(covariance_cross)


def _affine_controls(proxy_type):
    model = _affine_model()
    physical_coordinate = np.array([0.23, -0.31])
    rows = []
    for side in (-1, 1):
        physical = model.vertices + side * 0.5 * model.h * model.directors
        point = _shape(physical_coordinate)[0] @ physical
        normal = np.array([0.0, 0.0, float(side)])
        contact = Contact(point, -0.05 * normal + np.array([0.012, 0.004, 0.0]), normal,
                          radius=0.0025, friction=0.5, side=side, face_hint=0)
        exact, covariance_cross = _affine_disk_load(model, contact)
        computed, information = Q4LoadTransfer(model).transfer(contact, order=32)
        force, moment = _wrench(model, computed, point)
        rows.append(dict(name=f"affine_Q4_closed_disk_side_{side}", passed=_relative(computed, exact) < 2e-10,
                         load_relative=_relative(computed, exact), force_relative=_relative(force, contact.force),
                         normalized_original_point_moment=float(np.linalg.norm(moment) / (np.linalg.norm(force) * contact.radius)),
                         covariance_xi_eta=covariance_cross, info=information))
        proxy = proxy_type(model, subdivisions=1)
        analytical, analytic_info = proxy.transfer(contact, order=24, analytic=True, anchor_mode="fixed")
        numerical, numerical_info = proxy.transfer(contact, order=48, analytic=False, anchor_mode="fixed")
        rows.append(dict(name=f"proxy_analytic_vs_own_numerical_side_{side}", passed=_relative(analytical, numerical) < 2e-9,
                         load_relative=_relative(analytical, numerical), analytic_info=analytic_info, numerical_info=numerical_info,
                         original_Q4_vs_proxy_relative=_relative(analytical, exact),
                         comparison_note="Proxy consistency is separate from original Q4 load equivalence."))
        released = Contact(point, np.zeros(3), normal, radius=contact.radius, side=side, face_hint=0)
        reference_zero, _ = Q4LoadTransfer(model).transfer(released)
        proxy_zero, _ = proxy.transfer(released)
        rows.append(dict(name=f"zero_release_side_{side}", passed=bool(np.count_nonzero(reference_zero) == 0 and np.count_nonzero(proxy_zero) == 0)))
        frictionless = Contact(point, contact.force, normal, radius=contact.radius, friction=0.0, side=side, face_hint=0)
        for method, transfer in (("Q4", Q4LoadTransfer(model)), ("triangle", proxy)):
            rejected = False
            rejection = None
            try:
                transfer.transfer(frictionless, order=24)
            except ValueError as error:
                rejected = True
                rejection = str(error)
            rows.append(dict(name=f"explicit_zero_friction_rejects_shear_{method}_side_{side}", passed=rejected, rejection=rejection))
    return rows


def _warped_control(proxy_type):
    vertices = np.array([[-0.01, -0.01, 0.002], [0.01, -0.01, -0.002],
                         [0.01, 0.01, 0.002], [-0.01, 0.01, -0.002]])
    model = SurfaceGeometry(vertices, np.array([[0, 1, 2, 3]]), np.tile([0.0, 0.0, 1.0], (4, 1)))
    point = np.array([0.0, 0.0, 0.5 * model.h])
    contact = Contact(point, np.array([0.0, 0.0, -0.05]), np.array([0.0, 0.0, 1.0]),
                      radius=0.0002, friction=0.5, face_hint=0)
    reference, reference_info = Q4LoadTransfer(model).transfer(contact, order=48)
    rejected = False
    rejection = None
    try:
        proxy_type(model, subdivisions=1).transfer(contact, order=48, analytic=True, anchor_mode="fixed")
    except ValueError as error:
        rejected = True
        rejection = str(error)
    reanchored, reanchor_info = proxy_type(model, subdivisions=1).transfer(
        contact, order=48, analytic=True, anchor_mode="force_line"
    )
    reanchor_force, reanchor_moment = _wrench(model, reanchored, point)
    reanchor_wrench_error = max(_relative(reanchor_force, contact.force),
                               np.linalg.norm(reanchor_moment) / (np.linalg.norm(contact.force) * contact.radius))
    return dict(name="warped_Q4_real_support_coarse_triangle_miss", passed=bool(np.linalg.norm(reference) > 0 and rejected),
                original_support_valid=reference_info["valid"], radius_m=contact.radius,
                diagonal_normal_gap_m=0.002, fixed_proxy_rejected=rejected, rejection=rejection,
                force_line_diagnostic=reanchor_info, force_line_wrench_error=float(reanchor_wrench_error),
                meaning="A retained counterexample for coarse fixed facet geometry, not a numerical solve failure.")


def _tilted_proxy_control(proxy_type):
    vertices = np.array([[-0.01, -0.01, 0.001], [0.01, -0.01, -0.001],
                         [0.01, 0.01, 0.001], [-0.01, 0.01, -0.001]])
    model = SurfaceGeometry(vertices, np.array([[0, 1, 2, 3]]), np.tile([0.0, 0.0, 1.0], (4, 1)))
    coordinate = np.array([0.4, -0.35])
    point = _shape(coordinate)[0] @ (vertices + 0.5 * model.h * model.directors)
    normal = np.array([-0.1 * coordinate[1], -0.1 * coordinate[0], 1.0])
    normal /= np.linalg.norm(normal)
    force = -0.05 * normal + np.array([0.012, 0.0, 0.0])
    contact = Contact(point, force, normal, radius=0.002, friction=0.5, face_hint=0)
    transfer = proxy_type(model, subdivisions=1)
    analytical, analytic_info = transfer.transfer(contact, order=24, analytic=True, anchor_mode="fixed")
    numerical, numerical_info = transfer.transfer(contact, order=48, analytic=False, anchor_mode="fixed")
    error = _relative(analytical, numerical)
    return dict(name="tilted_triangle_offplane_center_affine_pressure", passed=bool(error < 2e-9 and analytic_info["analytic_count"] == 1),
                load_relative=error, analytic_info=analytic_info, numerical_info=numerical_info,
                meaning="Independent numerical integration exercises nonconstant positive affine pressure and eccentric friction.")


def _fixed_stiffness_controls(model, proxy_type):
    before = hashlib.sha256(model.K.data.tobytes() + model.K.indices.tobytes() + model.K.indptr.tobytes()).hexdigest()
    face = len(model.quads) // 3
    nodes = model.quads[face]
    physical = model.vertices[nodes] + 0.5 * model.h * model.directors[nodes]
    point = _shape([[0.23, -0.19]])[0] @ physical
    normal = np.cross(physical[1] - physical[0], physical[3] - physical[0])
    normal /= np.linalg.norm(normal)
    tangent = physical[1] - physical[0]
    tangent -= normal * np.dot(normal, tangent)
    tangent /= np.linalg.norm(tangent)
    contact = Contact(point, -0.05 * normal + 0.01 * tangent, normal,
                      radius=0.002, friction=0.5, face_hint=face)
    reference, _ = Q4LoadTransfer(model).transfer(contact, order=32)
    proxy, _ = proxy_type(model, subdivisions=4).transfer(contact, order=32, analytic=True, anchor_mode="fixed")
    loads = np.column_stack((reference, proxy))
    solution = model.solve(loads)
    fields = model.stress(solution)
    tensor = np.concatenate((fields["sigma_top"], fields["sigma_bottom"]), axis=1)
    residuals = [float(model.residual(solution[:, index], loads[:, index])) for index in range(2)]
    after = hashlib.sha256(model.K.data.tobytes() + model.K.indices.tobytes() + model.K.indptr.tobytes()).hexdigest()
    return dict(name="original_K_gauge_relief_stress_recovery_unchanged", passed=bool(before == after and max(residuals) < 1e-8),
                stiffness_sha256=before, solve_residuals=residuals,
                load_relative=_relative(proxy, reference), stress_tensor_relative=_relative(tensor[..., 1], tensor[..., 0]),
                M_relative=_relative(fields["M"][..., 1], fields["M"][..., 0]),
                Q_relative=_relative(fields["Q"][..., 1], fields["Q"][..., 0]),
                comparison_note="Spatial/load differences are reported without being hidden in numerical acceptance.")


def _inverse_controls():
    """Compare chart inversion against forward geometry and the original Newton oracle."""
    generator = np.random.default_rng(419703)
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    maximum_shape = 0.0
    maximum_coordinate = 0.0
    maximum_forward = 0.0
    maximum_old_forward = 0.0
    minimum_oriented_area = np.inf
    valid_closed = True
    valid_newton = True
    frozen = hashlib.sha256()
    for _ in range(100):
        angle = generator.uniform(-np.pi, np.pi)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        scale = generator.uniform(0.002, 0.04, size=2)
        first, second = rotation[:, 0] * scale[0], rotation[:, 1] * scale[1]
        mixed = generator.uniform(-0.12, 0.12, size=2) * min(scale)
        origin = generator.uniform(-0.08, 0.08, size=2)
        chart = origin + corners[:, :1] * first + corners[:, 1:] * second
        chart += (corners[:, 0] * corners[:, 1])[:, None] * mixed
        edges = np.roll(chart, -1, axis=0) - chart
        next_edges = np.roll(edges, -1, axis=0)
        oriented = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
        minimum_oriented_area = min(minimum_oriented_area, float(oriented.min()))
        coordinate = generator.uniform(-0.999, 0.999, size=(200, 2))
        coordinate[:8] = np.vstack((corners, [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
        shape = _shape(coordinate)
        physical = shape @ chart
        closed_shape, closed_coordinate, closed_valid = inverse_bilinear(chart, physical)
        old_shape, old_coordinate, old_valid = ShellDPRTransfer._inverse(chart, physical)
        valid_closed = bool(valid_closed and closed_valid.all())
        valid_newton = bool(valid_newton and old_valid.all())
        diameter = np.linalg.norm(chart - chart.mean(axis=0), axis=1).max()
        maximum_shape = max(maximum_shape, float(np.max(abs(closed_shape - old_shape))))
        maximum_coordinate = max(maximum_coordinate, float(np.max(abs(closed_coordinate - old_coordinate))))
        maximum_forward = max(maximum_forward, float(np.max(np.linalg.norm(closed_shape @ chart - physical, axis=1)) / diameter))
        maximum_old_forward = max(maximum_old_forward, float(np.max(np.linalg.norm(old_shape @ chart - physical, axis=1)) / diameter))
        frozen.update(chart.tobytes())
        frozen.update(coordinate.tobytes())
    threshold = 2e-12
    return dict(name="closed_inverse_vs_old_Newton_20000_points", passed=bool(valid_closed and valid_newton and minimum_oriented_area > 0
                and max(maximum_shape, maximum_coordinate, maximum_forward, maximum_old_forward) < threshold),
                seed=419703, charts=100, points_per_chart=200, queries=20000,
                input_sha256=frozen.hexdigest(), all_closed_valid=valid_closed, all_Newton_valid=valid_newton,
                maximum_shape_absolute_difference=maximum_shape,
                maximum_material_coordinate_absolute_difference=maximum_coordinate,
                maximum_forward_reconstruction_over_chart_radius=maximum_forward,
                maximum_Newton_forward_reconstruction_over_chart_radius=maximum_old_forward,
                minimum_convex_oriented_corner_area_m2=minimum_oriented_area, tolerance=threshold)


def _contact_at(model, face, coordinate, side, force=None):
    shape = _shape(coordinate)[0]
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    derivative_u = 0.25 * signs[:, 0] * (1 + signs[:, 1] * coordinate[1])
    derivative_v = 0.25 * signs[:, 1] * (1 + signs[:, 0] * coordinate[0])
    nodes = model.quads[face]
    physical = model.vertices[nodes] + side * 0.5 * model.h * model.directors[nodes]
    point = shape @ physical
    normal = side * np.cross(derivative_u @ physical, derivative_v @ physical)
    normal /= np.linalg.norm(normal)
    if force is None:
        tangent = derivative_u @ physical
        tangent /= np.linalg.norm(tangent)
        force = -0.05 * normal + 0.01 * tangent
    return Contact(point, force, normal, (50.0, 70.0), side=side, radius=None, friction=0.5, face_hint=face)


def _proxy_quadrature_controls(model, proxy_type):
    """Check clipped integration near fixed triangle and original material edges."""
    face = len(model.quads) // 3
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    edges = {}
    for i_face, nodes in enumerate(model.quads):
        for i_edge in range(4):
            key = tuple(sorted((int(nodes[i_edge]), int(nodes[(i_edge + 1) % 4]))))
            edges.setdefault(key, []).append((i_face, i_edge))
    shared_edge = next(entries for entries in edges.values() if len(entries) == 2 and entries[0][0] == face)
    contacts = []
    names = []
    for side in (-1, 1):
        fixed_force = _contact_at(model, face, [0.13, 0.13], side).force
        for signed in (-0.0001, 0.0, 0.0001):
            contacts.append(_contact_at(model, face, [0.13, 0.13 + signed], side, fixed_force))
            names.append(f"triangle_diagonal_s{side}_offset{signed:g}")
        first_face, first_edge = shared_edge[0]
        second_face, second_edge = shared_edge[1]
        first_midpoint = 0.5 * (corners[first_edge] + corners[(first_edge + 1) % 4])
        second_midpoint = 0.5 * (corners[second_edge] + corners[(second_edge + 1) % 4])
        source_first = _contact_at(model, first_face, first_midpoint, side)
        source_second = _contact_at(model, second_face, second_midpoint, side)
        average_normal = source_first.normal + source_second.normal
        average_normal /= np.linalg.norm(average_normal)
        tangent = model.vertices[model.quads[first_face][(first_edge + 1) % 4]] - model.vertices[model.quads[first_face][first_edge]]
        tangent /= np.linalg.norm(tangent)
        fixed_force = -0.05 * average_normal + 0.005 * tangent
        for signed in (-0.0005, 0.0, 0.0005):
            selected_face, midpoint = (first_face, first_midpoint) if signed <= 0 else (second_face, second_midpoint)
            contacts.append(_contact_at(model, selected_face, (1 - abs(signed)) * midpoint, side, fixed_force))
            names.append(f"material_edge_s{side}_offset{signed:g}")
    transfer = proxy_type(model, subdivisions=8)
    loads = []
    records = []
    for name, contact in zip(names, contacts):
        lower, lower_info = transfer.transfer(contact, order=24, analytic=True, anchor_mode="fixed")
        higher, higher_info = transfer.transfer(contact, order=48, analytic=True, anchor_mode="fixed")
        loads.extend((lower, higher))
        records.append(dict(name=name, load_relative=_relative(lower, higher), order24=lower_info, order48=higher_info))
    solution = model.solve(np.column_stack(loads))
    fields = model.stress(solution)
    for index, record in enumerate(records):
        low, high = 2 * index, 2 * index + 1
        energy = 0.0
        difference = 0.0
        for key in ("sigma_top", "sigma_bottom"):
            reference = fields[key][..., high]
            delta = fields[key][..., low] - reference
            energy += np.einsum("egij,egij,eg->", reference, reference, model.gauss_weights)
            difference += np.einsum("egij,egij,eg->", delta, delta, model.gauss_weights)
        record["full_tensor_relative"] = float(np.sqrt(difference / max(energy, 1e-300)))
        peaks = []
        for column in (low, high):
            local = np.concatenate((fields["local_top"][..., column], fields["local_bottom"][..., column]), axis=1)
            xx, yy, xy = np.moveaxis(local, -1, 0)
            peaks.append(np.sqrt(np.maximum(0, xx**2 + yy**2 - xx * yy + 3 * xy**2)).max())
        record["sampled_peak_relative"] = float(abs(peaks[0] - peaks[1]) / max(peaks[1], 1e-30))
    threshold = 1e-3
    maximum_load = max(record["load_relative"] for record in records)
    maximum_field = max(record["full_tensor_relative"] for record in records)
    maximum_peak = max(record["sampled_peak_relative"] for record in records)
    return dict(name="native_proxy_k8_order24_vs48_triangle_and_material_edges", passed=bool(max(maximum_load, maximum_field, maximum_peak) < threshold),
                contacts=len(contacts), proxy_subdivisions=8, integration_relative_tolerance=threshold,
                maximum_load_relative=maximum_load, maximum_field_relative=maximum_field,
                maximum_peak_relative=maximum_peak, rows=records,
                meaning="Integration uncertainty is separate from proxy-vs-Q4 spatial error.")


def run_checks(model, output_dir):
    """Save independent geometric, analytical and original-model solve controls."""
    rows = _work_controls(model)
    rows.extend(_affine_controls(TriangularLoadTransfer))
    rows.append(_tilted_proxy_control(TriangularLoadTransfer))
    rows.append(_warped_control(TriangularLoadTransfer))
    rows.append(_fixed_stiffness_controls(model, TriangularLoadTransfer))
    rows.append(_inverse_controls())
    rows.append(_proxy_quadrature_controls(model, TriangularLoadTransfer))
    result = dict(schema="triangular_load_cpu.independent_checks.v1", passed=all(row["passed"] for row in rows),
                  checks=len(rows), rows=rows, scope="CPU load approximation controls on unchanged Q4 elasticity.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "checks.json").write_text(json.dumps(result, indent=2))
    return result
