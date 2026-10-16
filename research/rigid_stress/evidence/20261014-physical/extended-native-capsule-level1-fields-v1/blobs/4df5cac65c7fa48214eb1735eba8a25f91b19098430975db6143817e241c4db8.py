"""Measure egg CAD approximation independently of elasticity and fixed-polyhedron contact refinement."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .audit_franka_reference import rotation
from .summarize_fine_references import closed_source


@dataclass(frozen=True)
class EggGeometry:
    equatorial_radius_m: float = 0.022
    polar_radius_m: float = 0.030
    taper: float = 0.18
    thickness_m: float = 0.0005
    density_kg_m3: float = 2000


def meridian(geometry, parameter, depth):
    a, c, k = geometry.equatorial_radius_m, geometry.polar_radius_m, geometry.taper
    scale = 1 - k * parameter
    sine_squared = 1 - parameter**2
    w = -k - parameter + 2 * k * parameter**2
    q = -1 + 3 * k * parameter - 2 * k * parameter**3
    denominator = numpy.sqrt(a * a * w * w + c * c * sine_squared)
    radial_normal = c * numpy.sqrt(sine_squared) / denominator
    vertical_normal = -a * w / denominator
    radius = numpy.sqrt(sine_squared) * (a * scale - depth * c / denominator)
    height = c * parameter - depth * vertical_normal
    jacobian = (a * scale - depth * c / denominator) * (denominator + depth * c * a * q / denominator**2)
    height_derivative = c + depth * c * c * a * q / denominator**3
    return radius, height, radial_normal, vertical_normal, jacobian, height_derivative


def continuum_integrals(geometry, order, depth_order):
    parameter, weights = numpy.polynomial.legendre.leggauss(order)
    depth, depth_weights = numpy.polynomial.legendre.leggauss(depth_order)
    depth = (depth + 1) * geometry.thickness_m / 2
    depth_weights *= geometry.thickness_m / 2
    radius, height, _, _, jacobian, _ = meridian(geometry, parameter[:, None], depth[None])
    assert numpy.all(jacobian > 0) and numpy.all(radius > 0)
    volume_weights = 2 * numpy.pi * weights[:, None] * depth_weights[None] * jacobian
    volume = volume_weights.sum()
    com_height = (volume_weights * height).sum() / volume
    radial_second = (volume_weights * radius**2).sum()
    vertical_second = (volume_weights * (height - com_height) ** 2).sum()
    inertia = geometry.density_kg_m3 * numpy.array(
        [radial_second / 2 + vertical_second, radial_second / 2 + vertical_second, radial_second]
    )
    outer = meridian(geometry, parameter, 0)
    inner = meridian(geometry, parameter, geometry.thickness_m)
    outer_volume = numpy.pi * numpy.sum(weights * outer[0] ** 2 * outer[5])
    inner_volume = numpy.pi * numpy.sum(weights * inner[0] ** 2 * inner[5])
    a, c, k = geometry.equatorial_radius_m, geometry.polar_radius_m, geometry.taper
    exact_outer_volume = numpy.pi * a * a * c * (4 / 3 + 4 * k * k / 15)
    numpy.testing.assert_allclose(outer_volume, exact_outer_volume, rtol=1e-12)
    numpy.testing.assert_allclose(volume, outer_volume - inner_volume, rtol=1e-11)
    disk_first = numpy.pi * numpy.sum(
        weights * (outer[1] * outer[0] ** 2 * outer[5] - inner[1] * inner[0] ** 2 * inner[5])
    )
    numpy.testing.assert_allclose(com_height, disk_first / volume, rtol=1e-10, atol=1e-15)
    return {
        "order": order,
        "depth_order": depth_order,
        "volume_m3": float(volume),
        "mass_kg": float(geometry.density_kg_m3 * volume),
        "com_m": [0.0, 0.0, float(com_height)],
        "inertia_diagonal_about_com_kg_m2": inertia.tolist(),
        "outer_area_m2": float(2 * numpy.pi * numpy.sum(weights * outer[4])),
        "inner_area_m2": float(2 * numpy.pi * numpy.sum(weights * inner[4])),
        "independent_disk_difference_volume_m3": float(outer_volume - inner_volume),
    }


def positions(geometry, parameter, angle, depth):
    radius, height, _, _, _, _ = meridian(geometry, parameter, depth)
    return numpy.column_stack((radius * numpy.cos(angle), radius * numpy.sin(angle), height))


def qualify_parameterization(geometry):
    sphere = EggGeometry(0.02, 0.02, 0, geometry.thickness_m, geometry.density_kg_m3)
    result = continuum_integrals(sphere, 64, 4)
    outer, inner = sphere.equatorial_radius_m, sphere.equatorial_radius_m - sphere.thickness_m
    volume = 4 * numpy.pi / 3 * (outer**3 - inner**3)
    inertia = 8 * numpy.pi / 15 * sphere.density_kg_m3 * (outer**5 - inner**5)
    numpy.testing.assert_allclose(result["volume_m3"], volume, rtol=1e-12)
    numpy.testing.assert_allclose(result["inertia_diagonal_about_com_kg_m2"], inertia, rtol=1e-12)
    numpy.testing.assert_allclose(result["com_m"], 0, rtol=0, atol=1e-15)
    parameter = numpy.linspace(-0.95, 0.95, 21)
    depth = numpy.linspace(0.03, 0.97, 21) * geometry.thickness_m
    angle = numpy.linspace(0.2, 5.7, 21)

    step = 1e-5
    derivative_s = (
        positions(geometry, parameter + step, angle, depth) - positions(geometry, parameter - step, angle, depth)
    ) / (2 * step)
    derivative_angle = (
        positions(geometry, parameter, angle + step, depth) - positions(geometry, parameter, angle - step, depth)
    ) / (2 * step)
    depth_step = 1e-7
    derivative_depth = (
        positions(geometry, parameter, angle, depth + depth_step)
        - positions(geometry, parameter, angle, depth - depth_step)
    ) / (2 * depth_step)
    numeric = abs(numpy.einsum("ni,ni->n", numpy.cross(derivative_s, derivative_angle), derivative_depth))
    analytic = meridian(geometry, parameter, depth)[4]
    error = numpy.max(abs(numeric / analytic - 1))
    assert error <= 1e-7
    return {
        "analytic_sphere_volume_and_inertia_passed": True,
        "maximum_finite_difference_parameter_jacobian_relative_error": float(error),
    }


def mesh_integrals(vertices, tetrahedra, geometry, continuum):
    corners = vertices[tetrahedra]
    volumes = abs(numpy.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    assert numpy.all(volumes > 0)
    volume = math.fsum(volumes)
    com = numpy.einsum("n,ni->i", volumes, corners.mean(axis=1)) / volume
    summed = corners.sum(axis=1)
    second = numpy.einsum("n,nvi,nvj->ij", volumes / 20, corners, corners)
    second += numpy.einsum("n,ni,nj->ij", volumes / 20, summed, summed)
    second -= volume * numpy.outer(com, com)
    inertia = geometry.density_kg_m3 * (numpy.trace(second) * numpy.eye(3) - second)
    faces = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    facets, first, counts = numpy.unique(
        numpy.sort(tetrahedra[:, faces].reshape(-1, 3)), axis=0, return_index=True, return_counts=True
    )
    assert counts.max() == 2
    boundary, owners = facets[counts == 1], first[counts == 1] // 4
    used, inverse = numpy.unique(boundary, return_inverse=True)
    local_faces = inverse.reshape(-1, 3)
    edges = local_faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)
    graph = sparse.coo_matrix((numpy.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(used),) * 2).tocsr()
    components, labels = connected_components(graph, directed=False)
    assert components == 2 and numpy.all(labels[local_faces] == labels[local_faces[:, :1]])
    triangles = vertices[boundary]
    normals = numpy.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centroid = triangles.mean(axis=1)
    signs = numpy.sign(numpy.einsum("ni,ni->n", normals, centroid - corners[owners].mean(axis=1)))
    assert numpy.all(signs != 0)
    signed = numpy.einsum("ni,ni->n", triangles[:, 0], numpy.cross(triangles[:, 1], triangles[:, 2])) * signs / 6
    component_volume = numpy.bincount(labels[local_faces[:, 0]], weights=signed)
    assert numpy.count_nonzero(component_volume > 0) == numpy.count_nonzero(component_volume < 0) == 1
    outer = component_volume[labels[local_faces[:, 0]]] > 0
    numpy.testing.assert_allclose(math.fsum(signed), volume, rtol=1e-12)
    areas = numpy.linalg.norm(normals, axis=1) / 2
    parameter = centroid[outer, 2] / geometry.polar_radius_m
    assert numpy.all(abs(parameter) < 1)
    analytic = meridian(geometry, parameter, 0)
    radial_deficit = analytic[0] - numpy.linalg.norm(centroid[outer, :2], axis=1)
    radial_direction = centroid[outer, :2] / numpy.linalg.norm(centroid[outer, :2], axis=1)[:, None]
    analytic_normal = numpy.column_stack((analytic[2][:, None] * radial_direction, analytic[3]))
    outward = signs[outer, None] * normals[outer] / (2 * areas[outer, None])
    angles = numpy.arccos(numpy.clip(numpy.einsum("ni,ni->n", outward, analytic_normal), -1, 1)) * 180 / numpy.pi
    return {
        "tetrahedra": len(tetrahedra),
        "vertices": len(vertices),
        "boundary_components": components,
        "volume_m3": volume,
        "mass_kg": volume * geometry.density_kg_m3,
        "com_m": com.tolist(),
        "inertia_about_com_kg_m2": inertia.tolist(),
        "relative_volume_and_mass_error_from_continuum": volume / continuum["volume_m3"] - 1,
        "outer_area_m2": float(areas[outer].sum()),
        "inner_area_m2": float(areas[~outer].sum()),
        "outer_facet_centroid_radial_deficit_at_same_z_m_percentiles": numpy.percentile(
            radial_deficit, [0, 50, 90, 99, 100]
        ).tolist(),
        "outer_facet_centroid_normal_angle_at_same_z_degrees_percentiles": numpy.percentile(
            angles, [0, 50, 90, 99, 100]
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geometry = EggGeometry()
    qualification = qualify_parameterization(geometry)
    continuum = [continuum_integrals(geometry, order, 4) for order in [64, 128, 256]]
    depth_check = continuum_integrals(geometry, 256, 8)
    for candidate in [continuum[-2], depth_check]:
        for key in ["volume_m3", "inertia_diagonal_about_com_kg_m2", "outer_area_m2", "inner_area_m2"]:
            numpy.testing.assert_allclose(candidate[key], continuum[-1][key], rtol=1e-10)
        numpy.testing.assert_allclose(candidate["com_m"], continuum[-1]["com_m"], rtol=0, atol=1e-14)
    specs = json.loads(args.mesh_spec.read_text())
    inputs = {str(args.mesh_spec): hashlib.sha256(args.mesh_spec.read_bytes()).hexdigest()}
    rows = []
    for spec in specs["meshes"]:
        path = Path(spec["path"])
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if "source_run" in spec:
            source = Path(spec["source_run"])
            closed_source(source)
            assert int((source / "returncode.txt").read_text()) == 0
        with numpy.load(path) as saved:
            vertices, tetrahedra = saved["vertices"].copy(), saved["tetrahedra"].copy()
        if "capture" in spec:
            capture = Path(spec["capture"])
            closed_source(capture)
            report_path = capture / "results/summary.json"
            inputs[str(report_path)] = hashlib.sha256(report_path.read_bytes()).hexdigest()
            report = json.loads(report_path.read_text())
            frame = [r for r in report["captures"] if r["mode"] == "stationary_lift_hold"][-1]
            matrix = rotation(frame["egg_quaternion_wxyz"])
            shift = numpy.array(frame["egg_position_m"]) - frame["egg_world_com_m"]
            vertices = (vertices - shift) @ matrix
        if spec.get("layered_analytic_vertices", False):
            layers = spec["layers"]
            assert len(vertices) % (layers + 1) == 0
            count = len(vertices) // (layers + 1)
            outer = vertices[:count]
            parameter = outer[:, 2] / geometry.polar_radius_m
            radius, height, radial_normal, vertical_normal, _, _ = meridian(geometry, parameter, 0)
            phi = numpy.arctan2(outer[:, 1], outer[:, 0])
            expected_outer = numpy.column_stack((radius * numpy.cos(phi), radius * numpy.sin(phi), height))
            normal = numpy.column_stack(
                (radial_normal * numpy.cos(phi), radial_normal * numpy.sin(phi), vertical_normal)
            )
            numpy.testing.assert_allclose(outer, expected_outer, rtol=0, atol=1e-14)
            for layer in range(layers + 1):
                expected = outer - layer / layers * geometry.thickness_m * normal
                numpy.testing.assert_allclose(
                    vertices[layer * count : (layer + 1) * count], expected, rtol=0, atol=1e-14
                )
        row = {
            "name": spec["name"],
            "mesh_sha256": inputs[str(path)],
            **mesh_integrals(vertices, tetrahedra, geometry, continuum[-1]),
        }
        if "metadata" in spec:
            metadata = Path(spec["metadata"])
            inputs[str(metadata)] = hashlib.sha256(metadata.read_bytes()).hexdigest()
            asset = json.loads(metadata.read_text())
            numpy.testing.assert_allclose(row["mass_kg"], asset["mass_kg"], rtol=1e-12)
            numpy.testing.assert_allclose(row["com_m"], asset["com_m"], rtol=0, atol=1e-14)
            numpy.testing.assert_allclose(
                row["inertia_about_com_kg_m2"], asset["inertia_kg_m2"], rtol=1e-10, atol=1e-18
            )
        rows.append(row)
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    result = {
        "scope": __doc__,
        "declared_continuum_geometry": asdict(geometry),
        "parameterization_qualification": qualification,
        "continuum_quadrature_levels": continuum,
        "continuum_depth_quadrature_check": depth_check,
        "rows": rows,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Geometry-only measurements keep every actual source mesh and its material/density unchanged. "
        "Radial deficits compare facet centroids with the analytic surface at the same height; "
        "they are not normal-distance or Hausdorff bounds. Volume/mass differences do not "
        "attribute a fraction of stress error or prove nonlinear contact singularity. Fixed-polyhedron "
        "local refinement and CAD approximation remain distinct. Stress/contact/feedback and "
        "physically verified accuracy/cost acceptance remain required.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"continuum": continuum[-1], "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
