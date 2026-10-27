"""Transfer spherical Hertz loads through fixed planar triangle proxies.

Each proxy vertex samples the original physical Q4 side and its displacement
map. Transposed interpolation transfers triangle forces to the original shell
translation and rotation coordinates, preserving virtual work for the proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial import cKDTree

from research.rigid_stress.shell_cpu.contacts import Contact, ShellDPRTransfer, _basis, _shape, hertz_radius
from research.rigid_stress.wrench import PadPressureFit, WrenchPatch


@dataclass(frozen=True)
class TriangleSurface:
    vertices: np.ndarray
    parent_shapes: np.ndarray
    parent_faces: np.ndarray
    normals: np.ndarray
    dual: np.ndarray
    centers: np.ndarray
    extents: np.ndarray
    maximum_extent: float
    tree: cKDTree


def _triangle_surface(model, subdivisions: int, side: int) -> TriangleSurface:
    coordinates = np.linspace(-1.0, 1.0, subdivisions + 1)
    triangles = []
    for j in range(subdivisions):
        for i in range(subdivisions):
            corners = np.array(
                [
                    [coordinates[i], coordinates[j]],
                    [coordinates[i + 1], coordinates[j]],
                    [coordinates[i + 1], coordinates[j + 1]],
                    [coordinates[i], coordinates[j + 1]],
                ]
            )
            triangles.extend((corners[[0, 1, 2]], corners[[0, 2, 3]]))
    local_triangles = np.stack(triangles)
    shapes, _, _ = _shape(local_triangles[..., 0], local_triangles[..., 1])
    if side < 0:
        shapes = shapes[:, [0, 2, 1]]
    physical_vertices = model.vertices + side * 0.5 * model.h * model.directors
    quad_vertices = physical_vertices[model.quads]
    vertices = np.einsum("tvi,fij->ftvj", shapes, quad_vertices).reshape((-1, 3, 3))
    parent_shapes = np.broadcast_to(shapes[None], (len(model.quads),) + shapes.shape).reshape((-1, 3, 4)).copy()
    parent_faces = np.repeat(np.arange(len(model.quads)), len(shapes))
    edge_first, edge_second = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    normal = np.cross(edge_first, edge_second)
    areas = np.linalg.norm(normal, axis=1)
    if np.any(areas <= 1e-24):
        raise ValueError("The triangle loading proxy contains a degenerate face.")
    normal /= areas[:, None]
    edges = np.stack((edge_first, edge_second), axis=1)
    gram = edges @ edges.transpose((0, 2, 1))
    dual = np.linalg.solve(gram, edges)
    centers = vertices.mean(axis=1)
    extents = np.linalg.norm(vertices - centers[:, None], axis=2).max(axis=1)
    return TriangleSurface(vertices, parent_shapes, parent_faces, normal, dual, centers, extents,
                           float(extents.max()), cKDTree(centers))


def _disk_intersects_triangle(polygon: np.ndarray, radius: float) -> bool:
    edges = np.roll(polygon, -1, axis=0) - polygon
    first, second = polygon[1] - polygon[0], polygon[2] - polygon[0]
    orientation = first[0] * second[1] - first[1] * second[0]
    signs = edges[:, 0] * (-polygon[:, 1]) - edges[:, 1] * (-polygon[:, 0])
    is_inside = np.all(signs >= -1e-28) if orientation > 0 else np.all(signs <= 1e-28)
    if is_inside:
        return True
    parameters = np.clip(-np.einsum("ij,ij->i", polygon, edges) / np.einsum("ij,ij->i", edges, edges), 0.0, 1.0)
    nearest = polygon + parameters[:, None] * edges
    return bool(np.min(np.einsum("ij,ij->i", nearest, nearest)) < radius * radius)


def _pressure_frame(force: np.ndarray) -> np.ndarray:
    direction = force / np.linalg.norm(force)
    tangent = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
    tangent /= np.linalg.norm(tangent)
    return np.column_stack((tangent, np.cross(direction, tangent)))


class TriangularLoadTransfer:
    """Apply native-law Hertz pressure on a fixed faceted physical surface.

    ``anchor_mode='fixed'`` keeps the supplied surface-ball center unchanged.
    ``'force_line'`` explicitly reanchors that center to the proxy along the
    complete force line, retaining the original force and point wrench.
    """

    def __init__(self, model, subdivisions: int = 1):
        if not isinstance(subdivisions, int) or subdivisions < 1:
            raise ValueError("subdivisions must be a positive integer.")
        started = time.perf_counter()
        self.model = model
        self.subdivisions = subdivisions
        self.outer = _triangle_surface(model, subdivisions, 1)
        self.inner = _triangle_surface(model, subdivisions, -1)
        adjacency = [set() for _ in model.quads]
        edge_faces = {}
        for face, quad in enumerate(model.quads):
            for i in range(4):
                edge = tuple(sorted((int(quad[i]), int(quad[(i + 1) % 4]))))
                edge_faces.setdefault(edge, []).append(face)
        for faces in edge_faces.values():
            for face in faces:
                adjacency[face].update(set(faces) - {face})
        self.adjacency = tuple(frozenset(faces) for faces in adjacency)
        self.preparation_seconds = time.perf_counter() - started
        self.array_bytes = sum(
            array.nbytes
            for surface in (self.outer, self.inner)
            for array in (
                surface.vertices, surface.parent_shapes, surface.parent_faces, surface.normals,
                surface.dual, surface.centers, surface.extents,
            )
        )

    def _force_line_anchor(self, surface: TriangleSurface, point: np.ndarray, direction: np.ndarray, face_hint):
        denominator = surface.normals @ direction
        eligible = denominator < -1e-10
        if face_hint is not None:
            hint = surface.parent_faces == face_hint
            if np.any(eligible & hint):
                eligible &= hint
        indices = np.flatnonzero(eligible)
        parameter = np.einsum("ij,ij->i", surface.normals[indices], surface.vertices[indices, 0] - point)
        parameter /= denominator[indices]
        hits = point + parameter[:, None] * direction
        local = np.einsum("tij,tj->ti", surface.dual[indices], hits - surface.vertices[indices, 0])
        inside = (local.min(axis=1) >= -2e-11) & (local.sum(axis=1) <= 1 + 2e-11)
        if not np.any(inside):
            raise ValueError("The original force line misses the selected triangle loading surface.")
        chosen = np.flatnonzero(inside)[np.argmin(np.abs(parameter[inside]))]
        return hits[chosen], int(indices[chosen])

    def _sections(self, surface: TriangleSurface, center: np.ndarray, radius: float, face_hint):
        candidates = surface.tree.query_ball_point(center, radius + surface.maximum_extent + 1e-12, return_sorted=True)
        sections = []
        for triangle in candidates:
            normal = surface.normals[triangle]
            height = (center - surface.vertices[triangle, 0]) @ normal
            remaining = radius * radius - height * height
            if remaining <= 0.0:
                continue
            section_center = center - height * normal
            plane = _basis(normal)
            polygon = (surface.vertices[triangle] - section_center) @ plane
            section_radius = np.sqrt(remaining)
            if _disk_intersects_triangle(polygon, section_radius):
                sections.append((triangle, section_center, plane, polygon, section_radius))
        if not sections:
            raise ValueError("The fixed surface-ball footprint misses the triangle proxy.")
        # Restrict a spatial query to one connected material patch.
        parents = {int(surface.parent_faces[section[0]]) for section in sections}
        if face_hint is not None and face_hint in parents:
            seed = int(face_hint)
        else:
            nearest = min(sections, key=lambda section: np.linalg.norm(section[1] - center))
            seed = int(surface.parent_faces[nearest[0]])
        selected, todo = {seed}, [seed]
        while todo:
            face = todo.pop()
            for other in self.adjacency[face] & parents - selected:
                selected.add(other)
                todo.append(other)
        sections = [section for section in sections if int(surface.parent_faces[section[0]]) in selected]
        return sections, len(candidates)

    def _scatter(self, surface, triangle_indices, barycentric, sample_force, side):
        parent_shapes = np.einsum("qi,qij->qj", barycentric, surface.parent_shapes[triangle_indices])
        parent_faces = surface.parent_faces[triangle_indices]
        load = np.zeros((len(self.model.vertices), 6))
        for face in np.unique(parent_faces):
            selection = parent_faces == face
            nodal = parent_shapes[selection].T @ sample_force[selection]
            nodes = self.model.quads[face]
            np.add.at(load[:, :3], nodes, nodal)
            np.add.at(load[:, 3:], nodes, side * 0.5 * self.model.h * np.cross(self.model.directors[nodes], nodal))
        return load

    def _analytic(self, surface, section, point, force, radius, side):
        triangle, center, plane, polygon, section_radius = section
        if not ShellDPRTransfer._full_disk(polygon, section_radius):
            return None
        frame = _pressure_frame(force)
        mean = np.r_[1.0, (center - point) @ frame / radius]
        tangent = np.eye(3) - np.outer(surface.normals[triangle], surface.normals[triangle])
        gram = np.outer(mean, mean)
        gram[1:, 1:] += section_radius**2 / (5 * radius**2) * (frame.T @ tangent @ frame)
        eigenvalues = np.linalg.eigvalsh(gram)
        if eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
            return None
        coefficient = np.linalg.solve(gram, np.array([1.0, 0.0, 0.0]))
        minimum = mean @ coefficient - section_radius / radius * np.linalg.norm(tangent @ (frame @ coefficient[1:]))
        if minimum < -2e-12:
            return None
        local = surface.dual[triangle] @ (center - surface.vertices[triangle, 0])
        bary = np.r_[1 - local.sum(), local]
        gradient = np.vstack((-surface.dual[triangle].sum(axis=0), surface.dual[triangle]))
        integrated_bary = bary * (mean @ coefficient)
        integrated_bary += section_radius**2 / (5 * radius) * (gradient @ tangent @ (frame @ coefficient[1:]))
        triangle_force = integrated_bary[:, None] * force
        load = self._scatter(surface, np.full(3, triangle), np.eye(3), triangle_force, side)
        return load, minimum

    def transfer(self, contact: Contact, order: int = 24, analytic: bool = True, anchor_mode: str = "fixed"):
        started = time.perf_counter()
        point = np.asarray(contact.point, dtype=float)
        force = np.asarray(contact.force, dtype=float)
        normal = np.asarray(contact.normal, dtype=float)
        magnitude = np.linalg.norm(force)
        if point.shape != (3,) or force.shape != (3,) or normal.shape != (3,) or not np.isfinite(np.r_[point, force, normal]).all():
            raise ValueError("Contact vectors must contain three finite components.")
        if contact.side not in (-1, 1) or np.linalg.norm(normal) == 0.0:
            raise ValueError("Contact side and outward normal must be valid.")
        normal = normal / np.linalg.norm(normal)
        compression = -force @ normal
        if magnitude == 0.0:
            return np.zeros(self.model.ndof), {"valid": True, "radius_m": 0.0, "method": "unloaded", "samples": 0,
                "n_candidates": 0, "n_triangles": 0, "analytic_count": 0, "force_error": 0.0, "moment_error": 0.0,
                "total_seconds": time.perf_counter() - started}
        friction = 1.0 if contact.friction is None else contact.friction
        if compression <= 0 or np.linalg.norm(force + compression * normal) > friction * compression * (1 + 1e-12):
            raise ValueError("The complete contact force violates its supplied normal/friction cone.")
        radius = contact.radius
        if radius is None:
            radius = hertz_radius(compression, self.model.E, self.model.nu, contact.curvatures)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("A finite positive contact radius is required.")
        surface = self.outer if contact.side > 0 else self.inner
        if anchor_mode == "fixed":
            center = point.copy()
        elif anchor_mode in ("force_line", "reanchor"):
            center, _ = self._force_line_anchor(surface, point, force / magnitude, contact.face_hint)
        else:
            raise ValueError("anchor_mode must be 'fixed' or 'force_line'.")
        sections, candidates = self._sections(surface, center, radius, contact.face_hint)
        queried = time.perf_counter()
        result = self._analytic(surface, sections[0], point, force, radius, contact.side) if analytic and len(sections) == 1 else None
        analytic_count, samples, evaluations, minimum_pressure = 0, 0, 0, None
        if result is not None:
            load, minimum_pressure = result
            analytic_count, method = 1, "analytic_contained_triangle"
            integrated = time.perf_counter()
            fitted = integrated
        else:
            positions, weights, barycentric, indices = [], [], [], []
            for triangle, section_center, plane, polygon, section_radius in sections:
                points, local_weights = ShellDPRTransfer._face_polar_samples(polygon, section_radius, order)
                if not len(local_weights):
                    continue
                physical = section_center + points @ plane.T
                local = (physical - surface.vertices[triangle, 0]) @ surface.dual[triangle].T
                bary = np.column_stack((1 - local.sum(axis=1), local))
                if np.min(bary) < -2e-8:
                    raise ValueError("Triangle pressure quadrature lies outside its material face.")
                positions.append(physical)
                weights.append(local_weights * (2 * np.pi * section_radius**3 / (3 * radius)))
                barycentric.append(bary)
                indices.append(np.full(len(local_weights), triangle))
            if not weights:
                raise ValueError("The triangle loading proxy has no resolved finite footprint.")
            positions = np.vstack(positions)
            weights = np.concatenate(weights)
            barycentric = np.vstack(barycentric)
            indices = np.concatenate(indices)
            samples = len(weights)
            integrated = time.perf_counter()
            patch = WrenchPatch(point, force, radius, friction, np.zeros(3), -normal)
            sample_force, diagnostics = PadPressureFit(positions, weights, patch).solve()
            evaluations = diagnostics.constrained_evaluations
            fitted = time.perf_counter()
            load = self._scatter(surface, indices, barycentric, sample_force, contact.side)
            method = "clipped_triangle_quadrature"
        scattered = time.perf_counter()
        total = load[:, :3].sum(axis=0)
        moment = np.cross(self.model.vertices, load[:, :3]).sum(axis=0) + load[:, 3:].sum(axis=0)
        force_error = np.linalg.norm(total - force) / magnitude
        moment_error = np.linalg.norm(moment - np.cross(point, force)) / (magnitude * radius)
        if force_error > 1e-8 or moment_error > 1e-8:
            raise ValueError(f"Triangle proxy scatter violates the original wrench: force={force_error:g}, moment={moment_error:g}.")
        info = {
            "valid": True, "radius_m": float(radius), "method": method, "samples": samples,
            "n_candidates": candidates, "n_triangles": len(sections), "analytic_count": analytic_count,
            "pressure_fit_evaluations": evaluations, "minimum_affine_pressure": minimum_pressure,
            "force_error": float(force_error), "moment_error": float(moment_error),
            "original_center_m": point.tolist(), "support_center_m": center.tolist(),
            "anchor_mode": anchor_mode, "anchor_shift_m": float(np.linalg.norm(center - point)),
            "anchor_shift_over_radius": float(np.linalg.norm(center - point) / radius),
            "query_seconds": queried - started, "integration_seconds": integrated - queried,
            "fit_seconds": fitted - integrated, "scatter_seconds": scattered - fitted,
            "total_seconds": time.perf_counter() - started,
        }
        return load.reshape(-1), info
