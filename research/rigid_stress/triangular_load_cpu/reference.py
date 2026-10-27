"""Integrate surface-ball Hertz pressure on the unchanged physical Q4 shell."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from research.rigid_stress.shell_cpu.contacts import Contact, ShellDPRTransfer, _basis, _shape, hertz_radius
from research.rigid_stress.wrench import PadPressureFit, WrenchPatch


def cross2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the signed two-dimensional cross product."""
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def inverse_bilinear(vertices: np.ndarray, points: np.ndarray):
    """Invert a nonfolded bilinear chart with a quadratic solve and one correction."""
    targets = np.atleast_2d(points)
    origin = vertices.mean(axis=0)
    first = (-vertices[0] + vertices[1] + vertices[2] - vertices[3]) / 4
    second = (-vertices[0] - vertices[1] + vertices[2] + vertices[3]) / 4
    mixed = (vertices[0] - vertices[1] + vertices[2] - vertices[3]) / 4
    relative = targets - origin
    coefficient_a = -cross2(first, mixed)
    coefficient_b = cross2(relative, mixed) - cross2(first, second)
    coefficient_c = cross2(relative, second)
    count = len(targets)
    scale = max(np.linalg.norm(vertices - origin, axis=1).max(), 1e-15)
    roots = np.empty((count, 2))
    if abs(coefficient_a) <= 1e-14 * scale**2:
        denominator = np.where(abs(coefficient_b) > 1e-30, coefficient_b, np.nan)
        roots[:, 0] = -coefficient_c / denominator
        roots[:, 1] = roots[:, 0]
    else:
        discriminant = coefficient_b**2 - 4 * coefficient_a * coefficient_c
        root_term = np.sqrt(np.maximum(discriminant, 0))
        stable = -0.5 * (coefficient_b + np.where(coefficient_b >= 0, root_term, -root_term))
        roots[:, 0] = stable / coefficient_a
        roots[:, 1] = np.divide(coefficient_c, stable, out=np.zeros(count), where=abs(stable) > 1e-30)
        roots[discriminant < -1e-14 * scale**4] = np.nan
    side = second[None, None, :] + roots[..., None] * mixed
    coordinate_v = np.divide(
        np.einsum('nri,nri->nr', relative[:, None, :] - roots[..., None] * first, side),
        np.einsum('nri,nri->nr', side, side),
        out=np.full_like(roots, np.nan),
        where=np.einsum('nri,nri->nr', side, side) > 1e-30,
    )
    reconstructed = origin + roots[..., None] * first + coordinate_v[..., None] * second
    reconstructed += (roots * coordinate_v)[..., None] * mixed
    errors = np.linalg.norm(reconstructed - targets[:, None, :], axis=-1)
    is_inside = (abs(roots) <= 1 + 1e-8) & (abs(coordinate_v) <= 1 + 1e-8)
    scores = np.where(np.isfinite(errors), errors / scale + np.where(is_inside, 0, 10), np.inf)
    choice = np.argmin(scores, axis=1)
    coordinate_u = roots[np.arange(count), choice]
    coordinate_v = coordinate_v[np.arange(count), choice]
    shape, derivative_u, derivative_v = _shape(coordinate_u, coordinate_v)
    residual = shape @ (vertices - origin) - relative
    jacobian_u, jacobian_v = derivative_u @ vertices, derivative_v @ vertices
    determinant = cross2(jacobian_u, jacobian_v)
    valid_jacobian = abs(determinant) > 1e-30
    correction_u = np.divide(cross2(residual, jacobian_v), determinant, out=np.zeros(count), where=valid_jacobian)
    correction_v = np.divide(cross2(jacobian_u, residual), determinant, out=np.zeros(count), where=valid_jacobian)
    coordinate_u -= correction_u
    coordinate_v -= correction_v
    shape = _shape(coordinate_u, coordinate_v)[0]
    error = np.linalg.norm(shape @ (vertices - origin) - relative, axis=1)
    valid = valid_jacobian & np.isfinite(coordinate_u + coordinate_v)
    valid &= (abs(coordinate_u) <= 1 + 1e-9) & (abs(coordinate_v) <= 1 + 1e-9)
    valid &= error < 2e-11 * scale
    return shape, np.column_stack((coordinate_u, coordinate_v)), valid


class Q4LoadTransfer:
    """Build current finite-pressure loads on the original inner or outer shell surface."""

    def __init__(self, model, inverse_method='closed'):
        if inverse_method not in ('closed', 'newton'):
            raise ValueError('inverse_method must be closed or newton')
        self.model = model
        self.surface = ShellDPRTransfer(model)
        self.inverse_method = inverse_method
        self.inverse = inverse_bilinear if inverse_method == 'closed' else self.surface._inverse

    def transfer(self, contact: Contact, order=24):
        started = perf_counter()
        model = self.model
        point, force = np.asarray(contact.point), np.asarray(contact.force)
        magnitude = np.linalg.norm(force)
        if contact.side not in (-1, 1):
            raise ValueError('Contact side must be -1 or +1')
        if magnitude == 0:
            return np.zeros(model.ndof), dict(valid=True, radius_m=0., samples=0, n_samples=0,
                force_error=0., moment_error=0., candidate_faces=0, hit_faces=0,
                analytic_count=0, base_mass=0., inverse_method=self.inverse_method,
                pressure_fit_evaluations=0, stages=dict(query=0.,integration=0.,fit=0.,scatter=0.,total=perf_counter()-started))
        normal = np.asarray(contact.normal)
        if not np.isfinite(np.concatenate((point,force,normal))).all() or np.linalg.norm(normal) <= 0:
            raise ValueError('Contact vectors must be finite and normal must be nonzero')
        normal = normal / np.linalg.norm(normal)
        radius = contact.radius
        if radius is None:
            radius = hertz_radius(-force @ normal, model.E, model.nu, contact.curvatures)
        if not radius > 0 or not np.isfinite(radius):
            raise ValueError('Require a finite positive contact radius')
        face_points = self.surface.faces[contact.side][0]
        plane = _basis(normal)
        candidates = self.surface._candidates(point, radius, contact.side)
        seed = None
        # The supplied current face witness restricts topology, never the footprint extent.
        if contact.face_hint is not None:
            hint = int(contact.face_hint)
            if hint < 0 or hint >= len(model.quads):
                raise ValueError('Invalid material face hint')
            chart = (face_points[hint] - point) @ plane
            shape, coordinate, valid = self.inverse(chart, [[0., 0.]])
            if not valid[0] or np.linalg.norm(shape[0] @ face_points[hint] - point) > 2e-10:
                raise ValueError('Supplied face witness does not contain the source center')
            seed = hint
        else:
            distances = []
            for face in candidates:
                chart = (face_points[face] - point) @ plane
                shape, _, valid = self.inverse(chart, [[0., 0.]])
                if valid[0]:
                    distance = np.linalg.norm(shape[0] @ face_points[face] - point)
                    distances.append((distance, int(face)))
            if distances:
                distance, seed = min(distances)
                if distance > 2e-10:
                    raise ValueError('Source center is off the physical Q4 surface')
        if seed is None:
            raise ValueError('No material Q4 face contains the source center')
        candidate_set = set(map(int, candidates))
        candidate_set.add(seed)
        component, frontier = {seed}, [seed]
        while frontier:
            face = frontier.pop()
            for neighbor in self.surface.adjacent[face] & candidate_set - component:
                component.add(neighbor)
                frontier.append(neighbor)
        queried = perf_counter()
        positions, base_weights, shapes, parent_faces = [], [], [], []
        for face in sorted(component):
            physical_vertices = face_points[face]
            chart = (physical_vertices - point) @ plane
            samples, projected_weights = self.surface._face_polar_samples(chart, radius, order)
            if len(projected_weights) == 0:
                continue
            shape, coordinate, valid = self.inverse(chart, samples)
            if not valid.all():
                raise ValueError('Q4 sample inversion failed inside the finite footprint')
            physical = shape @ physical_vertices
            squared_planar = np.einsum('ij,ij->i', samples, samples) / radius**2
            squared_physical = np.einsum('ij,ij->i', physical-point, physical-point) / radius**2
            _, derivative_u, derivative_v = _shape(coordinate[:, 0], coordinate[:, 1])
            physical_u, physical_v = derivative_u @ physical_vertices, derivative_v @ physical_vertices
            projected_u, projected_v = derivative_u @ chart, derivative_v @ chart
            jacobian = np.linalg.norm(np.cross(physical_u, physical_v), axis=1) / abs(cross2(projected_u, projected_v))
            weights = projected_weights * np.sqrt(np.maximum(0, 1-squared_physical))
            weights /= np.sqrt(np.maximum(1e-300, 1-squared_planar))
            weights *= jacobian
            active = weights > 0
            if active.any():
                positions.append(physical[active])
                base_weights.append(weights[active])
                shapes.append(shape[active])
                parent_faces.append(np.full(active.sum(), face, dtype=int))
        if not base_weights:
            raise ValueError('Q4 physical sphere support is empty')
        positions = np.vstack(positions)
        weights = np.concatenate(base_weights)
        shapes = np.vstack(shapes)
        parent_faces = np.concatenate(parent_faces)
        integrated = perf_counter()
        friction = 1. if contact.friction is None else contact.friction
        patch = WrenchPatch(point, force, radius, friction, np.zeros(3), -normal)
        sample_force, diagnostics = PadPressureFit(positions, weights, patch).solve()
        fitted = perf_counter()
        load = np.zeros((len(model.vertices), 6))
        for face in np.unique(parent_faces):
            selection = parent_faces == face
            nodal = shapes[selection].T @ sample_force[selection]
            nodes = model.quads[face]
            np.add.at(load[:, :3], nodes, nodal)
            np.add.at(load[:, 3:], nodes, contact.side*.5*model.h*np.cross(model.directors[nodes], nodal))
        resultant = load[:, :3].sum(axis=0)
        moment = np.cross(model.vertices-point, load[:, :3]).sum(axis=0) + load[:, 3:].sum(axis=0)
        force_error = np.linalg.norm(resultant-force)/magnitude
        moment_error = np.linalg.norm(moment)/(magnitude*radius)
        if force_error > 1e-8 or moment_error > 1e-8:
            raise ValueError(f'Q4 scatter wrench failure: force={force_error}, moment={moment_error}')
        done = perf_counter()
        return load.reshape(-1), dict(valid=True, radius_m=float(radius), samples=len(weights), n_samples=len(weights),
            force_error=float(force_error), moment_error=float(moment_error), candidate_faces=len(candidates),
            hit_faces=len(np.unique(parent_faces)), analytic_count=0, base_mass=float(weights.sum()),
            inverse_method=self.inverse_method, pressure_fit_evaluations=diagnostics.constrained_evaluations,
            stages=dict(query=queried-started,integration=integrated-queried,fit=fitted-integrated,
                        scatter=done-fitted,total=done-started))
