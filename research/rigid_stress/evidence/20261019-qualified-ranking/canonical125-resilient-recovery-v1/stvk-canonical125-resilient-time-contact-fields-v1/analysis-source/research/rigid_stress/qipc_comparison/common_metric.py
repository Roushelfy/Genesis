"""A shared material-column domain with exact P1/P2 through-wall integration partitions."""

from dataclasses import dataclass

import numpy as np


def requested_duration_s(receipt):
    config = receipt.get("config", {})
    if "requested_steps" not in receipt or "dt" not in config:
        return None
    return receipt["requested_steps"] * config["dt"]


def complete_control_grid(ticks, duration_s):
    if duration_s is None or not np.isfinite(duration_s) or duration_s <= 0:
        return False
    expected = round(duration_s / 0.01)
    observed = np.asarray(sorted(ticks))
    return bool(
        expected > 0
        and abs(expected * 0.01 - duration_s) < 1e-12
        and len(observed) == expected
        and np.array_equal(observed, np.arange(1, expected + 1))
    )


@dataclass
class TetSampler:
    vertices: np.ndarray
    tetrahedra: np.ndarray
    inverse: np.ndarray
    direct_point_query_fallbacks: int = 0

    @classmethod
    def allocate(cls, vertices, tetrahedra):
        edges = (vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]]).transpose(0, 2, 1)
        assert np.abs(np.linalg.det(edges)).min() > 0
        return cls(vertices, tetrahedra, np.linalg.inv(edges))

    def intervals(self, inner, outer):
        bary_free = np.einsum("tij,tj->ti", self.inverse, inner - self.vertices[self.tetrahedra[:, 0]])
        slope_free = np.einsum("tij,j->ti", self.inverse, outer - inner)
        bary = np.column_stack((1 - bary_free.sum(axis=1), bary_free))
        slope = np.column_stack((-slope_free.sum(axis=1), slope_free))
        moving = np.abs(slope) > 1e-12
        intersection = np.divide(-bary, slope, out=np.zeros_like(bary), where=moving)
        lower = np.maximum(0, np.where(slope > 1e-12, intersection, -np.inf).max(axis=1))
        upper = np.minimum(1, np.where(slope < -1e-12, intersection, np.inf).min(axis=1))
        parallel_outside = ((~moving) & (bary < -1e-10)).any(axis=1)
        valid = (upper - lower > 1e-11) & ~parallel_outside
        return np.flatnonzero(valid), lower[valid], upper[valid]

    def locate_on_column(self, points, fractions, intervals, selection_points=None):
        elements, lower, upper = intervals
        included = (fractions[:, None] >= lower[None] - 1e-11) & (fractions[:, None] <= upper[None] + 1e-11)
        covered = included.any(axis=1)
        selected = np.empty(len(points), dtype=np.int64)
        selected[covered] = elements[included[covered].argmax(axis=1)] if len(elements) else []
        # An almost parallel face can lose a valid interval near an endpoint.
        # Query the unchanged coordinates directly, retaining the existing
        # barycentric tolerance and independently checking the selected cell.
        # Boundary interpolation uses an interior selection point to keep its
        # one-sided cell contract, then evaluates at the exact boundary below.
        query = points if selection_points is None else selection_points
        for index in np.flatnonzero(~covered):
            free = np.einsum("tij,tj->ti", self.inverse, query[index] - self.vertices[self.tetrahedra[:, 0]])
            bary = np.column_stack((1 - free.sum(axis=1), free))
            candidate = bary.min(axis=1).argmax()
            assert bary[candidate].min() > -2e-9 and bary[candidate].max() < 1 + 2e-9, (
                "A common material column has a gap in the FEM domain"
            )
            vertices = self.vertices[self.tetrahedra[candidate]]
            independent = np.linalg.solve((vertices[1:] - vertices[0]).T, query[index] - vertices[0])
            checked = np.r_[1 - independent.sum(), independent]
            assert checked.min() > -2e-9 and checked.max() < 1 + 2e-9, (
                "Independent point query is outside the FEM domain"
            )
            selected[index] = candidate
            self.direct_point_query_fallbacks += 1
        bary_free = np.einsum(
            "qij,qj->qi", self.inverse[selected], points - self.vertices[self.tetrahedra[selected, 0]]
        )
        bary = np.column_stack((1 - bary_free.sum(axis=1), bary_free))
        assert bary.min() > -2e-9 and bary.max() < 1 + 2e-9
        return selected, bary


@dataclass
class CommonColumns:
    points: np.ndarray
    face_index: np.ndarray
    weights: np.ndarray
    z: np.ndarray
    area: np.ndarray
    thickness: np.ndarray
    normal: np.ndarray
    reference_elements: np.ndarray
    reference_barycentric: np.ndarray
    native_elements: np.ndarray
    native_barycentric: np.ndarray


@dataclass
class CommonSurfaces:
    points: np.ndarray
    face_index: np.ndarray
    weights_m2: np.ndarray
    z: np.ndarray
    side: np.ndarray
    normal: np.ndarray
    reference_elements: np.ndarray
    reference_barycentric: np.ndarray
    native_elements: np.ndarray
    native_barycentric: np.ndarray


def build_surfaces(reference, native, outer_vertices, inner_vertices, faces, columns):
    outer, inner = outer_vertices[faces].mean(axis=1), inner_vertices[faces].mean(axis=1)
    reference_elements, reference_barycentric, native_elements, native_barycentric = [], [], [], []
    for p_inner, p_outer in zip(inner, outer):
        points = np.stack((p_inner, p_outer))
        # Select the one-sided cell entered from inside the wall, then
        # interpolate at the exact boundary point. No surface smoothing or
        # coordinate clipping is applied to the stress query.
        fractions = np.array([1e-8, 1 - 1e-8])
        for sampler, elements, barycentric in (
            (reference, reference_elements, reference_barycentric),
            (native, native_elements, native_barycentric),
        ):
            selection_points = p_inner + fractions[:, None] * (p_outer - p_inner)
            cell, bary = sampler.locate_on_column(
                points, fractions, sampler.intervals(p_inner, p_outer), selection_points=selection_points
            )
            elements.append(cell)
            barycentric.append(bary)
    return CommonSurfaces(
        np.stack((inner, outer), axis=1).reshape(-1, 3),
        np.repeat(np.arange(len(faces)), 2),
        np.repeat(columns.area, 2),
        (columns.thickness[:, None] * np.array([-0.5, 0.5])).reshape(-1),
        np.tile(np.arange(2), len(faces)),
        columns.normal,
        np.concatenate(reference_elements),
        np.concatenate(reference_barycentric),
        np.concatenate(native_elements),
        np.concatenate(native_barycentric),
    )


def build_columns(reference, native, outer_vertices, inner_vertices, faces):
    outer, inner = outer_vertices[faces].mean(axis=1), inner_vertices[faces].mean(axis=1)
    midpoint = (outer_vertices + inner_vertices) / 2
    edges = midpoint[faces[:, 1:]] - midpoint[faces[:, :1]]
    normal = np.cross(edges[:, 0], edges[:, 1])
    area = np.linalg.norm(normal, axis=1) / 2
    normal /= 2 * area[:, None]
    thickness = np.einsum("fi,fi->f", outer - inner, normal)
    assert (thickness > 0).all()
    abscissa, gauss_weight = np.polynomial.legendre.leggauss(2)
    points, face_index, weights, z = [], [], [], []
    reference_elements, reference_barycentric, native_elements, native_barycentric = [], [], [], []
    for face, (p_inner, p_outer) in enumerate(zip(inner, outer)):
        reference_intervals = reference.intervals(p_inner, p_outer)
        native_intervals = native.intervals(p_inner, p_outer)
        breaks = np.sort(
            np.concatenate(
                ([0, 1], reference_intervals[1], reference_intervals[2], native_intervals[1], native_intervals[2])
            )
        )
        breaks = breaks[np.concatenate(([True], np.diff(breaks) > 1e-11))]
        breaks[0], breaks[-1] = 0, 1
        lengths = np.diff(breaks)
        fractions = ((breaks[:-1] + breaks[1:])[:, None] / 2 + lengths[:, None] * abscissa / 2).reshape(-1)
        sample = p_inner + fractions[:, None] * (p_outer - p_inner)
        ref_elements, ref_bary = reference.locate_on_column(sample, fractions, reference_intervals)
        nat_elements, nat_bary = native.locate_on_column(sample, fractions, native_intervals)
        points.append(sample)
        face_index.append(np.full(len(sample), face))
        weights.append((lengths[:, None] * gauss_weight / 2).reshape(-1) * thickness[face] * area[face])
        z.append((fractions - 0.5) * thickness[face])
        reference_elements.append(ref_elements)
        reference_barycentric.append(ref_bary)
        native_elements.append(nat_elements)
        native_barycentric.append(nat_bary)
    return CommonColumns(
        np.concatenate(points),
        np.concatenate(face_index),
        np.concatenate(weights),
        np.concatenate(z),
        area,
        thickness,
        normal,
        np.concatenate(reference_elements),
        np.concatenate(reference_barycentric),
        np.concatenate(native_elements),
        np.concatenate(native_barycentric),
    )


def voigt_tensor(voigt):
    tensor = np.empty((*voigt.shape[:-1], 3, 3))
    tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2] = voigt[..., 0], voigt[..., 1], voigt[..., 2]
    tensor[..., 0, 1] = tensor[..., 1, 0] = voigt[..., 3]
    tensor[..., 0, 2] = tensor[..., 2, 0] = voigt[..., 4]
    tensor[..., 1, 2] = tensor[..., 2, 1] = voigt[..., 5]
    return tensor


def corotation(authored, current, masses):
    center_reference = np.einsum("v,vi->i", masses, authored) / masses.sum()
    center_current = np.einsum("v,vi->i", masses, current) / masses.sum()
    covariance = np.einsum("v,vi,vj->ij", masses, authored - center_reference, current - center_current)
    left, _, right = np.linalg.svd(covariance)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(right.T @ left.T)
    return (right.T * sign) @ left.T


def tangent(tensor, columns):
    normals = columns.normal[columns.face_index]
    projector = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    return projector @ tensor @ projector


def field_error(actual, reference, columns, floor_pa=1.0):
    return tensor_error(actual, reference, columns.weights, floor_pa)


def tensor_error(actual, reference, weights, floor_pa=1.0):
    difference = actual - reference
    denominator = np.einsum("q,qij,qij->", weights, reference, reference)
    numerator = np.einsum("q,qij,qij->", weights, difference, difference)
    reference_rms = np.sqrt(denominator / weights.sum())
    return {
        "relative_tensor_error": float(np.sqrt(numerator / denominator)) if reference_rms > floor_pa else None,
        "absolute_tensor_rms_error_pa": float(np.sqrt(numerator / weights.sum())),
        "reference_tensor_rms_pa": float(reference_rms),
        "relative_floor_pa": floor_pa,
    }


def resultants(tensor, columns):
    membrane = np.zeros((len(columns.area), 3, 3))
    bending = np.zeros_like(membrane)
    weights = columns.weights / columns.area[columns.face_index]
    np.add.at(membrane, columns.face_index, weights[:, None, None] * tensor)
    np.add.at(bending, columns.face_index, (weights * columns.z)[:, None, None] * tensor)
    return membrane, bending


def time_integrated_error(times, reference_norms, error_norms, floor=1.0):
    """Same-observable L2 over all fixed end-of-interval observation times."""
    times, reference_norms, error_norms = map(np.asarray, (times, reference_norms, error_norms))
    widths = np.diff(np.r_[0.0, times])
    assert len(times) and (widths > 0).all()
    assert np.isfinite(reference_norms).all() and np.isfinite(error_norms).all()
    denominator = np.dot(widths, reference_norms**2)
    numerator = np.dot(widths, error_norms**2)
    reference_rms = np.sqrt(denominator / times[-1]).item()
    return {
        "relative_error": np.sqrt(numerator / denominator).item() if reference_rms > floor else None,
        "reference_rms_over_time": reference_rms,
        "absolute_rms_error_over_time": np.sqrt(numerator / times[-1]).item(),
        "observations": len(times),
        "duration_s": times[-1].item(),
        "relative_denominator_floor": floor,
        "rule": "End-of-interval time rectangles on the common observation grid. Every finite observation contributes, including near-zero-load intervals; no per-time relative-error mask or phase cut is applied. This is one field norm over time, not an additive error budget across modeling layers.",
    }
