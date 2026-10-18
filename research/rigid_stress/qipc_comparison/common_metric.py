"""A shared material-column domain with exact P1/P2 through-wall integration partitions."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TetSampler:
    vertices: np.ndarray
    tetrahedra: np.ndarray
    inverse: np.ndarray

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

    def locate_on_column(self, points, fractions, intervals):
        elements, lower, upper = intervals
        included = (fractions[:, None] >= lower[None] - 1e-11) & (fractions[:, None] <= upper[None] + 1e-11)
        assert included.any(axis=1).all(), "A common material column has a gap in the FEM domain"
        selected = elements[included.argmax(axis=1)]
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
    difference = actual - reference
    denominator = np.einsum("q,qij,qij->", columns.weights, reference, reference)
    numerator = np.einsum("q,qij,qij->", columns.weights, difference, difference)
    reference_rms = np.sqrt(denominator / columns.weights.sum())
    return {
        "relative_tensor_error": float(np.sqrt(numerator / denominator)) if reference_rms > floor_pa else None,
        "absolute_tensor_rms_error_pa": float(np.sqrt(numerator / columns.weights.sum())),
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
