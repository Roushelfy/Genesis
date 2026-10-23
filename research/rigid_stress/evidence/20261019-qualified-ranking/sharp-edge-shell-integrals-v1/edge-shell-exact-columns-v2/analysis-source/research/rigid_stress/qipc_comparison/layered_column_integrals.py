"""Exact affine-stress line integrals on verified layered tetrahedral geometry."""

import numpy as np
from scipy.spatial import cKDTree

from .common_metric import TetSampler


class LayeredColumnIntegrals:
    """Use layer topology only to find candidates; intersect actual Cartesian tetrahedra."""

    def __init__(self, vertices, tets, outer, inner, faces, top, bottom, projection):
        count, nfaces = len(outer), len(faces)
        layers = len(vertices) // count - 1
        assert len(vertices) == (layers + 1) * count and len(tets) == 3 * layers * nfaces
        np.testing.assert_array_equal(vertices[:count], outer)
        np.testing.assert_array_equal(vertices[-count:], inner)
        expected = np.sort(faces)
        # Check the actual source topology, including each tetrahedron's prism.
        for kind in range(3 * layers):
            nodes = tets[kind * nfaces : (kind + 1) * nfaces]
            assert np.all(np.isin(nodes // count, [kind // 3, kind // 3 + 1]))
            assert np.all(np.any(nodes[:, :, None] % count == expected[:, None], axis=2))
            assert np.all(np.any(nodes[:, :, None] % count == expected[:, None], axis=1))
        triangles = outer[faces]
        tree = cKDTree(triangles.mean(axis=1))
        chosen, ambiguity, reconstruction = [], 0, 0.0
        for start in range(0, len(top), 512):
            query = top[start : start + 512]
            end = bottom[start : start + len(query)]
            _, candidate = tree.query(query, k=min(64, nfaces))
            corners = triangles[candidate]
            edges = corners[:, :, 1:] - corners[:, :, :1]
            relative = query[:, None] - corners[:, :, 0]
            gram = np.einsum("nkai,nkbi->nkab", edges, edges)
            free = np.linalg.solve(gram, np.einsum("nkai,nki->nka", edges, relative)[..., None])[..., 0]
            bary = np.concatenate((1 - free.sum(axis=-1, keepdims=True), free), axis=-1)
            recovered_top = np.einsum("nkc,nkci->nki", bary, corners)
            recovered_bottom = np.einsum("nkc,nkci->nki", bary, inner[faces[candidate]])
            distance = np.maximum(
                np.linalg.norm(recovered_top - query[:, None], axis=-1),
                np.linalg.norm(recovered_bottom - end[:, None], axis=-1),
            )
            valid = (bary.min(axis=-1) >= -1e-10) & (distance < 2e-15)
            assert valid.any(axis=1).all(), "Material columns do not match the captured paired surfaces."
            selected = np.where(valid, candidate, nfaces).argmin(axis=1)
            chosen.append(candidate[np.arange(len(query)), selected])
            reconstruction = max(reconstruction, float(distance[np.arange(len(query)), selected].max()))
            ambiguity += int((valid.sum(axis=1) > 1).sum())
        self.faces = np.concatenate(chosen)
        self.elements = self.faces[:, None] + np.arange(3 * layers) * nfaces
        self.top, self.bottom, self.projection = top, bottom, projection
        vector = bottom - top
        self.height = np.sqrt(np.einsum("ni,nij,nj->n", vector, np.eye(3) - projection, vector))
        assert self.height.min() > 0
        sampler = TetSampler.allocate(vertices, tets)
        inverse, origin = sampler.inverse[self.elements], vertices[tets[self.elements, 0]]
        a3 = np.einsum("nkij,nkj->nki", inverse, top[:, None] - origin)
        b3 = np.einsum("nkij,nj->nki", inverse, vector)
        a, b = (
            np.concatenate((1 - a3.sum(axis=-1, keepdims=True), a3), axis=-1),
            np.concatenate((-b3.sum(axis=-1, keepdims=True), b3), axis=-1),
        )
        moving = np.abs(b) > 1e-12
        crossing = np.divide(-a, b, out=np.zeros_like(a), where=moving)
        lower = np.maximum(0, np.where(b > 1e-12, crossing, -np.inf).max(axis=-1))
        upper = np.minimum(1, np.where(b < -1e-12, crossing, np.inf).min(axis=-1))
        valid = (upper - lower > 1e-11) & ~((~moving) & (a < -1e-10)).any(axis=-1)
        length = np.where(valid, upper - lower, 0)
        raw_coverage = length.sum(axis=1)
        assert raw_coverage.min() >= 1 - 1e-9, (
            "Uncovered material column",
            np.flatnonzero(raw_coverage < 1 - 1e-9).tolist(),
        )
        # No cancellation of an overlap with a missing interval is allowed.
        order = np.argsort(np.where(valid, lower, np.inf), axis=1)
        lo, hi = np.take_along_axis(lower, order, axis=1), np.take_along_axis(upper, order, axis=1)
        active = np.take_along_axis(valid, order, axis=1)
        gaps = np.where(active[:, 1:], lo[:, 1:] - hi[:, :-1], 0)
        ordinary = (np.abs(raw_coverage - 1) <= 1e-9) & (np.abs(gaps).max(axis=1) < 1e-9)
        np.testing.assert_allclose(lo[ordinary, 0], 0, rtol=0, atol=1e-9)
        np.testing.assert_allclose(hi[np.flatnonzero(ordinary), valid[ordinary].sum(axis=1) - 1], 1, rtol=0, atol=1e-9)
        midpoint = np.where(valid, (lower + upper) / 2, 0)
        shape = a + b * midpoint[..., None]
        assert shape[valid].min() >= -2e-9 and shape[valid].max() <= 1 + 2e-9
        recovered = np.einsum("nkc,nkci->nki", shape, vertices[tets[self.elements]])
        expected_points = top[:, None] + midpoint[..., None] * vector[:, None]
        np.testing.assert_allclose(recovered[valid], expected_points[valid], rtol=0, atol=2e-16)
        self.membrane_coeff = length[..., None] * shape
        self.bending_coeff = length[..., None] * ((0.5 - midpoint[..., None]) * shape - b * length[..., None] ** 2 / 12)
        # Exact interface lines can belong to multiple tetrahedra. Integrate
        # the union once, selecting the first containing cell as in TetSampler.
        for row in np.flatnonzero(~ordinary):
            breaks = np.sort(np.r_[0, 1, lower[row, valid[row]], upper[row, valid[row]]])
            breaks = breaks[np.r_[True, np.diff(breaks) > 1e-11]]
            breaks[0], breaks[-1] = 0, 1
            self.membrane_coeff[row] = 0
            self.bending_coeff[row] = 0
            for left, right in zip(breaks[:-1], breaks[1:]):
                middle, span = (left + right) / 2, right - left
                selected = np.flatnonzero(valid[row] & (lower[row] <= middle + 1e-11) & (upper[row] >= middle - 1e-11))
                assert len(selected), ("Gap in interval union", row, left, right)
                cell = selected[0]
                bary = a[row, cell] + b[row, cell] * middle
                assert bary.min() >= -2e-9
                self.membrane_coeff[row, cell] += span * bary
                self.bending_coeff[row, cell] += span * ((0.5 - middle) * bary - b[row, cell] * span**2 / 12)
        coverage = self.membrane_coeff.sum(axis=(1, 2))
        np.testing.assert_allclose(coverage, 1, rtol=0, atol=1e-9)
        self.sampler, self.vertices, self.tets = sampler, vertices, tets
        self.diagnostics = {
            "columns": len(top),
            "segments": int(valid.sum()),
            "ambiguous_surface_face_columns": ambiguity,
            "surface_reconstruction_max_m": reconstruction,
            "coverage_error": float(np.abs(coverage - 1).max()),
            "maximum_raw_gap_or_overlap": float(np.abs(gaps).max()),
            "union_interface_columns": np.flatnonzero(~ordinary).tolist(),
        }

    def integrate(self, corners):
        membrane = np.einsum("nkc,nkcij->nij", self.membrane_coeff, corners[self.elements]) * self.height[:, None, None]
        bending = (
            np.einsum("nkc,nkcij->nij", self.bending_coeff, corners[self.elements]) * self.height[:, None, None] ** 2
        )
        return tuple(self.projection @ value @ self.projection for value in [membrane, bending])

    def controls(self, actual_corners):
        constant = np.array([[3.0, 2, 1], [2, -4, 5], [1, 5, 7]])
        direction = np.array([231.0, -137, 89])
        scalar = self.vertices[self.tets] @ direction
        field = (1 + scalar[..., None, None]) * constant
        membrane, bending = self.integrate(field)
        middle = (self.top + self.bottom) / 2
        tangent = self.projection @ constant @ self.projection
        expected_m = (self.height * (1 + middle @ direction))[:, None, None] * tangent
        expected_b = (-(self.height**2) * ((self.bottom - self.top) @ direction) / 12)[:, None, None] * tangent
        np.testing.assert_allclose(membrane, expected_m, rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(bending, expected_b, rtol=1e-7, atol=1e-17)
        actual_m, actual_b = self.integrate(actual_corners)
        checked, interfaces, maximum = 0, 0, 0.0
        for index in np.linspace(0, len(self.top) - 1, 64, dtype=int):
            elements, low, high = self.sampler.intervals(self.top[index], self.bottom[index])
            # A line lying on an interface has two valid one-sided fields.
            if abs(np.sum(high - low) - 1) > 1e-9:
                interfaces += 1
                continue
            m, moment = np.zeros((3, 3)), np.zeros((3, 3))
            for offset in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
                fraction = (low + high) / 2 + (high - low) * offset / 2
                points = self.top[index] + fraction[:, None] * (self.bottom[index] - self.top[index])
                free = np.linalg.solve(
                    (self.vertices[self.tets[elements, 1:]] - self.vertices[self.tets[elements, :1]]).transpose(
                        0, 2, 1
                    ),
                    (points - self.vertices[self.tets[elements, 0]])[..., None],
                )[..., 0]
                bary = np.column_stack((1 - free.sum(axis=1), free))
                tensors = np.einsum("qc,qcij->qij", bary, actual_corners[elements])
                m += np.einsum("q,qij->ij", (high - low) / 2, tensors) * self.height[index]
                moment += np.einsum("q,qij->ij", (high - low) / 2 * (0.5 - fraction), tensors) * self.height[index] ** 2
            p = self.projection[index]
            for observed, reference in [(actual_m[index], p @ m @ p), (actual_b[index], p @ moment @ p)]:
                difference = np.linalg.norm(observed - reference) / max(np.linalg.norm(reference), 1e-20)
                maximum = max(maximum, float(difference))
                np.testing.assert_allclose(observed, reference, rtol=1e-7, atol=1e-12)
            checked += 1
        assert checked >= 32
        return {
            "affine_membrane_bending_passed": True,
            "independent_full_mesh_columns": checked,
            "full_mesh_interface_columns": interfaces,
            "full_mesh_maximum_relative_change": maximum,
        }
