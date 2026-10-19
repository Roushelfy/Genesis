"""Common tangent-plane peaks and separate normal/transverse stress measures."""

import numpy as np


def tangent_invariants(tensor):
    """Von Mises and nonnegative tensile stress of the tangent tensor embedded in 3D."""
    xx, yy, xy = tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 0, 1]
    mises = np.sqrt(xx * xx - xx * yy + yy * yy + 3 * xy * xy)
    tension = np.maximum((xx + yy) / 2 + np.sqrt(((xx - yy) / 2) ** 2 + xy * xy), 0)
    return mises, tension


class CommonColumnMeasures:
    def __init__(self, basis, normal, outer, inner, shell):
        self.basis, self.normal = basis, normal
        self.outer, self.inner, self.shell = outer, inner, shell
        self.peaks = [{name: None for name in ("mises", "tension")} for _ in range(2)]
        self.measure = 0.0
        self.squares = np.zeros((2, 2))
        self.difference_squares = np.zeros(2)
        self.full_squares = np.zeros(2)

    def update_peaks(self, tensors, depth):
        for peaks, tensor in zip(self.peaks, tensors):
            for name, values in zip(("mises", "tension"), tangent_invariants(tensor)):
                index = int(values.argmax())
                value = float(values[index])
                if peaks[name] is None or value > peaks[name]["stress_pa"]:
                    position = self.outer[index] * (1 - depth[index]) + self.inner[index] * depth[index]
                    peaks[name] = {
                        "stress_pa": value,
                        "common_reference_rest_position_m": position.tolist(),
                        "column_index": index,
                        "wall_depth": float(depth[index]),
                    }

    def integrate_transverse(self, tensors, weights):
        components = []
        self.measure += float(weights.sum())
        for index, tensor in enumerate(tensors):
            traction = np.einsum("qij,qj->qi", tensor, self.normal)
            normal = np.einsum("qi,qi->q", traction, self.normal)
            shear = np.einsum("qij,qi->qj", self.basis, traction)
            components.append((normal, shear))
            self.squares[index] += (weights @ (normal * normal), weights @ np.einsum("qi,qi->q", shear, shear))
            self.full_squares[index] += np.einsum("qij,qij,q->", tensor, tensor, weights)
        normal_error, shear_error = (a - r for a, r in zip(components[0], components[1]))
        self.difference_squares += (
            weights @ (normal_error * normal_error),
            weights @ np.einsum("qi,qi->q", shear_error, shear_error),
        )

    def result(self):
        peaks = []
        for name in ("mises", "tension"):
            actual, reference = (p[name] for p in self.peaks)
            difference = abs(actual["stress_pa"] - reference["stress_pa"])
            peaks.append(
                {
                    "metric": name,
                    "actual": actual,
                    "reference": reference,
                    "absolute_difference_pa": difference,
                    "relative_difference": difference / reference["stress_pa"] if reference["stress_pa"] else None,
                }
            )
        transverse = []
        for component, name in enumerate(("normal_stress", "transverse_shear_vector")):
            norms = self.squares[:, component]
            transverse.append(
                {
                    "component": name,
                    "rms_pa": [None if s else float(np.sqrt(n / self.measure)) for s, n in zip(self.shell, norms)],
                    "rms_error_pa": None
                    if any(self.shell)
                    else float(np.sqrt(self.difference_squares[component] / self.measure)),
                    "relative_l2_error": None
                    if any(self.shell) or norms[1] == 0
                    else float(np.sqrt(self.difference_squares[component] / norms[1])),
                }
            )
        return {
            "tangent_peaks": peaks,
            "full3d_only_components": transverse,
            "shell_has_no_independent_transverse_stress": list(self.shell),
            "transverse_frobenius_energy_fraction": [
                None if shell or full == 0 else float((square[0] + 2 * square[1]) / full)
                for shell, square, full in zip(self.shell, self.squares, self.full_squares)
            ],
            "scope": "Common tangent tensors embedded with zero normal stress and zero transverse shear define both peak invariants. This is distinct from each method's raw full3D peak. Identical angular rays sample every unioned tetrahedron boundary on both sides, outer/inner boundaries and depth Gauss nodes; surface/depth sampling convergence remains required. Positions are on the common reference rest columns. Normal and transverse shear components are integrated separately with the field metric weights; Shell lacks these independent 3D components and receives no transverse error rank.",
        }
