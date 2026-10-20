"""Four free rigid coordinates of the declared compression-only spring foundation."""

import numpy as np


def shell_pad_boundary_gaps(outer, midpoint, faces, current, coordinates):
    """Exact vertex extrema of the affine reference-geometry face envelope."""
    relative = midpoint[faces] - midpoint[faces].mean(axis=1, keepdims=True)
    gram = np.eye(3) * (relative**2).sum(axis=(1, 2))[:, None, None]
    gram -= np.einsum("fki,fkj->fij", relative, relative)
    inverse = np.linalg.inv(gram)
    displacement = (current - midpoint)[faces]
    delta = np.empty((len(faces), 3))
    for corner in range(3):
        offset = outer[faces[:, corner]] - midpoint[faces[:, corner]]
        couple = np.cross(offset, np.array([0.0, 0.0, 1.0]))
        multiplier = np.einsum("fij,fj->fi", inverse, couple)
        weights = np.cross(multiplier[:, None], relative)
        weights[:, corner, 2] += 1
        delta[:, corner] = np.sum(weights * displacement, axis=(1, 2))
    point = outer[faces]
    delta += coordinates[0] + coordinates[1] * point[..., 1] / 0.03 - coordinates[2] * point[..., 0] / 0.03
    bottom = point[..., 2] + delta - outer[:, 2].min()
    top = outer[:, 2].max() + coordinates[3] - point[..., 2] - delta
    return bottom, top


def normal_pad_equilibrium(gap, rigid, area, stiffness, load, initial=None):
    """Minimize pad energy plus plate work for a fixed elastic displacement field."""
    scale = load / (stiffness * np.pi * 0.002**2)
    weight = stiffness * area * scale / load
    coordinates = np.array([-1.0, 0.0, 0.0, -2.0]) if initial is None else initial / scale
    target = np.array([0.0, 0.0, 0.0, 1.0])
    for iteration in range(40):
        actual_gap = gap / scale + rigid @ coordinates
        compression = np.maximum(-actual_gap, 0)
        force = weight * compression
        gradient = target - rigid.T @ force
        if np.linalg.norm(gradient) <= 1e-10:
            return coordinates * scale, force * load, float(np.linalg.norm(gradient)), iteration + 1
        active = actual_gap < 0
        hessian = rigid[active].T @ (weight[active, None] * rigid[active])
        direction = np.linalg.solve(hessian, -gradient)
        energy = 0.5 * np.sum(weight * compression**2) + target @ coordinates
        slope = gradient @ direction
        assert slope < 0
        step = 1.0
        for backtrack in range(40):
            trial = coordinates + step * direction
            trial_compression = np.maximum(-(gap / scale + rigid @ trial), 0)
            trial_energy = 0.5 * np.sum(weight * trial_compression**2) + target @ trial
            if trial_energy <= energy + 1e-4 * step * slope + 16 * np.finfo(float).eps * abs(energy):
                coordinates = trial
                break
            step *= 0.5
        else:
            raise RuntimeError("Pad rigid-coordinate line search did not converge")
    raise RuntimeError("Pad rigid-coordinate force/moment balance did not converge")
