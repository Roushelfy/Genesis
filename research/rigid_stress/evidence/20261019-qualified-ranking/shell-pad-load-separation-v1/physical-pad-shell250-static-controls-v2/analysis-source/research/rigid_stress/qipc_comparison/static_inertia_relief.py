"""Six rigid-mode mass projection for a free static elastic load."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticInertiaRelief:
    rigid: np.ndarray
    mass_modes: np.ndarray
    gram: np.ndarray

    @classmethod
    def build(cls, vertices, masses):
        assert np.isfinite(vertices).all() and np.isfinite(masses).all() and (masses > 0).all()
        center = masses @ vertices / masses.sum()
        rigid = np.zeros((len(vertices), 3, 6))
        rigid[:, :, :3] = np.eye(3)
        for axis in range(3):
            rigid[:, :, 3 + axis] = np.cross(np.eye(3)[axis], vertices - center)
        rigid = rigid.reshape(-1, 6)
        mass_modes = rigid * np.repeat(masses, 3)[:, None]
        return cls(rigid, mass_modes, rigid.T @ mass_modes)

    def apply(self, force):
        load = force.reshape(-1)
        correction = self.mass_modes @ np.linalg.solve(self.gram, self.rigid.T @ load)
        return (load - correction).reshape(force.shape), correction.reshape(force.shape)
