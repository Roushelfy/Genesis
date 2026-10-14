"""Exact fixed-geometry centrifugal mass loads, with six changing angular-velocity coefficients.

For every node r relative to the COM, omega x (omega x r) is quadratic in omega and linear in r. Multiplying its six
coefficient fields by the complete consistent mass matrix offline preserves the original nodal load. This cache does
not describe or restrict contacts, and makes no low-rank approximation to the elastic displacement or stress.
"""

import cupy as cp
import numpy as np

from .body_gpu import FixedFieldProductGPU


class CentrifugalInertiaGPU:
    def __init__(self, mass, relative_position_m: np.ndarray, fused: bool = False) -> None:
        nodes = len(relative_position_m)
        fields = np.zeros((nodes, 3, 6))
        x, y, z = relative_position_m.T
        fields[:, 1, 0], fields[:, 2, 0] = -y, -z
        fields[:, 0, 1], fields[:, 2, 1] = -x, -z
        fields[:, 0, 2], fields[:, 1, 2] = -x, -y
        fields[:, 0, 3], fields[:, 1, 3] = y, x
        fields[:, 0, 4], fields[:, 2, 4] = z, x
        fields[:, 1, 5], fields[:, 2, 5] = z, y
        self.mass_fields = cp.asarray(mass @ fields.reshape(3 * nodes, 6))
        self.product = FixedFieldProductGPU(self.mass_fields) if fused else None

    def __call__(self, omega_rad_s: cp.ndarray) -> cp.ndarray:
        x, y, z = omega_rad_s.T
        coefficients = cp.stack((x * x, y * y, z * z, x * y, x * z, y * z))
        return self.mass_fields @ coefficients if self.product is None else self.product(coefficients)
