"""FP64 fused CPU oracle for the global unaveraged affine P2 von Mises peak."""

import numpy as np
from numba import njit

from .mechanics import EDGES


@njit(cache=True)
def scan(gradients, elements, displacement):
    best, best_element, best_corner = -1.0, 0, 0
    for element in range(len(elements)):
        origin = displacement[elements[element, 0]]
        for corner in range(4):
            xx = yy = zz = xy = yz = xz = 0.0
            for node in range(10):
                if node < 4:
                    coefficient = 3.0 if node == corner else -1.0
                    gx, gy, gz = coefficient * gradients[element, node]
                else:
                    a, b = EDGES[node - 4]
                    if a == corner:
                        gx, gy, gz = 4 * gradients[element, b]
                    elif b == corner:
                        gx, gy, gz = 4 * gradients[element, a]
                    else:
                        continue
                ux, uy, uz = displacement[elements[element, node]] - origin
                xx += ux * gx
                yy += uy * gy
                zz += uz * gz
                xy += ux * gy + uy * gx
                yz += uy * gz + uz * gy
                xz += ux * gz + uz * gx
            value = 2 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + yz**2 + xz**2)
            if value > best:
                best, best_element, best_corner = value, element, corner
    return np.sqrt(best), best_element, best_corner


class P2Peak:
    def __init__(self, gradients: np.ndarray, elements: np.ndarray, mu: float) -> None:
        self.gradients = np.ascontiguousarray(gradients, dtype=np.float64)
        self.elements = np.ascontiguousarray(elements, dtype=np.int64)
        self.mu = float(mu)

    def __call__(self, displacement: np.ndarray) -> tuple[float, int, int]:
        nodal = np.ascontiguousarray(displacement.reshape(-1, 3), dtype=np.float64)
        value, element, corner = scan(self.gradients, self.elements, nodal)
        return self.mu * float(value), int(element), int(corner)

    def warmup(self, displacement: np.ndarray) -> tuple[float, int, int]:
        return self(displacement)
