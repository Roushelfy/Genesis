"""Slender sharp-edge pressure scales from strip equilibrium and surface compatibility."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma


@dataclass(frozen=True)
class EdgeScales:
    width: float
    along: float
    exponent: float
    amplitude: float


def edge_scales(load, modulus, slope, curvature, shear_ratio, poisson=0.3):
    """Use a constant slope across the edge and quadratic gap along the edge.

    Strip equilibrium gives p=C*((w*(1-y^2/b^2)-x)/x)^lambda and total force
    2*pi*E_star*slope*lambda*w*b/3. Matching center/end surface displacements
    for w/b << 1 gives one logarithmic equation for b. Its larger root is the
    slender branch. No measured support width or reference pressure is fitted.
    """
    assert min(load, modulus, slope, curvature) > 0 and -1 < poisson < 0.5
    beta = (1 - 2 * poisson) / (2 * (1 - poisson))
    exponent = 0.5 + np.arctan(beta * shear_ratio) / np.pi
    coefficient = 3 * load / (np.pi * modulus * curvature)
    product = 3 * load / (2 * np.pi * modulus * slope * exponent)
    constant = -digamma(1 - exponent) + digamma(2) - 1.5 - np.pi * beta * shear_ratio / 2

    def equation(along):
        return along**3 / coefficient - np.log(2 * along**2 / product) - constant

    lower = (2 * coefficient / 3) ** (1 / 3)
    assert equation(lower) < 0, "No slender contact branch for these inputs."
    upper = lower * 2
    for _ in range(12):
        if equation(upper) > 0:
            break
        upper *= 2
    along = brentq(equation, lower, upper, xtol=1e-16, rtol=1e-13)
    width = product / along
    assert width / along < 0.05, "The slender edge approximation needs width/along < 0.05."
    amplitude = modulus * slope * np.sin(np.pi * exponent) / 2
    return EdgeScales(width, along, exponent, amplitude)
