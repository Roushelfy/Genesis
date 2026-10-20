from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base, SamplerType


class Liquid(Base):
    """
    The liquid material class for SPH.

    Parameters
    ----------
    rest_density : float, optional
        The rest density (kg/m³). Default is 1000.0.
    viscosity : float, optional
        The viscosity of the liquid. Default is 0.005.
    surface_tension : float, optional
        The surface tension of the liquid. Default is 0.01.
    sampler : str, optional
        Particle sampler. Defaults to 'regular' for numerical stability with SPH.
    """

    rest_density: PositiveFloat = 1000.0
    stiffness: PositiveFloat = 50000.0
    exponent: PositiveFloat = 7.0
    viscosity: NonNegativeFloat = 0.005
    surface_tension: NonNegativeFloat = 0.01
    sampler: SamplerType = "regular"
