"""
Paper material for IPC-based shell simulation with plastic bending.

This material inherits from Cloth, replacing elastic bending (DiscreteShellBending)
with plastic bending from libuipc. Two plasticity models are available:

- ``"stress"`` (default): StressPlasticDiscreteShellBending — yields when the
  bending moment exceeds ``yield_stress``.
- ``"strain"``: StrainPlasticDiscreteShellBending — yields when the dihedral
  angle deviation exceeds ``yield_threshold``.

The membrane response stays unchanged (StrainLimitingBaraffWitkinShell).
"""

from typing import Literal

from genesis.typing import NonNegativeFloat, PositiveFloat

from .cloth import Cloth


class Paper(Cloth):
    """
    Paper material with plastic bending for thin shell simulation using IPC.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa). Default is 1e5.
    nu : float, optional
        Poisson's ratio. Default is 0.49.
    rho : float, optional
        Density (kg/m³). Default is 700.
    thickness : float, optional
        Shell thickness (m). Default is 0.0003 (0.3 mm).
    bending_stiffness : float, optional
        Bending stiffness (kPa). Default is 4e3.
    plasticity_model : str, optional
        ``"stress"`` for stress-based yield or ``"strain"`` for
        strain-based yield. Default is ``"stress"``.
    yield_stress : float, optional
        Yield stress on generalized bending moment (stress model only).
        Default is 960.0.
    yield_threshold : float, optional
        Yield threshold on dihedral angle deviation in radians
        (strain model only). Default is 0.02.
    hardening_modulus : float, optional
        Hardening modulus. 0 gives perfect plasticity. Default is 0.0.
    friction_mu : float, optional
        Friction coefficient. Default is 0.3.
    contact_resistance : float | None, optional
        Per-entity IPC contact stiffness override. Default is None.

    Examples
    --------
    >>> # Stress-based (default)
    >>> paper = gs.materials.FEM.Paper(yield_stress=960.0)
    >>> # Strain-based
    >>> paper = gs.materials.FEM.Paper(plasticity_model="strain", yield_threshold=0.02)
    """

    E: PositiveFloat = 1e5
    nu: PositiveFloat = 0.49
    rho: PositiveFloat = 700.0
    thickness: PositiveFloat = 0.0003
    bending_stiffness: NonNegativeFloat | None = 4e3
    plasticity_model: Literal["stress", "strain"] = "stress"
    yield_stress: PositiveFloat = 960.0
    yield_threshold: PositiveFloat = 0.02
    hardening_modulus: NonNegativeFloat = 0.0
    model: Literal["strain_limiting_baraff_witkin", "neohookean"] = "strain_limiting_baraff_witkin"
    friction_mu: NonNegativeFloat = 0.3
