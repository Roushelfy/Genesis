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

    def __init__(
        self,
        E=1e5,
        nu=0.49,
        rho=700.0,
        thickness=0.0003,
        bending_stiffness=4e3,
        plasticity_model="stress",
        yield_stress=960.0,
        yield_threshold=0.02,
        hardening_modulus=0.0,
        model="stable_neohookean",
        friction_mu=0.3,
        contact_resistance=None,
    ):
        super().__init__(
            E=E,
            nu=nu,
            rho=rho,
            thickness=thickness,
            bending_stiffness=bending_stiffness,
            model=model,
            friction_mu=friction_mu,
            contact_resistance=contact_resistance,
        )

        if plasticity_model not in ("stress", "strain"):
            from genesis.utils.misc import raise_exception

            raise_exception(f"Unknown plasticity_model '{plasticity_model}'. Use 'stress' or 'strain'.")

        self._plasticity_model = plasticity_model
        self._yield_stress = yield_stress
        self._yield_threshold = yield_threshold
        self._hardening_modulus = hardening_modulus

    @property
    def plasticity_model(self):
        """Plasticity model: ``'stress'`` or ``'strain'``."""
        return self._plasticity_model

    @property
    def yield_stress(self):
        """Yield stress on generalized bending moment (stress model)."""
        return self._yield_stress

    @property
    def yield_threshold(self):
        """Yield threshold on dihedral angle deviation in radians (strain model)."""
        return self._yield_threshold

    @property
    def hardening_modulus(self):
        """Hardening modulus for plastic bending."""
        return self._hardening_modulus

    def __repr__(self):
        if self._plasticity_model == "stress":
            yield_str = f"yield_stress={self.yield_stress}"
        else:
            yield_str = f"yield_threshold={self.yield_threshold}"
        return (
            f"<gs.materials.FEM.Paper(E={self.E}, nu={self.nu}, rho={self.rho}, "
            f"thickness={self.thickness}, bending_stiffness={self.bending_stiffness}, "
            f"plasticity_model='{self.plasticity_model}', {yield_str}, "
            f"hardening_modulus={self.hardening_modulus})>"
        )
