"""
Paper material for IPC-based shell simulation with stress-based plastic bending.

This material inherits from Cloth, replacing elastic bending (DiscreteShellBending)
with stress-based plastic bending (StressPlasticDiscreteShellBending) from libuipc.
The membrane response stays unchanged (StrainLimitingBaraffWitkinShell).
"""

from .cloth import Cloth


class Paper(Cloth):
    """
    Paper material with stress-based plastic bending for thin shell simulation using IPC.

    Extends Cloth by adding plasticity parameters that drive
    ``StressPlasticDiscreteShellBending`` in the IPC backend.  When the
    bending moment at an edge exceeds ``yield_stress``, the rest angle
    evolves permanently (crease formation).  ``hardening_modulus`` controls
    isotropic hardening in stress space.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa). Default is 1e5 (stiffer than fabric).
    nu : float, optional
        Poisson's ratio. Default is 0.49.
    rho : float, optional
        Density (kg/m³). Default is 700 (typical paper/cardboard).
    thickness : float, optional
        Shell thickness (m). Default is 0.0003 (0.3 mm, standard paper).
    bending_stiffness : float, optional
        Bending stiffness (kPa). Required for plastic bending to have
        any effect. Default is 4e3.
    yield_stress : float, optional
        Yield stress on generalized bending moment. Bending beyond this
        stress causes permanent plastic deformation. Default is 960.0.
    hardening_modulus : float, optional
        Isotropic hardening modulus in stress space. 0 gives perfect
        plasticity; positive values increase the yield stress after
        each plastic event. Default is 0.0.
    friction_mu : float, optional
        Friction coefficient. Default is 0.3.
    contact_resistance : float | None, optional
        Per-entity IPC contact stiffness override. Default is None.

    Examples
    --------
    >>> paper = scene.add_entity(
    ...     morph=gs.morphs.Mesh(file="sheet.obj"),
    ...     material=gs.materials.FEM.Paper(
    ...         E=1e5, thickness=0.0003,
    ...         bending_stiffness=4e3,
    ...         yield_stress=960.0,
    ...     ),
    ... )
    """

    def __init__(
        self,
        E=1e5,
        nu=0.49,
        rho=700.0,
        thickness=0.0003,
        bending_stiffness=4e3,
        yield_stress=960.0,
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

        self._yield_stress = yield_stress
        self._hardening_modulus = hardening_modulus

    @property
    def yield_stress(self):
        """Yield stress on generalized bending moment."""
        return self._yield_stress

    @property
    def hardening_modulus(self):
        """Isotropic hardening modulus in stress space."""
        return self._hardening_modulus

    def __repr__(self):
        return (
            f"<gs.materials.FEM.Paper(E={self.E}, nu={self.nu}, rho={self.rho}, "
            f"thickness={self.thickness}, bending_stiffness={self.bending_stiffness}, "
            f"yield_stress={self.yield_stress}, hardening_modulus={self.hardening_modulus})>"
        )
