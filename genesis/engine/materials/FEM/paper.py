"""
Paper material for IPC-based shell simulation with plastic bending.

This material inherits from Cloth, replacing elastic bending (DiscreteShellBending)
with plastic bending (PlasticDiscreteShellBending) from libuipc.  The membrane
response stays unchanged (StrainLimitingBaraffWitkinShell).
"""

from .cloth import Cloth


class Paper(Cloth):
    """
    Paper material with plastic bending for thin shell simulation using IPC.

    Extends Cloth by adding plasticity parameters that drive
    ``PlasticDiscreteShellBending`` in the IPC backend.  When the dihedral
    angle at an edge exceeds ``yield_threshold``, the rest angle evolves
    permanently (crease formation).  ``hardening_modulus`` controls how
    much the yield threshold grows with accumulated plastic strain.

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
    yield_threshold : float, optional
        Yield threshold on dihedral angle deviation (radians). Bending
        beyond this angle causes permanent plastic deformation. Default
        is 0.02 (~1.15°).
    hardening_modulus : float, optional
        Linear hardening modulus. 0 gives perfect plasticity (constant
        yield threshold); positive values increase the threshold after
        each plastic event. Default is 0.1.
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
    ...         yield_threshold=0.02,
    ...         hardening_modulus=0.1,
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
        yield_threshold=0.02,
        hardening_modulus=0.1,
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

        self._yield_threshold = yield_threshold
        self._hardening_modulus = hardening_modulus

    @property
    def yield_threshold(self):
        """Yield threshold on dihedral angle deviation (radians)."""
        return self._yield_threshold

    @property
    def hardening_modulus(self):
        """Linear hardening modulus for plastic bending."""
        return self._hardening_modulus

    def __repr__(self):
        return (
            f"<gs.materials.FEM.Paper(E={self.E}, nu={self.nu}, rho={self.rho}, "
            f"thickness={self.thickness}, bending_stiffness={self.bending_stiffness}, "
            f"yield_threshold={self.yield_threshold}, hardening_modulus={self.hardening_modulus})>"
        )
