"""
Rope material for IPC-based 1D rod/string simulation.

This material is used with FEMEntity and IPCCoupler for 1D rod simulation.
It maps to HookeanSpring (stretch) + KirchhoffRodBending (bending) in libuipc.
"""

from .base import Base


class Rope(Base):
    """
    Rope material for 1D rod/string simulation using IPC.

    This material is designed for ropes, strings, cables, and other 1D
    flexible elements.  It uses HookeanSpring for axial stretch resistance
    and optionally KirchhoffRodBending for bending stiffness in the IPC
    backend.

    The mesh should be a line mesh (edges connecting vertices in sequence).

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa) for stretch stiffness via HookeanSpring.
        Default is 1e6 (1 MPa).
    nu : float, optional
        Poisson's ratio (unused for 1D, kept for compatibility).
        Default is 0.0.
    rho : float, optional
        Material density (kg/m³). Default is 1000.
    thickness : float, optional
        Rod radius (m). Used for mass computation and contact thickness.
        Default is 0.005 (5 mm).
    bending_stiffness : float, optional
        Bending stiffness for KirchhoffRodBending (Pa). If None, no
        bending resistance (pure string). Default is None.
    friction_mu : float, optional
        Friction coefficient. Default is 0.3.
    contact_resistance : float | None, optional
        Per-entity IPC contact stiffness override. Default is None.

    Examples
    --------
    >>> rope = scene.add_entity(
    ...     morph=gs.morphs.Mesh(file="rope.obj"),
    ...     material=gs.materials.FEM.Rope(
    ...         E=1e6, rho=1000, thickness=0.005,
    ...         bending_stiffness=1e4,
    ...     ),
    ... )
    """

    def __init__(
        self,
        E=1e6,
        nu=0.0,
        rho=1000.0,
        thickness=0.005,
        bending_stiffness=None,
        friction_mu=0.3,
        contact_resistance=None,
    ):
        super().__init__(E=E, nu=nu, rho=rho, friction_mu=friction_mu, contact_resistance=contact_resistance)

        self._thickness = thickness
        self._bending_stiffness = bending_stiffness

    @property
    def thickness(self):
        """Rod radius (m)."""
        return self._thickness

    @property
    def bending_stiffness(self):
        """Bending stiffness for KirchhoffRodBending (Pa)."""
        return self._bending_stiffness

    def __repr__(self):
        return (
            f"<gs.materials.FEM.Rope(E={self.E}, rho={self.rho}, "
            f"thickness={self.thickness}, bending_stiffness={self.bending_stiffness})>"
        )
