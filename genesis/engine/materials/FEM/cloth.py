"""
Cloth material for IPC-based cloth simulation.

This material is used with FEMEntity and IPCCoupler for shell/membrane simulation.
"""

from .base import Base

VALID_SHELL_MODELS = ("neohookean", "strain_limiting_baraff_witkin")


class Cloth(Base):
    """
    Cloth material for thin shell/membrane simulation using IPC.

    This material is designed for cloth, fabric, and other thin flexible materials.
    It uses shell-based FEM formulation in the IPC backend.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa), controlling stiffness. Default is 1e4 (10 kPa).
    nu : float, optional
        Poisson's ratio, describing volume change under stress.
        Default is 0.49 (nearly incompressible).
    rho : float, optional
        Material density (kg/m³). Default is 200 (typical fabric).
    thickness : float, optional
        Shell thickness (m). Default is 0.001 (1mm).
    bending_stiffness : float, optional
        Bending resistance coefficient. If None, no bending resistance.
        Default is None.
    model : str, optional
        Shell constitution model. Options:

        - ``"strain_limiting_baraff_witkin"`` (default): Baraff-Witkin model
          with strain limiting. Prevents unrealistic stretching.
        - ``"neohookean"``: Standard Neo-Hookean shell model. More physically
          grounded, allows larger elastic deformation.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override. ``None`` uses the IPC coupler
        global default. Default is None.

    Examples
    --------
    >>> cloth = scene.add_entity(
    ...     morph=gs.morphs.Mesh(file="cloth.obj"),
    ...     material=gs.materials.FEM.Cloth(
    ...         E=10e3, nu=0.49, rho=200,
    ...         thickness=0.001, bending_stiffness=10.0,
    ...         model="strain_limiting_baraff_witkin",
    ...     ),
    ... )
    """

    def __init__(
        self,
        E=1e4,
        nu=0.49,
        rho=200.0,
        thickness=0.001,
        bending_stiffness=None,
        model="strain_limiting_baraff_witkin",
        friction_mu=0.1,
        contact_resistance=None,
    ):
        super().__init__(E=E, nu=nu, rho=rho, friction_mu=friction_mu, contact_resistance=contact_resistance)

        if model not in VALID_SHELL_MODELS:
            from genesis.utils.misc import raise_exception

            raise_exception(f"Unknown shell model '{model}'. Valid: {VALID_SHELL_MODELS}")

        self._thickness = thickness
        self._bending_stiffness = bending_stiffness
        self._model = model

    @property
    def thickness(self):
        """Shell thickness (m)."""
        return self._thickness

    @property
    def bending_stiffness(self):
        """Bending stiffness coefficient."""
        return self._bending_stiffness

    @property
    def model(self):
        """Shell constitution model name."""
        return self._model

    def __repr__(self):
        return (
            f"<gs.materials.FEM.Cloth(E={self.E}, nu={self.nu}, rho={self.rho}, "
            f"thickness={self.thickness}, model='{self.model}')>"
        )
