"""
Cloth material for IPC-based cloth simulation.

This material is used with FEMEntity and IPCCoupler for shell/membrane simulation.
"""

from typing import Literal

from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base


class Cloth(Base):
    """
    Cloth material for thin shell/membrane simulation using IPC.

    This material is designed for cloth, fabric, and other thin flexible materials.
    It uses shell-based FEM formulation (NeoHookeanShell) in the IPC backend.

    When used with FEMEntity, it signals to IPCCoupler that this entity should be
    treated as a 2D shell (cloth) rather than a 3D volumetric FEM object.

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
    aerodynamic_drag : float, optional
        Aerodynamic drag coefficient (C_d * rho_air / 2). Models air resistance
        as a dissipative force proportional to the face-normal velocity component
        and face area. If None, no aerodynamic damping is applied. Default is None.
    model : str, optional
        Shell constitution model. ``"strain_limiting_baraff_witkin"`` uses
        ``StrainLimitingBaraffWitkinShell``, ``"neohookean"`` uses ``NeoHookeanShell``.
        Default is ``"strain_limiting_baraff_witkin"``.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override. ``None`` uses the IPC coupler
        global default. Default is None.
    contact_d_hat : float | None, optional
        Per-entity contact distance threshold override. ``None`` uses the
        global ``IPCCouplerOptions.contact_d_hat``. Default is None.

    Notes
    -----
    - Only works with IPCCoupler enabled
    - Requires GPU backend
    - Only accepts surface mesh morphs (Mesh, etc.)
    - Uses FEMEntity infrastructure but simulated as 2D shell in IPC

    Examples
    --------
    >>> cloth = scene.add_entity(
    ...     morph=gs.morphs.Mesh(file="cloth.obj"),
    ...     material=gs.materials.FEM.Cloth(
    ...         E=10e3, nu=0.49, rho=200, thickness=0.001,
    ...         bending_stiffness=10.0, aerodynamic_drag=1.0,
    ...     ),
    ... )
    """

    E: PositiveFloat = 1e4
    nu: PositiveFloat = 0.49
    rho: PositiveFloat = 200.0
    thickness: PositiveFloat = 0.001
    bending_stiffness: NonNegativeFloat | None = None
    aerodynamic_drag: NonNegativeFloat | None = None
    curvature_drag_scale: NonNegativeFloat = 0.0
    curvature_inflate_scale: NonNegativeFloat = 0.0
    model: Literal["strain_limiting_baraff_witkin", "neohookean"] = "strain_limiting_baraff_witkin"
