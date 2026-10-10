from typing import Annotated, Literal

from pydantic import Field, StrictBool

from genesis.typing import PathType, PositiveFloat, PositiveInt, ValidFloat

from .options import Options


class RigidStressOptions(Options):
    """Configure auxiliary small-strain stress observation on a rigid link.

    ``mesh`` names an NPZ asset containing ``vertices`` in the authored link
    frame, affine ``tetrahedra`` and exterior ``surface_triangles``. Recovery
    uses quadratic displacement on the complete tetrahedral mesh. The mesh
    does not change rigid collision geometry or feed displacement into motion.

    ``contact_radius`` declares the finite pressure footprint in metres. A
    larger footprint spreads the same contact wrench over more shell area;
    choose it from the contact model rather than the elastic mesh resolution.
    ``quadrature`` controls surface integration cost and accuracy.

    ``tolerance`` bounds the complete equilibrium residual relative to the
    load norm; ``absolute_tolerance`` supplies the force floor near zero.
    A smaller tolerance costs more solve work and does not by itself certify
    a stress error. Invalid loads or unresolved observations set the rigid
    solver error flag and produce a NaN observation.

    ``method="direct"`` builds a shared native sparse block factor once.
    ``method="pcg"`` trades factor storage for iterative work, bounded by
    ``max_iterations``; thin shells can require many iterations. Its
    ``warm_start`` reuses only the same environment's preceding displacement
    and is invalidated by state changes.
    """

    mesh: PathType
    young: PositiveFloat = 1e10
    poisson: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.3
    density: PositiveFloat = 2000.0
    contact_radius: PositiveFloat = 0.006
    quadrature: Annotated[PositiveInt, Field(le=16)] = 10
    tolerance: PositiveFloat = 1e-8
    absolute_tolerance: PositiveFloat = 1e-11
    max_iterations: PositiveInt = 2000
    warm_start: StrictBool = True
    method: Literal["direct", "pcg"] = "direct"
    preconditioner: Literal["diagonal", "block"] = "block"
