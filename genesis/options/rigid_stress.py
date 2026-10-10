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

    ``method="auto"`` uses a full pinned inverse when its storage fits
    ``inverse_max_bytes`` (64 MiB by default), otherwise a shared sparse
    block factor. Both are constructed and applied in Quadrants. The inverse
    handles arbitrary nodal loads and costs quadratic shared storage; it is
    intended for small meshes. ``method="direct"`` forces the sparse factor.
    CUDA meshes up to 1,024 P2 nodes use a cooperative warp solve when
    ``cooperative_solve`` is enabled; other cases use the serial native path.
    ``method="inverse"`` explicitly requests the inverse and rejects an
    insufficient storage budget. ``inverse_precision="32"`` stores it in
    FP32 while keeping arithmetic in scene precision. Failed complete
    residuals receive up to ``inverse_corrections`` corrections, then a
    masked sparse solve. The default inverse uses scene precision.
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
    history_size: Literal[0, 4] = 0
    method: Literal["auto", "direct", "inverse", "pcg"] = "auto"
    cooperative_solve: StrictBool = True
    cooperative_pressure: StrictBool = True
    inverse_max_bytes: PositiveInt = 64 * 1024 * 1024
    inverse_precision: Literal["64", "32"] = "64"
    inverse_corrections: Annotated[int, Field(ge=0, le=8)] = 2
    preconditioner: Literal["diagonal", "block"] = "block"
