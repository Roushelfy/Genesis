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
    ``cooperative_pressure`` assigns a CUDA warp to each active contact;
    CPU and disabled cooperation use the same native serial pressure law.
    ``cooperative_scatter`` reduces loads on each surface face before node
    accumulation. CUDA contacts benefit from fewer writes. Disable it for
    a scalar scheduling comparison. CPU uses the scalar path.
    ``face_parallel_scatter`` additionally schedules CUDA contact faces in
    parallel. ``scatter_tasks_per_env`` bounds its workspace, independently
    of the actual contacts. On overflow the device fully recomputes the
    original contact-warp scatter; no load or sample is truncated. Disable
    it to compare scheduling. Both options select storage at scene build.
    ``cached_peak`` reuses shared immutable P2 corner gradients in the full
    global stress scan. Disable it for a geometry reuse comparison.

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
    ``surface_inverse`` accelerates exterior contact loads using their complete
    nodal space and the rigid inertia fields. It adds shared storage within
    the inverse budget. Disable it to compare the full-load inverse path.
    ``packed_surface_loads`` refreshes an exact list of nonzero boundary nodes
    on the device before applying that operator. It skips only exactly zero
    vectors and preserves node order. Disable it to measure dense application.
    ``fused_pipeline`` captures the serial CUDA association, contact fit,
    complete solve/residual/peak and acceptance passes in one native graph
    for cooperative exterior inverse recovery without load history. Other
    configurations retain the independently callable native passes.
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
    ``history_size=4`` enables a checked four-column load/displacement
    predictor per environment. The default zero avoids its extra memory
    and basis maintenance cost on changing-contact workloads.

    ``output_mode="max"`` stores only the global von Mises maximum.
    ``output_mode="full"`` also stores the authored-frame symmetric stress
    tensor and von Mises value at all four corners of every tetrahedron.
    Full fields describe the latest recovery substep. Select the mode before
    building the scene; changing it requires rebuilding.
    """

    mesh: PathType
    output_mode: Literal["max", "full"] = "max"
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
    cooperative_scatter: StrictBool = True
    face_parallel_scatter: StrictBool = True
    scatter_tasks_per_env: PositiveInt = 32
    inverse_max_bytes: PositiveInt = 64 * 1024 * 1024
    inverse_precision: Literal["64", "32"] = "64"
    surface_inverse: StrictBool = True
    packed_surface_loads: StrictBool = True
    fused_pipeline: StrictBool = True
    cached_peak: StrictBool = True
    cached_face_bounds: StrictBool = True
    inverse_corrections: Annotated[int, Field(ge=0, le=8)] = 2
    preconditioner: Literal["diagonal", "block"] = "block"
