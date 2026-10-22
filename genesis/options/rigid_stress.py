from typing import Annotated, Literal

from pydantic import Field, StrictBool
from typing_extensions import Self

from genesis.typing import PathType, PositiveFloat, PositiveInt, ValidFloat

from .options import Options


class RigidStressOptions(Options):
    """Configure auxiliary small-strain stress observation on a rigid link.

    ``mesh`` names an NPZ asset containing ``vertices`` in the authored link
    frame, affine ``tetrahedra`` and exterior ``surface_triangles``. Recovery
    uses quadratic displacement on the complete tetrahedral mesh. The mesh
    does not change rigid collision geometry or feed displacement into motion.

    ``RigidStressOptions.dpr(mesh=...)`` selects Directional Patch Recovery:
    finite tractions aligned with each supplied normal-plus-friction force.
    ``RigidStressOptions.scr(mesh=...)`` selects Surface-Constrained Recovery:
    intrinsic surface support with the body-source-prior partition. These
    constructors select complete load routes without changing any numerical
    default. Other options can be passed as keywords. Use the explicit main
    constructor when choosing a different partition or support variant.

    ``contact_radius`` declares the finite traction footprint in metres. A
    larger footprint spreads the same contact wrench over more shell area;
    choose it from the contact model rather than the elastic mesh resolution.
    ``quadrature`` controls surface integration cost and accuracy.
    ``load_transfer="resolved"`` declares that pressure-mode input forces are
    already integrated boundary samples. It transfers each sample through the
    P2 shape functions of its witnessed material face, without another finite
    disk. The witness may identify an outer, inner or nonconvex face. Missing
    or ambiguous provenance and an unrepresented original-point moment are
    rejected. Boundary association obeys the declared projection limit and
    final force/moment checks. This declaration does not infer
    physical support or make sparse collision points resolved pressure data.
    The default ``"patch"`` retains the manual finite-footprint baseline.
    ``contact_model="intrinsic"`` preserves collision geom/pair witnesses and
    fits vector tractions in the local inward material Coulomb cones. The
    finite profile multiplies positive area weights by a Gaussian of width
    0.45 times the physical radius and a squared compact taper. Surface paths
    follow shared topological edge ports. Verified covered planar segments
    use their exact length. ``surface_edge_subdivisions`` declares the port
    discretization. Its graph distances can overestimate continuous distance.
    ``intrinsic_integration="surface_fan"`` partitions each face around its
    closest point to the anchor using three finite triangles with squared
    integration coordinates to resolve support close to edges. The earlier
    ``"anchor_fan"`` rule partitions only the witnessed face and remains an
    explicit integration comparison. Neither rule changes the physical radius.
    Adjacent faces additionally use a verified two-triangle unfolded path.
    ``cooperative_tractions`` distributes intrinsic fitting, original-contact
    acceptance and signed scatter across CUDA warp lanes. Unresolved fits use
    a device-masked retry of the canonical scalar algorithm; CPU retains that
    scalar route. Disabling this option provides a matched scheduling baseline
    without changing the load law, sampling or final acceptance budgets.
    ``surface_fan_frontier`` schedules CUDA fan sampling on a compact list of
    candidate faces. It preserves each original sample and intrinsic distance
    check, and includes the device maximum readback. CPU retains its original
    full-face route. Disable it for a matched scheduling comparison.
    ``warm_start_tractions`` allows CUDA fitting to start from a previous
    successful fit. This reduces work on smooth loads but can add cold-retry
    work after abrupt changes. Every current input and final result still
    receives complete validation; no contact identity is assumed. Disable
    this option to start every fit from zero.
    ``intrinsic_support="pair_union"`` explicitly partitions each original
    point wrench over the union of all finite witnessed disks from that
    collision geom/pair. Each component retains its original force, point,
    free moment and moment budget. Every destination retains its own radius,
    material normal and friction coefficient. The default ``"contact"``
    confines each component to its own disk. Pair unions may contain separate
    islands; they do not create surface adjacency or enlarge any disk.
    ``intrinsic_load_partition="body_source_prior"`` is an explicit alternative
    load law: fit the complete link wrench on all its witnessed disks, with
    each original contact force as a separate local traction prior. It preserves
    self-balanced compression even when the net wrench vanishes. Individual
    point wrenches are priors, while their sum is the conserved constraint;
    every source retains its actual normal, friction and radius. The default
    ``"componentwise"`` preserves the original per-point constraints. Body
    partitioning uses all sources independently of ``intrinsic_support`` and
    grows its workspace to fit the actual count without truncation.
    ``contact_block_size`` bounds the concurrent workspace. All
    contacts are processed, including subsequent blocks when needed. Pair-union
    blocks keep complete pairs together; their workspace is at least the
    collider's applicable per-pair capacity to prevent truncating a union.
    ``source_projection_limit`` bounds collision-witness projection onto the
    reference boundary. ``source_geometry_tolerance`` declares mesh/reference
    vertex correspondence. ``source_uncertainty`` bounds ambiguity around the
    witness. These geometry bounds preserve the original application point,
    force, moment and physical footprint radius. The intrinsic path retains
    the independently callable recovery kernels.
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
    ``cooperative_balance`` reduces the complete rigid-mode wrench in CUDA
    node tiles and applies a shared mass-mode projection. CPU uses its
    original balance order. Disable it to compare scheduling and projection.
    ``contact_wrench_reuse`` optionally projects the actual integrated contact
    wrench instead of reducing it again from all nodes. Overflow retains the
    complete nodal reduction. ``contact_moment_reuse`` integrates current-step
    P2 load moments alongside the pressure Gram matrix. Strictly positive
    affine pressures reuse those moments; constrained, refined and overflowing
    cases retain complete sampling. It needs CUDA cooperative pressure and
    face scatter, adds 217 bytes per task capacity, and assumes no contact
    history. ``packed_block_size`` selects the CUDA inverse block size;
    ``coalesced_residual`` changes only the complete residual row scheduling.
    These four options default to their original routes below 8,192 environments
    and enable their large-batch routes above that size. Set explicit values
    to compare scheduling. All options are fixed when building the scene.

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

    ``update_mode="every_step"`` observes each scene step, including its maximum over physical substeps.
    ``update_mode="manual"`` saves observation work between requested samples. Call the link's
    ``request_stress_update()`` before each scene step to observe. Getters retain the latest sampled values between
    requests, so callers must track their sample times. State changes invalidate observations as usual.
    """

    mesh: PathType
    output_mode: Literal["max", "full"] = "max"
    update_mode: Literal["every_step", "manual"] = "every_step"
    young: PositiveFloat = 1e10
    poisson: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.3
    density: PositiveFloat = 2000.0
    contact_radius: PositiveFloat = 0.006
    contact_model: Literal["pressure", "intrinsic", "hertz"] = "pressure"
    hertz_reduced_modulus: PositiveFloat | None = None
    load_transfer: Literal["patch", "resolved"] = "patch"
    intrinsic_integration: Literal["surface_fan", "anchor_fan"] = "surface_fan"
    intrinsic_support: Literal["contact", "pair_union"] = "contact"
    intrinsic_load_partition: Literal["componentwise", "body_source_prior"] = "componentwise"
    cooperative_tractions: StrictBool = True
    surface_fan_frontier: StrictBool = True
    warm_start_tractions: StrictBool = True
    contact_block_size: PositiveInt = 16
    surface_edge_subdivisions: PositiveInt = 8
    source_projection_limit: PositiveFloat = 1e-4
    source_geometry_tolerance: PositiveFloat = 1e-8
    source_uncertainty: Annotated[ValidFloat, Field(ge=0)] = 0.0
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
    cooperative_balance: StrictBool = True
    contact_wrench_reuse: StrictBool | None = None
    face_parallel_scatter: StrictBool = True
    scatter_tasks_per_env: PositiveInt = 32
    contact_moment_reuse: StrictBool | None = None
    inverse_max_bytes: PositiveInt = 64 * 1024 * 1024
    inverse_precision: Literal["64", "32"] = "64"
    surface_inverse: StrictBool = True
    packed_surface_loads: StrictBool = True
    packed_block_size: Literal[0, 256, 512] | None = None
    coalesced_residual: StrictBool | None = None
    fused_pipeline: StrictBool = True
    fused_rigid_post: StrictBool | None = None
    cached_peak: StrictBool = True
    cached_face_bounds: StrictBool = True
    inverse_corrections: Annotated[int, Field(ge=0, le=8)] = 2
    preconditioner: Literal["diagonal", "block"] = "block"

    @classmethod
    def hertz(cls, *, mesh: PathType, quadrature: PositiveInt = 8, **options) -> Self:
        """Estimate Hertz support and friction stress from current rigid contacts.

        This route currently requires a locally planar material source, supported
        half-space geometry and a representable force line. Sixteen radial
        intervals retain vector friction memory under changing normal and shear
        loads. Free couples and ambiguous history correspondence are rejected.
        The finite-body correction includes every material boundary, including
        cavity faces omitted from the asset's selected contact surface.
        A missing reduced modulus uses this material's
        E/(1-nu**2), declaring an infinitely stiff opposing material.
        """
        return cls(mesh=mesh, contact_model="hertz", quadrature=quadrature, **options)

    @classmethod
    def dpr(cls, *, mesh: PathType, **options) -> Self:
        """Directional Patch Recovery, including the supplied tangential force."""
        return cls(mesh=mesh, contact_model="pressure", **options)

    @classmethod
    def scr(cls, *, mesh: PathType, **options) -> Self:
        """Surface-Constrained Recovery with the complete body-source-prior route."""
        return cls(mesh=mesh, contact_model="intrinsic", intrinsic_load_partition="body_source_prior", **options)
