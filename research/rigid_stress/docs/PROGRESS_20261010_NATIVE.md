# Native Quadrants rigid stress recovery: iteration journal

## Direction change, 2026-10-10 04:21 UTC

Synced user brief commit `efc21e37`. Production numerical computation must now
use Quadrants throughout and integrate with the existing rigid solver. The old
external-library prototype remains a reference. High-resolution physical
convergence is deferred until the low-resolution performance milestone.

Cancelled only own old replay Slurm job `703330` after two accepted 1,200-step
repeats (107.8027 and 107.7554 recovery transitions/s). No final three-repeat
aggregate or stage pass is claimed. Completed rigid/live/policy measurements and
both 60-frame CPU oracles remain archived. The old frozen level-6 configuration
is historical and will not be the native default.

Inspected upstream PR #3372 at `c7d29e08f5abd049b95ec187591f64c33553c18a`,
still an open draft. Borrow integration and profiling patterns rather than a
new solver registration. Quadrants sparse-library wrappers internally delegate
to CUDA sparse libraries; production recovery will instead use owned Quadrants
kernels and typed arrays. Independent offline SciPy FP64 validation is separate.

## Initial implementation

Start with full-shell level 1 or 2, explicit finite footprint law and shared
P2 sparse operators. Read contact force/state at the same pre-integration
substep. Add link configuration, ordinary observations, partial-reset
invalidation and errno failures. Measure native iterative/preconditioned solve
before choosing further scheduling, reuse and precision optimizations.

## First native operator/recovery checks, 04:36 UTC

Full-shell level 1 has 810 P2 nodes, 2,430 DOFs and 480 tetrahedra.
Owned Quadrants kernels assemble shared block-CSR stiffness/consistent mass,
rigid modes, relief and six centrifugal fields. Mesh connectivity is processed
symbolically in Python; no external numerical solver runs in production.
A symbolic minimum-degree node order feeds a shared native block-LDL factor.
Every physical factor value and batched solve is computed in Quadrants.

GPU operator equality and five-column recovery checks pass against the
independent SciPy/Numba FP64 oracle, including zero load, asymmetric nodal loads
and angular velocity. Full residual maxima are 4.281e-12, 7.461e-12, 6.604e-12
and 5.194e-12 N for the four nonzero columns. These are initial numerical checks,
not live-contact feature acceptance or physical convergence. Raw logs are in
`evidence/20261010-native/raw/`.

The first block-diagonal PCG probe reaches 4,000 iterations without complete
residual acceptance (5.10e-6 to 1.72e-5 N); its log is retained. This motivates
the sparse direct baseline and stronger preconditioners. The PCG path explicitly
marks unresolved solves invalid. Contact mapping, native lifecycle integration,
profiling and end-to-end optimization remain active.

## Native contact/lifecycle integration, 04:52 UTC

The rigid solver now resolves contact forces, recovers stress from that same
pre-integration state and then performs its ordinary integration. Link opt-in,
device observations and selected-environment invalidation use the existing
solver lifecycle. No research callback runs in the engine. Unsupported joint
supports, nonstandard couplers and rolling/torsional contact moments are
explicitly rejected. Disabled scenes retain the existing fused rigid kernel.

Quadrants surface quadrature, force-line anchoring, nonnegative constant-ratio
pad pressure, P2 scatter and force/moment checks match six independent CPU
patches to the specified same-mesh budgets. Three environments with distinct
contact onset pass actual 1- and 4-substep scenes, and resetting environment 1
preserves environments 0 and 2 bitwise. These are focused numerical checks;
the printed scene FPS from these tests includes diagnostic synchronization
and is not a benchmark rate.

Added the normal Panda example with small hollow-mass/inertia URDF assets,
changing footprints, friction, slip/release phases and independent resets.
Detailed baseline profiling is running on level 1. The first profiling
attempt failed because engine source was edited during delayed compilation;
the rerun freezes its source. Baseline measurements must precede optimization.
No new high-resolution physical convergence workload has been run.
