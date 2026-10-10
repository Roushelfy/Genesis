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

## Low-resolution optimization and selection, 05:00–07:30 UTC

Kept the complete L1 hollow shell, FP64 scene, Q10 surface sampling, finite
pad law and full-row residual acceptance throughout performance selection.
Cooperative native sparse solves, an exact sample grid and a bounded shared
inverse replaced the measured serial/full-scan bottlenecks. Direct paths
stopped allocating unused Krylov buffers. All production numerical work,
including inverse construction and optional history, remains Quadrants.

Resident batches were swept through B=2048. B=1024 is the largest accepted
whole workload. B=2048 encounters an unrepresentable finite-footprint wrench
at tick 632; the independent CPU mapper rejects the same fixture, now a
regression test. No rejected rate or silently enlarged footprint is used.

Measured optional four-column native history with complete residual rejection
and independent reset: 52.1% sampled prediction hits, but 31,471 versus
34,487 transitions/s and 318.6 MB extra storage. Selected history zero.
Mixed inverse storage exposed a correction-sign bug during code review.
Corrected `rhs-Ku` updates add the inverse residual. Two-correction numerical
tests pass without fallback; actual strict live mixed storage still reaches
only 12,783 transitions/s with many fallbacks, so FP64 remains selected.

Pressure was the next measured hotspot. Native device packing and a warp per
active contact cooperate on both Gram and constrained evaluations. The same
Newton/Armijo pressure law is retained. Final pressure wall time falls from
7.49 to 1.38 ms, and final three-repeat live throughput rises from corrected
serial pressure's 34,047 to 42,867 transitions/s (+25.9%). CPU and explicitly
selected serial pressure remain validated. Final numerical source is
`dbfdf20d`, pushed before measurement; its all-16 GPU suite passes in 162.77 s
and 14 CPU numerical cases pass in 73.07 s.

A checkpoint test initially assumed saved stress remained valid after rigid
restore. Ordinary state-change notices invalidate auxiliary peaks/history;
the test now verifies that contract and the complete final suite passes.
The earlier failed test log is retained with this explanation.

## Completed performance milestone, 07:30 UTC

Three-repeat aggregates at B=1024: rigid 108,231; policy-rigid 105,571;
frozen-input recovery 75,845; live 42,867; policy-live 42,515 environment
transitions/s. Final B=8 live is 1,200.376 versus normalized serial baseline
57.464, a 20.89x improvement. Every live/policy repeat includes 1,024
independent resets; policy includes device observations and seeded inference.
The actual warmup has bilateral lift/hold for 940/1,024 environments in both
live scopes; unsuccessful grasps remain counted. A separate every-step audit
checks 4,300,800 environment steps across both scopes without invalid results.

Final stress storage is 274,817,437 bytes across 82 arrays. Stage profiling,
intrusive CPU/CUDA trace, per-buffer memory, build/JIT timings, numerical
errors, rejected candidates and provenance are recorded in
[NATIVE_RESULTS_20261010.md](NATIVE_RESULTS_20261010.md) and the hashed evidence
manifest. Full traces and large RHS probes remain in the task data directory.
Shared inverse application (7.29 ms), scatter (2.86 ms) and ordinary rigid/
host scheduling remain the measured bottlenecks. The applicable low-mesh
performance work is complete; physical convergence remains a separate later
study and was not rerun during this native phase.
