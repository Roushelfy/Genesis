# Native componentwise recovery on declared pair support

This implements the constructive candidate in
[the shared-support checkpoint](NATIVE_SHARED_SUPPORT_PROGRESS_20261011.md)
on source base `1977fca50c51ff2011b83deda8a06316ff5ab5b0`.
It repairs the two saved plane-contact fixtures without altering their forces,
application points, radii, friction or final numerical budgets. The active
nonconvex goal remains incomplete: broader source coverage, new-law profiling,
optimization and repeated large-batch four-scope measurements remain required.

## Public law and native implementation

`RigidStressOptions.intrinsic_support="pair_union"` explicitly permits each
original contact's vector traction to use all actual finite disks belonging to
its collision geom/pair. The existing `"contact"` scope remains available.
Both the Franka example and benchmark expose `--intrinsic-support` and retain
the existing contact, recovery and full/max APIs.

Each disk retains its original intrinsic path, radius, squared-coordinate Q10
surface fans and material-owner normal. The prior integration measures form a
mixture over the shared support: overlapping source profiles add, and the
combined positive weights are normalized for each original wrench component.
This declares a pressure-distribution prior; it does not infer a unique
physical pressure from rigid point forces. Disconnected actual islands may
share the declared pair load, but no additional surface adjacency is created.

Each component retains its own original force, free moment, application point,
anchor and physical-radius moment scale. Every destination uses its actual
source friction coefficient. The final commit independently recomputes each
original wrench and every local circular cone on all its destination samples.
Summing component tractions preserves the complete input pair wrench. No
aggregate-only acceptance replaces the original component checks.

Production geometry/profile storage remains per source. Six-dimensional dual
fitting iterates the shared source ranges without allocating a replicated
sample cloud per target. Nodal scatter sums the component tractions at each
destination before evaluating its signed P2 basis and accumulating nodes.
All numerical operations remain Quadrants.

The stream sorts actual raw indices by source geom/pair and packs whole groups
into blocks without padding or changing original indices. The workspace is at
least the collider's applicable per-pair capacity. In the live GJK/patch runs,
requested block size4 resolves to size8; the benchmark records both. Separate
geom/pair identities remain separate, and valid zero-force sources still supply
their finite geometry. An unresolved source invalidates its complete group.

## Same-mesh and lifecycle validation

The unchanged contact suite passes 47 CPU and 47 CUDA cases. Two added saved
pair fixtures cover tick710 and tick1160, each with three environments. The
first retains both original measured coefficients; the other environments
exercise distinct destination coefficients `mu*[1.15,1.4]` and `[0.85,1.2]`.
The independent CPU profile/geometry, convex-dual plus root-polish fit, signed
nodal loads, full equilibrium including gauge rows, full tensor/von Mises and
global unaveraged peak all retain their existing acceptance budgets.

A separate two-case native lifecycle/data-enumeration module passes CPU
(57.30 s) and CUDA (59.77 s). Its stream test interleaves
six identified source pairs in twelve raw slots, compares requested blocks
12/2/7, releases and restores one environment, invokes partial reset, keeps a
zero-load support source, and injects one bad source only in env0. Its complete
pair fails with the existing error/NaN observation contract; the other
environments remain valid. Restoring sources and resetting env0 restores its
accepted recovery. Tests use the actual begin/recover/accept/reset functions.

The first lifecycle test controls are retained. One used a precision option
that belongs to URDF rather than Mesh. Another assumed reset makes the scalar
valid flag false; the existing reset contract instead returns a valid zero
maximum and NaN full fields until recovery. A third omitted the production
acceptance call and tested the equilibrium solver's valid flag as a contact
acceptance flag. The corrected test exercises the complete production chain;
no existing acceptance threshold or assertion was weakened.

Native data enumeration also exposed a missing `DataKind.SCRATCH` declaration
on the nested shared-support struct. Numerical cases and live trajectories
had already passed, but the rigid-data audit and benchmark memory report
failed during enumeration. The declaration is now present and has a direct
data-enumeration regression. These failed integration controls are retained;
numerical acceptance alone did not establish complete feature integration.

The complete B16/seed623001 trajectory also passes a rigid-data isolation
audit. At 33 pre-integration recovery calls, all 213 non-stress checkpoint
fields have identical arrays and byte hashes before/after recovery. The run
completes 1280 steps and 19 saved-state partial resets. This directly checks
that the auxiliary recovery does not alter the rigid solver state.

## Complete live trajectories

Both trajectories use the same full hollow low shell, FP64, Q10, s8 surface
paths, squared integration, timestep0.01s, one substep, mesh64 source and
GJK with contact patches. They check every original contact wrench every step
and include all scheduled saved-state partial resets. Each has nine independent
CPU complete equilibrium/full-field/peak captures. The oracle receives the
native assembled load in these live captures; the separate saved-fixture
regressions independently fit the load itself.

| Batch / seed | Complete steps | Effective env-steps | CPU full-field comparisons | Resets | Maximum evaluations | Original force / moment normalized maxima |
|---|---:|---:|---:|---:|---:|---|
| 16 / 623001 | 1280 | 20,480 | 144 | 19 | 59 | 8.999734e-9 / 8.947480e-9 |
| 32 / 623101 | 1280 | 40,960 | 288 | 37 | 48 | 8.995334e-9 / 8.984473e-9 |

All 61,440 live transitions pass, including the former tick710/tick1160 failure
frames. Complete residual maxima are `1.39351e-11 N` and `2.33013e-11 N`, inside
each original absolute/relative budget. Final wrench acceptance remains
`1e-8`, complete residual `max(1e-11 N,1e-8||rhs||)`, and stress
`rtol=1e-4,atol=1e-3 Pa`. Internal fitting remains `9e-9` and 128 evaluations.
The runs include 877/2012 contact-slot changes and actual combined coefficients
from about0.4243 to0.8281. No failed or slow environment is excluded.

These are intrusive validation runs, not throughput measurements or universal
source acceptance. They do not establish physical mesh convergence or full RL
training throughput. Hardware is RTX PRO6000 Blackwell Server Edition,
driver595.71.05, Xeon Platinum8562Y+, container1_26, Genesis1.4.3 and
Quadrants1.3.3. Source hashes before/after every closed run match.

## Matched scheduling profile

An experimental Quadrants warp-per-component fit passes 15 CUDA cases,
including both saved pair fixtures, stream lifecycle, surface-fan and vector
traction checks. It remains outside production pending complete online and
end-to-end measurements. A B32/seed623101 snapshot after 600 varied live
steps uses the same saved condition bank, law, samples and precision on both
paths. Twenty profile repetitions measure the intrinsic pipeline at
157.95 ms with scalar fitting and 29.25 ms with the warp candidate.
Kernel profiling identifies scalar/warp fitting at 134.245/5.707 ms,
original-component commit at 9.904/9.895 ms, and nodal scatter at
9.981/9.993 ms. The repeated commit/scatter sample scans are now actionable
hotspots. Snapshot preparation preserves original application points and
retains the unchanged wrench, equilibrium and peak checks on both paths.
These synchronized snapshot profiles establish neither live throughput nor
a performance plateau.

## Reproducible checkpoint

The [publication manifest](../evidence/20261011-nonconvex/native-pair-support-checkpoint-v1/manifest.json)
contains 574 artifacts (18,539,219 bytes), including all source epochs,
accepted and failed controls, complete trajectories, CPU comparisons,
rigid-data audit and experimental matched profiles. Every artifact and its
data-side original passed SHA256 verification on an allocated CPU node.
Manifest SHA256 is
`d96e8ed82a4d64a9d4493a7a0f94b686620210eb84671feaa2d268304eaffb2b`.
The numerical 47-case suites precede the Python-only nested DataKind repair;
the final two-case lifecycle/enumeration suites, rigid audit and profiles
include that repair. These source epochs are explicit in the preserved hashes.

## Remaining work

Keep this new law fixed when rebuilding the baseline. Evaluate shared-source
warp fitting, scatter/commit fusion, profile candidate scheduling and memory
layout, then remaining applicable operator/history/precision/graph candidates
by end-to-end evidence. Extend actual primitive/SDF/decomposition provenance,
continuous failure diagnostics and source/refinement coverage. Retain the
no-feedback rigid contract and complete lifecycle checks. Only after robust
acceptance rebuild repeated rigid-only, recovery-only, live and policy rollout
measurements at1024/2048/larger batches with outcomes, tail work and memory.
