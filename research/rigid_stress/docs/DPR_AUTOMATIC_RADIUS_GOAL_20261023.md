# Goal: inexpensive automatic radius for the existing DPR

Latest user correction, 2026-10-23: keep the mature DPR and its prefactorized
linear FEM. Add the cheapest useful estimate of the contact footprint. Stop
developing the analytic-particular / boundary-traction / combined-stress route.
This correction supersedes conflicting goals and historical experiments.

## Immediate goal: maximize native automatic-DPR throughput

Latest user follow-up, 2026-10-23: optimize this path as far as practical.
The feature is now integrated. The next deliverable is substantially better
measured complete-scene throughput for cheap automatic radius + Hertz pressure
+ original DPR/prefactorized FEM, with maximum and full stress output.

Keep low-resolution Franka/hollow-egg iteration and the existing load model.
Only fixed shape/material may be preprocessed; contact locations, symmetry,
convexity and normal/friction forces remain variable. Production stays inside
the existing rigid solver, entirely in Quadrants. Continue the original DPR
path; analytic stress enrichment and boundary correction remain stopped.

### Optimize the measured bottlenecks

Use the current profile to guide implementation, and pursue other material
bottlenecks it reveals. These are leads, not a prescribed architecture:

- Use analytical pressure moments for contained planar Hertz disks to integrate
  P2 nodal loads directly where valid. Keep correct clipped/multiface handling
  for supports crossing mesh edges/vertices and nonconvex surfaces.
- Build compact intersecting-face candidates once per contact update and reuse
  them through fitting and scatter. Hoist contact/face projection, containment,
  fan/clipping geometry and immutable quadrature work out of sample loops.
- Extend pressure/Gram/P2 moment reuse to refined supports where valid, so
  accepted pressures do not require a second full integration for scatter.
  Eliminate proven-zero fan ranges and duplicate geometry/weight evaluations.
- Optimize the remaining constrained-fit path: reuse invariant integration
  data, avoid unnecessary Newton/line-search evaluations and allow coherent
  early exit after convergence. The current fixed upper-bound loop and
  repeated all-face traversal deserve inspection.
- Reuse collision witnesses, fixed-shape curvature and continuous-contact local
  geometry. Keep current force-dependent radii fresh; invalidate caches for
  release, feature/geometry changes and partial resets.
- Tune parallel granularity, active-work packing, fusion, kernel launches,
  workspaces, allocations, device traffic and batch size. Evaluate precision
  and dense/batched FEM options if profiling makes them worthwhile; validate
  numerical error and count the complete end-to-end cost.

A smaller integration rule is allowed when independent load/stress controls
establish equivalent accuracy. Do not freeze the expensive q16 algorithm as
a requirement, but retain the actual q10 failure and validate replacements.
Exact fast paths should preserve the physical support, pressure law, supplied
normal/friction force and moment, original FEM and runtime acceptance.
Keep unsupported geometry and edge-surrogate statuses explicit. Contact-region
regrouping changes force-dependent radii and needs separate model evidence;
it is not an equivalent performance-only substitution.

### Demonstrate the improvement

Use [the current implementation report](AUTOMATIC_DPR_IMPLEMENTATION_20261023.md)
and its retained evidence as the starting baseline. At engine source
`40775c1d0`, B64 full scene times on RTX PRO 6000 Blackwell are 1.734 ms
rigid, 2.671 ms manual DPR, 5.339 ms radius-only DPR and 4.564 ms Hertz-profile
DPR. The separate loaded-snapshot profile places Hertz-profile pressure
integration/fit at 1.697 ms, scatter at 0.380 ms, radius estimation at 0.196 ms
and FEM solve at 0.195 ms; intrusive stage times are not additive scene costs.
These are starting measurements, not throughput targets or final scalability.

Compare pure rigid, original manual DPR, radius-only DPR and automatic Hertz
pressure on matching geometry/material/FEM, controller/trajectories, source
loads, update cadence and output semantics. Measure small batches and search
for the throughput-optimal useful batch; do not force B1024 or stop at B64.
Retain the four-condition baseline and add a compact set of diverse changing
contacts so tiling alone is not mistaken for broad workload coverage.

Report paired, repeated, profiler-off scene timings; separate module, estimator,
device/launch and pressure-integration/fit/scatter/FEM breakdowns. Include max
and full output, env-step/s, added cost relative to rigid/manual DPR, memory,
repeat variation and invalid-step coverage. Full fields remain on the GPU.
Keep detailed research exports and intrusive audits outside primary timing.
Explain regressions or declined candidates and use the final profile to
identify remaining limits. A local kernel speedup alone is not completion.

### Numerical scope and completion

Preserve original wrench/FEM residual checks, normal/friction handling,
nonconvex sheet identity, batch isolation and partial resets. Validate affected
fast paths with independent disk/sector P2 moments, same-discretization stress
controls and focused CPU/GPU checks. Use established accuracy-based checks;
independent runs need not be bitwise identical. Retain the previous bitwise
audit failure separately without rewriting it as a pass.

Keep the reproducible 130-frame baseline. Add inexpensive numerical coverage
of moving/asymmetric contacts, contact-count/load/friction changes, release,
recontact and reset; extend a low-batch run through the grasp/release cycle
where supported and report remaining failures. Distinguish supported-window
throughput from complete-cycle coverage. Broad high-resolution convergence,
Code_Aster installation and new coupled contact references are not prerequisites
for this performance phase; retain existing physical-error limitations and do
not promote numerical consistency into physical qualification.

Deliver the optimized native feature, usable commands and recommended compact
settings on this branch. Continue through the material profile-supported
opportunities rather than stopping after the first local improvement; document
remaining opportunities, measured tradeoffs and practical stopping reasons.

## Intended runtime path

Current rigid contacts -> cheap Hertz/contact-size estimate -> finite contact
pressure -> existing DPR load transfer -> existing prefactorized FEM -> existing
max/full observations.

Only rest shape/material are fixed. Keep changing, asymmetric convex/nonconvex
contacts, supplied normal and friction forces, batching, environment isolation
and partial resets. The feature stays inside the Genesis rigid solver and all
production numerical work stays in Quadrants.

The target is Hertz pressure feeding the original prefactorized FEM through
DPR's existing load-transfer path. Implement the contact-size estimate as a
small addition to mature DPR, retaining its friction force handling, elastic
operators and observation semantics. Keep manual-radius DPR as the baseline.
Distinguish the effect of radius estimation from any change of pressure profile:
if the original DPR profile is retained in a radius-only control, call that
Hertz-sized DPR, not exact Hertz pressure. Both controls use the same FEM;
neither needs a new stress-recovery system.

## Reuse the work already completed

The original contact radius field and setter already accept per-contact,
per-environment values. The native Hertz lookup already estimates ellipse axes
from supplied normal force, physical reduced modulus and relative curvature.
Current collision witnesses, immutable geometry and useful local-query
acceleration are reusable. Prefer the simplest adequate estimate, including
the circular Hertz cube-root relation, before introducing extra machinery.

Use fixed-shape curvature preprocessing and available current collision features.
Reuse local geometry under continuous contact, with proper release/reset and
feature-change handling. Do not assume contact locations or symmetry are fixed.
State the behavior where Hertz curvature assumptions do not apply; a cheap,
declared estimate or unsupported status is preferable to an expensive default
contact solve. Preserve the established DPR acceptance checks and distinguish
its failures from estimator failures.

Do not transplant analytic-enrichment-specific half-space, pressure-centroid,
point-source or friction-history requirements into the original DPR. Preserve
actual contact provenance and measured force/moment behavior without silently
changing loads or loosening numerical checks.

## Stop the unrelated runtime expansion

At a safe boundary, retain current outputs and source identities, then stop
scheduling further analytic particular-field, whole-boundary traction-correction,
combined-field/probe, and sharp-edge singular-maximum development for this goal.
Standalone friction-history solvers, coupled contact references and two-face
pressure research are not prerequisites for automatic-radius DPR.

Keep their reproducible evidence as historical/optional research. Do not delete
useful shared code, hot-edit running captured sources, or require their physical
qualification before delivering the inexpensive radius path.

## Validate the intended feature and its cost

Compare manual-radius DPR with automatic-radius DPR on the same elastic mesh,
contact inputs, output modes and sampling cadence. Separately identify any
optional change from the original pressure profile to Hertz pressure. Use
compact CPU physical controls, preferably the requested Code_Aster contact
reference when available, to assess the radius/load model and complete FEM
stress. Do not import the accuracy of analytic enrichment or a fine elasticity
mesh as the accuracy of native coarse DPR.

Prioritize low-resolution Franka/hollow-egg implementation and throughput now.
Measure estimator-only added cost, complete stress-module cost and paired
pure-rigid/manual-DPR/automatic-DPR max/full scene costs. Keep numerical
conformance, runtime validity and changing normal/friction contacts; keep
research export and detailed audits outside production timing. Use focused
tests and useful batch sizes rather than new broad convergence campaigns.

Deliver one directly usable entrypoint in this task branch, recommended compact
configuration and measured accuracy/throughput tradeoffs. An isolated lookup,
monkeypatched research callback or separately pinned engine snapshot is not the
completed feature.

## Current evidence and gaps

The bullets below record the starting state when this goal was set. The
integrated option, matched timing, numerical checks and remaining physical
limitations are now documented in the
[implementation report](AUTOMATIC_DPR_IMPLEMENTATION_20261023.md).

- Supplied-region native Hertz lookup costs about 21 microseconds on the measured
  GPU, excluding curvature lookup, grouping, transfer and FEM.
- The research input adapter has current force/curvature/grouping controls but
  is not an integrated automatic-radius DPR option.
- A research Hertz-pressure -> P2 -> original FEM bridge exists. Its optimized
  B1024 max/full times are 11.749/11.885 ms versus 7.349 ms rigid, but only
  52.77% of its loaded observations pass its research gates. It changes the
  pressure profile and is not the required radius-only baseline.
- The exact automatic-radius + original DPR path has not yet been delivered or
  measured. The later roughly 120x overhead belongs to the abandoned enrichment
  route and is not a cost estimate for this feature.

Reuse the [existing component and bridge evidence](HARD_CONTACT_ESTIMATES_20261020.md).
Follow the existing storage policy and preserve prior baselines/failures.
