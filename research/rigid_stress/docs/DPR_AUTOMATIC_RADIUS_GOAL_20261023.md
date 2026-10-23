# Goal: inexpensive automatic radius for the existing DPR

Latest user correction, 2026-10-23: keep the mature DPR and its prefactorized
linear FEM. Add the cheapest useful estimate of the contact footprint. Stop
developing the analytic-particular / boundary-traction / combined-stress route.
This correction supersedes conflicting goals and historical experiments.

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
