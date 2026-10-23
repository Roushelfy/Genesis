# Historical goal: cheap hard-contact stress estimates and native throughput

**Superseded where conflicting, 2026-10-23:** the active target is
[cheap automatic radius for the original DPR](DPR_AUTOMATIC_RADIUS_GOAL_20261023.md).
Keep the existing prefactorized FEM. Stop analytic particular-field,
boundary-traction and combined-stress expansion. This older document is retained
for provenance and reusable component/optimization evidence; it does not add
additional enrichment, friction-history or coupled-reference prerequisites to
the minimal automatic-radius DPR delivery.

Latest user direction, 2026-10-20: prioritize inexpensive Hertz/thin-shell
contact approximations and estimates. Do not assume or develop a deformable
gripper pad. Avoid expensive automatic contact-region solves unless they
demonstrate very accurate physical stress results and a worthwhile measured
accuracy/cost advantage. This direction supersedes earlier requirements to
finish the coupled spring-pad pressure model or its public integration.

The target remains a rigid Franka gripping a fixed-shape hollow egg, with
changing/asymmetric contact positions and normal/friction loads. Shape/material
preprocessing is allowed; contact layout, symmetry and convexity are not fixed.
Production belongs to Genesis's existing rigid solver and uses Quadrants,
with maximum/full output, batching, environment isolation and partial resets.

## First priority: useful inexpensive contact estimates

Implement and compare cheap estimates before extending a coupled contact solver.
Use Hertz where its assumptions are adequate, and inexpensive thin-shell or
finite-thickness corrections where shell compliance matters. Analytical formulas,
small tables/interpolants or validated low-dimensional surrogates are appropriate.
The implementation agent should choose the simplest useful method from evidence;
do not turn this into another broad framework or a large option matrix.

For a smooth locally spherical nonadhesive elastic contact, the basic Hertz
baseline uses

```text
a = (3 Fn R* / (4 E*))^(1/3)
p(r) = 3 Fn/(2*pi*a^2) * sqrt(1-r^2/a^2),       0 <= r <= a
1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2
1/R* = 1/R1 + 1/R2                             # two convex spheres
```

Use both principal relative curvatures and orientation when an elliptical
approximation is needed. Treat thin-shell bending/thickness and finite geometry
as approximation questions to test; no universal shell radius correction is
assumed. Rigid dynamics does not make the observed material infinitely stiff.
Do not infer a physical modulus from a numerical penalty or collision margin.

Predict pressure as well as support. A radius inserted into the existing
Gaussian kernel is not automatically Hertz contact. Determine physical contact
regions before allocating their loads; splitting collision samples must not
arbitrarily change predicted support. Preserve source/wrench and actual boundary
provenance, including legitimate inner/nonconvex contacts. Keep unresolved
geometry or regimes visible rather than inventing a radius.

Retain friction and its relevant history. A normal Hertz footprint alone does
not determine tangential stress. Begin with a cheap explicitly declared
tangential approximation and quantify its error; use richer history only where
it yields useful accuracy at acceptable cost.

## Validate accuracy and cost in parallel

Use a compact CPU study of hard-gripper contacts: an analytical Hertz control,
the hollow egg, and useful asymmetric/nonconvex and changing normal/friction
cases. Compare pressure/support and damage-relevant global tensile/von-Mises
maxima as well as complete fields, membrane/bending and peak locations. State
where local curvature/half-space or shell approximations are adequate.

Validate new hard-contact physical claims against matched independent contact
references. Prescribing the same Hertz pressure to two FEM solvers only verifies
their elasticity; it does not independently validate Hertz pressure/support.
Do not revive an unresolved point-load peak sweep or require a large convergence
campaign before low-resolution implementation and performance iterations.
Use targeted reference effort where it can distinguish the candidate estimates.

Exploit fixed-shape preprocessing, cached curvature/material data, table lookup
and smooth contact history. Test whether displacement already computed during
stress recovery can cheaply assess or correct the contact estimate. Extra
iterations must earn their cost; do not add an838-coordinate Newton-PCG pressure
solve per environment per step by default.

Integrate the useful cheap estimate into native Quadrants and benchmark actual
moving Franka/egg scenes. Profile geometry/region estimation, traction transfer,
elastic recovery, max/full reduction, history, dispatch, synchronization and
copies. Measure GPU-resident module work and paired pure-rigid/max/full scene
throughput over useful batch sizes. Include prediction/preprocessing/memory and
failure behavior. Keep every-step and sparse output distinct. Existing manual
radius and resolved-pressure routes remain labelled controls; resolved transfer
alone does not estimate unknown collider pressure.

## Immediate priority: reduce contact-input overhead

The user explicitly requests profiling and optimization of the current cheap
input path. Fixed-shape curvature lookup should be inexpensive. Do not treat
the research adapter's repeated geometric reconstruction as an inherent Hertz
cost or as a production performance limit.

The [direct-normal-root pilot](HARD_CONTACT_ESTIMATES_20261020.md#direct-planar-normal-roots),
executed before this update, measures4.025ms per B4 scene step versus2.249ms
for paired pure rigid, approximately1.777ms added. It includes contact-input
preparation and audits but no recovered elastic stress. Only1381/2808 loaded
environment steps pass geometric checks. These are preserved diagnostic
measurements, not usable stress throughput or a forecast of large-batch cost.
The separate approximately21us supplied-region Hertz lookup is a component
measurement on a different workload; it does not attribute the1.777ms.

First profile the actual added work, then implement and measure all applicable
optimizations below. These are directions to investigate, not a prescribed
architecture or a demand to keep a variant that fails to help.

| Direction | Practical implementation opportunities |
| --- | --- |
| Reuse rigid collision results | Carry current geom/pair, material witness and available face/vertex/barycentric provenance into the estimator. Query cached curvature at an identified feature instead of rediscovering it with a full mesh scan. Measure a declared witness-based cheap estimate where appropriate; changing the physical center model requires its own error comparison. |
| Exploit continuous contact | Cache each region's material feature, center, frame and region identity. Try the previous face and its adjacency first, updating barycentric coordinates and curvature locally. Search more broadly for new contacts, invalid candidates, topology/feature transitions or discontinuous motion. Report cache hit rate, local work, fallback frequency and tail cost. |
| Precompute fixed geometry | Cache face/vertex curvature, barycentric transforms, normal-field inverses, adjacency and useful spatial/normal search acceleration once per shared shape. Reuse across environments; benchmark preparation time and memory separately. Avoid rebuilding shape-only data or repeatedly inverting fixed small matrices per query. |
| Fuse work and reuse intermediates | Reduce repeated coordinate transforms, contact-array passes, temporary writes and Quadrants dispatch between association, provenance, feature/curvature query, Hertz estimation and reduction. Keep production inputs and outputs GPU-resident; reuse buffers. Put detailed research auditing/readback in separate runs while retaining required runtime acceptance. |
| Parallelize grouping and load reduction | Profile the current per-environment serial and worst-case quadratic source comparisons. Test appropriate feature/region keys, segmented or small-group reductions and parallelism over contacts as well as environments. Operate on actual active counts where useful; active compaction must earn its own overhead. Preserve separate physical regions and force/moment under source splitting. |
| Tune useful batch sizes | Sweep from small batches toward the best valid aggregate throughput and practical memory limit. Measure homogeneous replay separately from independently evolving Franka environments. Amortize fixed costs and inspect utilization/layout/dispatch; B4 timings alone cannot predict RL throughput. |

Only shape/material is fixed. Temporal caching must preserve arbitrary changing,
asymmetric convex/nonconvex contacts and normal/friction loads; contact positions,
symmetry and contact count are not preprocessing assumptions. A normal-direction
lookup alone must not jump to a different nonconvex surface. Preserve material
surface/component identity, current-pair and finite-face checks, with a controlled
search on cache failure. Invalidate the affected environment's geometric cache
and friction history appropriately on release, reset, pair/feature change and
other invalidating events. Shared immutable shape data may persist. Reusing
geometry does not justify discarding frictional memory or retaining stale loads.

Separate the following phases in a deliberately instrumented run: contact
association/provenance; geometric search or cached local update; curvature/frame
evaluation; region grouping and force/moment reduction; normal/tangential law
and lookup; traction transfer; elastic recovery; max/full output; history and
fallback work; and research-only audit. Account for CPU dispatch, synchronization,
copies and GPU launches, noting that one Quadrants kernel call may launch several
GPU tasks. Distinguish inclusive and exclusive timing and do not sum overlapping
or independently sampled phases into a fabricated total.

Use matched warmed inclusive wall-clock runs as the primary performance result.
Compare current baseline and optimized variants on the same inputs and actual
low-resolution moving scene, with pure rigid/max/full and normal/friction cases.
Retain equivalent validity and numerical outputs for implementation-only changes;
separately label and validate any model approximation. Include new contact,
continuous movement, release, invalidation, nonconvex/disconnected regions,
source splitting, environment isolation and partial reset in compact checks.
Batch mode, precision and caching changes need focused numerical conformance,
not an unrelated high-resolution FEM convergence campaign.

Publish phase costs, inclusive added cost and aggregate env-step/s, scaling,
memory, cache/fallback statistics, validity/failure counts and numerical
differences, with source/configuration identities and runnable commands. State
what improved and what did not help; do not promise a speedup factor before
measurement. Distinguish input-only, complete stress-module and complete scene
results. A rejected or omitted contact/environment does not count as a useful
stress update. Deliver the useful optimized path inside the existing rigid
solver, entirely in Quadrants, rather than closing with a research-only lookup
microbenchmark. Continue compact hard-contact accuracy work alongside this
low-resolution throughput priority.

## Expensive models require strong physical evidence

The coupled pressure/Newton-PCG work is an optional offline reference or research
candidate, not the default runtime route, automatic fallback or a prerequisite
for delivering the cheap model. Do not spend the next phase expanding its pad
geometry, material configuration, integration or test campaigns.

Promote an expensive hard-contact method only after it shows very accurate
complete and peak stresses against an independently qualified reference on the
same physical problem, with a clear accuracy advantage over cheap estimates
beyond reference uncertainty. Predeclare a meaningful accuracy target and report
measured incremental module/scene cost. Numerical CPU/GPU agreement, a tiny
equilibrium residual, or agreement with another implementation of the same
discrete pad model does not satisfy this condition. If that benefit is absent,
retain the work as optional reference evidence and prioritize throughput.

## Error judgments and practical acceptance budgets

Keep numerical solver convergence, permitted load-model discrepancy and physical
stress accuracy separate. Do not blindly require1e-8 reconstruction accuracy,
or loosen existing assertions until a case passes. Use physically meaningful
absolute/relative scales near zero and measure the actual stress response.

Reuse the [bounded-input results](RESOLVED_PAD_AND_SOURCE_INPUTS_20261020.md):
the tiny detachment corrections and1000-step histories are historical numerical
controls. They authorize no further pad study. Preserve strict inputs/certificates
and separately label any engineering approximation and its correction magnitude.
Exact-model infeasibility alone does not decide whether a nearby-load estimate
is useful; aggregate wrench closeness alone does not establish stress closeness.

The [engineering sensitivity study](ENGINEERING_LOAD_SENSITIVITY_20261020.md)
already finds8.76%/14.66% contact-only stress changes for selected1%/0.1%
aggregate budgets, including substantial source redistribution. Reuse those
results instead of repeating the completed matrix; no candidate there is an
accepted general tolerance. Keep local compression/source priors, current-frame
checks and genuine failure behavior when testing cheap new models.

The resolved-boundary transfer removes the historical approximately42% inner
loading in the external-pad case, but its coarse reference error did not improve:
signed error cancellation matters. A transfer-induced field difference, smaller
stress value or same-load implementation match is not a physical accuracy claim.
Keep pressure/source, transfer, elastic-discretization and model errors distinct;
join error and speed only for the same actual configuration.

## Scope and delivery

The [completed soft-pad comparison](QUALIFIED_FRICTION_RESULTS_20261020.md) and
[pad-compliance study](PAD_COMPLIANCE_STUDY_20261020.md) remain historical evidence.
Their accuracy and timing do not transfer to hard-contact estimates. No new
soft-pad model, pad material API or coupled-pad production integration is
required. Preserve completed artifacts and running captured sources; do not
delete useful generic geometry/elastic code merely to change priorities.

Deliver a practical cheap hard-contact estimator, its applicable regimes,
error/stability maps and measured native throughput, with concise runnable
commands and recommended settings. Keep public configuration small and state
limits. Use the [accuracy/throughput protocol](ACCURACY_AND_THROUGHPUT_PROTOCOL.md)
and existing storage/reproduction workflow. Avoid new orchestration/audit
frameworks and redundant expensive sweeps. A status report or prototype alone
does not replace executed accuracy and performance results.

## Starting mechanics references

- [Abaqus Hertz contact relations](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-demanalysis.htm).
- [Sun and Paulose, shell indentation response](https://arxiv.org/abs/2103.06307):
  a starting point for shell compliance, not a universal finite-contact radius law.
