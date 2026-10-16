# Application-first validation and delivery

User direction, 2026-10-16, for `Roushelfy/Genesis:rigid-stress-recovery`.

## Goal and precedence

Deliver a useful, fast rigid stress observation for damage-aware grasping and
RL, led by the Franka/full hollow egg application. Establish what the feature
can reliably predict, choose practical settings and expose measured limitations.

This direction supersedes the 2026-10-14 requirement to finish the entire
physical-convergence matrix before delivery. Read the earlier
[validation plan](LOAD_MODEL_PHYSICAL_VALIDATION.md) and
[ledger](PHYSICAL_ACCEPTANCE_LEDGER.md) as evidence and diagnostic resources.
Their statements that every original gate remains mandatory do not override
this update. Preserve all earlier results and failures; changing priorities
does not turn an unresolved result into an accuracy pass.

The engineering delivery, physical-model qualification and experimental
fracture calibration are separate milestones. Do not block engineering delivery
on universal continuum convergence or on calibrating a real failure threshold.

## Accept the verification already earned

Distinguish three questions in both execution and reporting:

| Question | Evidence and acceptance |
|---|---|
| Is the discrete elastic implementation correct? | Independent FEM assembly and load transfer, complete equilibrium including inertia relief/gauge, displacement, energy, full tensor and raw peak for equivalent inputs. Accept the already-qualified cases within their demonstrated scope. |
| Is the selected discretization adequate for the application? | Representative mesh/integration sensitivity and an explicit accuracy/cost choice. Agreement between solvers on one mesh does not answer this question. |
| Does reconstructed loading represent contact adequately? | Compare against independently specified traction and, especially, loads produced by a suitably qualified deformable-contact reference. Total force/moment balance alone does not determine pressure or shear distribution. |

Existing prescribed-traction comparisons are genuine partial load-model
validation. Existing same-load FEM agreement is genuine elastic verification.
Neither alone establishes actual contact-distribution accuracy. Do not repeat
large independent-assembly campaigns for unchanged, already-qualified code.
Add targeted regression checks when a change or observed defect warrants them.

## First priority: isolate the error that matters

Reuse the existing references and snapshots to make the following comparison
on representative states. Keep geometry, material, mass, physical coordinates,
friction, supports and load conventions matched. Begin where deformation is
small enough for linear recovery to be meaningful.

1. **R: resolved contact.** Independently solve deformation and frictional contact
   together, retaining its actual traction/nodal loads and stress.
2. **L: resolved-load replay.** Apply those resolved loads to the native linear
   recovery with consistent supports or inertia relief.
3. **M: reconstructed-load replay.** Extract only the contact information the
   rigid feature actually receives (points, forces, enabled moments and source
   identities), reconstruct the finite footprint with the selected model and
   recover stress through the same elastic path as L.

Compare M with L to isolate reconstruction effects conditional on the reference
contact data. Compare L with R to measure the remaining elastic approximation
and any explicitly identified support/inertia/frame differences. Keep M versus R
as the combined comparison. Do not attribute every L/R difference solely to
linearization unless the other differences have been controlled.

For the actual grasp, also compare a corresponding rigid run under matched
actuation/supports. It can produce different forces than the deformable run.
Measure that omitted-feedback effect separately; do not silently replace the
rigid run's forces with the reference forces and call it application accuracy.
Choose a compact set of informative snapshots and histories, rather than
requiring a fresh fine full-field audit at every initializer/substep.

The contact reference is a numerical reference, not experimental ground truth.
Check its relevant spatial/time/contact sensitivity enough to make the comparison
informative. If its uncertainty prevents a conclusion, label that conclusion
unverified and use a simpler qualified control before expanding the sweep.

## Representative application coverage

Prioritize the full hollow egg grasp, one nonconvex case with a global interior
hotspot (for example a rounded U or thin neck), and a small smooth convex control.
Reuse other completed shapes when they answer a distinct question; the complete
shape x load x model x resolution Cartesian product is not an acceptance gate.

Cover asymmetric/changing contacts, normal force plus friction, multiple sources,
self-balanced grips, contact transitions/release and environment resets using
small fixtures or representative histories. Fixed geometry does not authorize
fixed contact locations, symmetry or a contact-only peak search. Smooth contact
history is a reuse opportunity, with correct invalidation at transitions.

For the applicable fast and robust model options, quantify the differences that
affect the actual workload: raw global maximum and location, selected full-field
maps, per-patch/island force/moment redistribution, footprint/profile sensitivity
and cost. A similar scalar maximum does not establish similar load distribution.
Keep unsupported moment/friction cases explicit rather than modifying their
inputs to obtain acceptance. Suggest a default and explain when another option
is needed, without claiming that one traction law is universally correct.

## Bounded convergence work and practical acceptance

Use convergence studies to choose useful settings and diagnose material error,
not as an open-ended prerequisite for all shapes. Reuse the existing curves.
Do the smallest additional controlled refinement needed to assess a selected
setting or distinguish a consequential geometry/load/integration defect.
Hold the physical shape, finite footprint and loading law fixed when interpreting
refinement; changing their physical meaning is a different case.

Choose observable-specific criteria from the intended application before judging
new results. Report raw peak error/sensitivity, hotspot behavior and whether the
uncertainty could change a damage penalty or episode-termination decision.
For a reward proxy, assess repeatability and useful discrimination between
grasp choices within the declared range. Do not assume a universal 3% threshold
or promise reliable decisions inside an unresolved uncertainty band.

An unresolved peak tail, including the recorded 14-17% convex examples, may be
retained as a quantified limitation without blocking the entire engineering
delivery. It cannot support a claim of absolute stress accuracy at that setting.
If that uncertainty defeats the intended decision, improve the relevant model
or settings, or explicitly mark that operating case unsupported. Do not hide
the issue with smoothing, a changed radius/material, or a substituted percentile.
Raw global unaveraged maximum remains the requested output; additional metrics
may be offered only as separately named options. A claim of a physical fracture
threshold requires material/experimental calibration beyond this delivery.

## Deliver the useful native feature

Keep this a feature of the existing Genesis rigid solver with all production
numerics in Quadrants. Preserve optional recovery, full/max output selection,
arbitrary convex/nonconvex contact, friction, shared fixed operators, partial
resets and the qualified throughput paths. External FEM/contact is reference-only.

After selecting practical settings, measure warmed end-to-end multi-environment
throughput, memory and incremental stress cost, with rigid baseline and scope
clearly stated. Include contact mapping, recovery, checks, peak, history/reset
and policy costs in the applicable scopes. Pair performance with its measured
accuracy/model limitations; previous low-mesh rates remain scoped historical
performance, not automatically physically qualified rates.

Deliver a concise application report and runnable commands: recommended mesh,
load model/radius assumptions, accuracy-versus-cost results, representative stress
and traction maps, reproducible source/configuration identity, and an explicit
supported/conditional/unsupported operating-range table. Retain enough actual
data to reproduce conclusions and diagnose failures, using existing archives;
avoid making repeated giant data publication or duplicate audits the critical path.

Completion means the usable feature and recommended settings are documented,
existing elastic verification is accepted within scope, the representative
load-reconstruction and feedback effects are measured or clearly bounded,
application-relevant regressions pass, and selected settings have measured cost.
If a useful operating case cannot be qualified, state the concrete blocker and
ship only the supported/conditional scope. Do not describe an unavailable
reference check as passed. Further universal convergence studies, extra shapes,
full RL training and real fracture calibration can remain separately listed work.

Use engineering judgment on algorithms and experiment sizes. Spend the next
compute budget on the smallest experiment that changes an application decision.
