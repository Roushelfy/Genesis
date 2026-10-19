# Goal: compare DPR, SCR and Shell with two parallel compression plates

**Superseded as the active goal, 2026-10-18:** follow
[QUALIFIED_STRESS_RANKING_GOAL.md](QUALIFIED_STRESS_RANKING_GOAL.md).
The bounded diagnostic study below is historical. Its unsupported-quantity
completion clause does not close the new physical error-ranking goal.

User direction,2026-10-18: use a simple scene, such as two parallel planes
squeezing the hollow egg, for the stress-error comparison. Remove the Franka
robot, joint/controller mapping and full grasp lifecycle from the **primary
accuracy benchmark**. Deliver useful DPR/SCR/Shell error, speed and stability
comparisons with a qualified QIPC solid-FEM reference on this simpler problem.

This goal supersedes conflicting full-Franka reference/convergence gates in
[the previous goal](QIPC_REFERENCE_COMPARISON_GOAL.md). Existing Franka results
remain engineering/application evidence; a physically qualified complete
Franka grasp is not required to finish this stage. Preserve the existing data
and assess running jobs at a safe boundary without hot-editing workers.

## A small physical benchmark

Use the same full hollow egg geometry/material conventions and two large,
flat, parallel rigid plates. Keep the contact region away from plate edges.
Use the complete model; do not introduce a symmetry reduction or hard-code
contact locations. Robot links, joints, self-collision filters and support-floor
contact are unnecessary in the baseline. Initially disable gravity and friction.

Prefer **quasi-static force-controlled compression** for the primary comparison:
fix one plate and apply a common, slowly ramped normal force to the other plate,
whose permitted motion is along the plate normal. The egg remains free;
contact determines the actual reactions, indentation and footprint. A simple
verified force controller or direct plate force is sufficient. Do not freeze
egg surface vertices to stabilize the model and create artificial stress peaks.
Any rigid-mode gauge in a static solve must not supply an unreported reaction.

Choose a few small pre-failure load levels after inexpensive pilot checks.
Reach mechanically verified equilibrium at each level, then save the complete
spatial stress field. Record residuals, remaining kinetic energy/inertia, plate
force balance, position and actual contact loads. Retain continuous history
between levels. A static solve is also suitable if its boundary/contact and
friction treatment is verified; no new static solver is required.

Do not prescribe both the actual plate separation and its reaction force.
Equal jaw separation can cause different loads under rigid-contact regularization,
Shell coupling and QIPC; a prescribed-gap test is a useful **separate** response
comparison, not the primary equal-force stress test. In the equal-force test,
different predicted indentation is an observable rather than something to tune
away. Verify measured plate/egg reactions, not just the commanded force.

After the normal frictionless baseline, add only a few discriminating controls:
nonzero common friction with a small prescribed tangential plate motion while
normal force remains controlled, and an off-axis/oriented-egg case where useful.
Keep the same physical geometry and history across methods. These tests do not
relax the production feature's arbitrary convex/nonconvex contact assumptions.

## Common reference and error separation

Use pinned cuda-graph-qipc **3D hollow solid FEM** as the candidate reference,
with independent analytic/CPU FEM controls already available. DPR/SCR retain
their existing native rigid mechanics and load reconstruction. Keep original
PR22 Shell, and label any mass/thickness/contact or numerical variant separately.
Use the established names and exact API mappings:
DPR = pressure; SCR = intrinsic + body_source_prior.

Match the actual rest surface, wall, material, mass, initial pose, plate geometry
and applied loads. Report remaining Shell thickness/inertia/contact-envelope
differences and constitutive assumptions. Do not force the reference pressure
distribution to follow our contact_radius or fit that radius to evaluation fields.
Predeclare reconstruction parameters; show a compact sensitivity where relevant.

Start with short-window profiling and a bounded tolerance study, then check
surface/thickness resolution, load-ramp/settling or timestep sensitivity, and
contact/friction regularization on the selected equilibrium fields. QIPC's
finite completion and non-penetration alone are not reference qualification.
Use actual residuals and admitted solver exits. Retain the current uncertainty
targets as engineering guidance, not a demand for a universal mesh sweep.
Localize convergence problems before spending time on larger meshes.

Compare all three methods at common material positions and in compatible frames:
tangential stress fields, membrane/bending resultants, inner/outer stress,
spatial peak magnitude/location, plate reaction and indentation/contact footprint.
Report full3D solid fields separately and Shell's missing components explicitly.
Preserve unsmoothed extrema and any unresolved contact singularity; include
absolute errors when a reference is near zero. Sparse observations mean full
spatial fields at selected equilibria, not dense temporal stress histories.

Separate same-load elasticity from contact/reconstruction error. Reuse resolved
QIPC traction replay and independent elastic checks where informative; then
compare complete plate responses. Equal total force does not uniquely determine
the pressure footprint. Preserve DPR/SCR interface rejection and Shell static/
dynamic failures rather than silently replacing them with successful subsets.

## Cost, stability and delivery

Profile solver, contact, reconstruction and observation separately. Primary
cost uses continuous plate mechanics with occasional full-field observations,
including actual admission checks; exclude rendering and intrusive exports
unless explicitly measured. Compare matched hardware, useful batch sizes,
latency/aggregate env-step/s, memory and error-versus-cost. Retain original and
useful tuned settings; an inaccurate/no-contact run is not an accepted speedup.

Check repeatability, force/tolerance dependence, loading/unloading and friction
where applicable. A short equilibrium test is enough to expose a defect;
do not rebuild a complete robot to reproduce it. Preserve unmodified baselines,
exact sources/inputs, failure states and compact evidence.

Deliver a clear plate-benchmark report with common-scale stress/traction maps,
force-indentation curves, per-load error/uncertainty and stability tables,
cost/profile/scaling results, runnable commands and practical DPR/SCR/Shell
recommendations. Keep prior Franka performance separately labeled; refresh
sparse application cost only where useful, without reopening its full physical
qualification campaign. Keep native production numerical work in Quadrants.
Completion requires actual comparable, qualified simple-case results or an
explicitly diagnosed unsupported quantity, not just another submitted sweep.
