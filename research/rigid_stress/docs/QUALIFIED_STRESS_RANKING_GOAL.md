# Goal: complete qualified DPR / SCR / Shell stress-error rankings

User direction, 2026-10-18: finish the error comparison, rather than close
another diagnostic campaign. The primary application benchmark remains two
parallel rigid plates squeezing a full hollow egg. Deliver measured stress
errors, uncertainty, speed and stability for DPR, SCR and Shell against a
qualified physical reference, with practical configuration recommendations.

This is the active goal. It supersedes the unsupported-quantity completion
clause in [the preceding plate goal](PARALLEL_PLATE_COMPARISON_GOAL.md).
The [bounded plate report](PARALLEL_PLATE_RESULTS_20261018.md) and
[closeout packet](../evidence/20261018-parallel-plates/closeout-v1/README.md)
remain immutable diagnostic evidence. Their closure is not completion of
physical accuracy qualification. Full Franka qualification stays out of scope.

## Outcomes and scientific scope

Produce at least one admitted, nontrivial full-egg parallel-plate case with
finite-area contact and comparable complete fields from DPR, SCR and an
admitted Shell configuration. Expand to several small pre-failure normal
loads and a discriminating asymmetric/friction case after that first success.
Keep the egg free, use the full model, and compare common measured normal
loads. Do not pin egg vertices, assume symmetric contacts in the algorithms,
or prescribe both plate gap and reaction. Keep prescribed-gap response separate.

Use the existing names: DPR = pressure; SCR = intrinsic + body_source_prior.
Production recovery remains a feature of the Genesis rigid solver, entirely
in Quadrants, with arbitrary moving convex/nonconvex contacts, normal force,
friction, max/full output, environment isolation and partial resets preserved.

A ranking may contain statistically/numerically indistinguishable ties.
A method that cannot pass admission has a reliability failure and no valid
numerical error rank; do not invent a stress value or assign it infinite
measured error. Preserve original baselines and add explicitly named minimal
repairs where needed to obtain an admitted Shell comparison. Conclusions must
distinguish original Shell from any repaired or geometry-matched variant.

## Repair the blockers, with observable regressions

**SCR:** fix both the normal status40 rejection and the separate shear status33
fit rejection retained in the packet. Nearly cancelling source forces require
dimensionally consistent force/moment scales based on the physical source
loads and moment arms, with justified relative/absolute roundoff treatment.
Do not scale acceptance solely by a tiny net force, special-case exact zero,
disable wrench/friction checks, or globally loosen a tolerance to pass.

Diagnose the fit conditioning/rank/stopping problem separately from final
acceptance. Check feasible reconstruction against an independent CPU solve
where useful. Cover balanced and nearly balanced loads, normal plus friction,
rotated/translated configurations and scale changes; keep genuinely infeasible
loads rejected. Add focused regressions that fail before and pass after repair.

**Shell:** first test whether an unchanged practical setting can satisfy the
complete coupled equations and settle. Tight inner PCG alone is insufficient:
the measured frozen split defect is (A-M) delta_v_contact, with A=M+dt^2 K
in the zero-shell-damping test. This is a linearized diagnostic, not an exact
nonlinear reaction reconstruction. If required, implement the smallest sound
contact/implicit-elastic coupling repair in an isolated, pinned Shell variant.
Use implicit compliance or a converged coupled iteration/solve as appropriate;
validate the actual final equations, action/reaction, contact and friction.

Keep original PR22 results, settings and failures. Preserve its constitutive
model and existing feature tests; do not replace Shell with QIPC or silently
change material/contact parameters. Its original tests mainly check stretch,
relaxed bending, tearing/plasticity and total weight transfer; they do not
qualify high-stiffness contact stress. Add an observable compressed-shell
regression that detects the coupling defect. Any static velocity clearing is
a separately labeled relaxation method, not evidence of correct dynamics.

## Establish a spatially resolved physical reference

Reuse pinned QIPC 3D hollow-solid FEM and independent CPU elasticity controls.
Necessary minimal reference repairs are authorized in an isolated variant,
with original results and exact patches retained. If a suitable independent
open-source contact FEM gives a more efficient qualified reference, use it
with clear provenance and cross-check QIPC on admitted small cases.

The current one-polar-vertex load on each plate is not a resolved pressure
footprint. Same-mesh, same-traction CPU agreement controls elasticity only.
Passing force balance or CCD does not qualify the contact stress field.

Estimate contact length scales cheaply before large GPU jobs, then resolve
the polar/contact neighborhoods and wall locally. When refining, project new
surface vertices onto the same declared smooth outer/inner geometry; merely
subdividing existing planar facets can preserve the artificial sharp vertex.
Track geometry, contact interpolation, mesh quality, thickness and the actual
force-bearing support. Reuse operators/assets and avoid blind global sweeps.

Match the continuum rest geometry/wall, E=10GPa, nu=0.3, density, mass, plate
geometry and actual loads across the compared variants. Quantify finite-mesh
and thin-shell reductions. Preserve the original nominal-thickness Shell as
a separate baseline rather than hiding its mass/contact-envelope differences.

Qualify stable equilibrium windows using measured reactions, plate velocity,
egg kinetic/inertial terms, nodal equation residuals, contact validity and
solver exits. An exhausted PCG/Newton cap is not convergence. Check spatial
surface/contact and thickness resolution, timestep/loading or static-solve
sensitivity, contact regularization, and repeatability separately. A passing
endpoint does not qualify an earlier failed motion history or a separate cost run.

Predeclare quantitative uncertainty targets and the comparison domain.
Useful starting targets are <=2% common-field change and <=5% raw-peak change,
with materially smaller solver/timestep uncertainty; justify the targets
against intended method separation. Use a refinement trend with independent
controls, not one lucky adjacent pair. Tighten reference accuracy when gaps
between methods are smaller than its uncertainty. Do not retune thresholds
after observing failures, trim polar peaks, smooth extrema or fit a radius
against evaluation fields to obtain a favorable rank.

Obtain a qualified common finite-area-load elasticity/reconstruction comparison
early, using independent prescribed tractions and FEM where useful. Keep this
separate from end-to-end contact error: replay alone does not close the plate
benchmark. If the hard smooth-plate case remains expensive, a declared finite
flat/smooth contact land or physical compliant-pad case can provide an additional
tractable benchmark, with its geometry/material and compliant response matched
across methods. Do not silently substitute it for the original hard-plate case;
retain that case's status and limits. Choose a physically meaningful, resolvable
case rather than indefinitely rerunning the same unresolved point contact.

## Compute comparable errors and uncertainty

At common material locations and in compatible frames, compare complete
tangential stress tensors, membrane/bending resultants, inner/outer stress,
and damage-relevant tensile/von-Mises peaks. Use physical volume/area weights
and report both absolute and relative errors with justified near-zero floors.
Report full3D-only components separately where Shell lacks them.

Align peak definitions: native corner maxima and solid element-center maxima
are different samplings. Retain each method's raw unsmoothed global maximum,
and also evaluate a common spatial definition with resolution uncertainty.
Show peak location, contact footprint, reaction and force-indentation response.
Do not rank all methods using a conveniently selected point or percentiles
alone. Keep any engineering averaged/regularized damage measure supplemental.

Separate (1) same-domain/same-traction elasticity error, (2) contact/load recovery
error and (3) complete common-force plate response. Predeclare DPR/SCR footprint
parameters and show compact sensitivity without selecting them from test errors.
Include a nonzero sustained tangential-load case with verified support and
local reaction balance: a freely moving egg whose shear relaxes to zero does
not validate frictional stress. Match any extra physical constraints across
methods and disclose their stresses. Preserve changing/asymmetric contacts;
this benchmark does not narrow the feature's general contact contract.

Rank methods per metric and case with reference uncertainty, repeat dispersion
and failure rate visible. Do not assume native recovery or a repaired Shell
will win, and do not force one universal order when metrics disagree.

## Cost, stability and delivery

**User update, 2026-10-19:** apply the
[separate accuracy/throughput protocol](ACCURACY_AND_THROUGHPUT_PROTOCOL.md).
Accuracy verification and performance timing are independent execution tracks.
Measure/tune low-resolution native module and production scene throughput now;
fine-reference convergence does not gate those iterations. Keep research
workflow cost, production cost and GPU module cost separately named, and join
physical errors to throughput only for matching configurations.


Profile short valid workloads before long jobs. Reuse accepted assets,
environments and captures, and verify numerical consistency for performance
changes. Physical error claims still require reference qualification; they
can remain pending while independently validated performance work proceeds.

Measure actual compared configurations on matched hardware with warm repeats:
step latency, aggregate env-step/s, memory, preprocessing/factorization, contact,
elastic solve/reconstruction, observation and transfers. Distinguish every-step
stress from manual sparse sampling (including cadence), and max from full output.
The earlier ~4% sparse DPR overhead is not an every-step stress measurement.
Keep intrusive profiles separate from primary wall time. A failing primary
timing run cannot be replaced by a different passing profile run. Report
error-versus-cost only for admitted configurations; keep raw failed costs labeled.

Verify loading/unloading, repeated runs, load/radius/tolerance dependence and
the chosen friction/asymmetric control. Publish common-scale stress and actual
traction maps, force-indentation curves, error/uncertainty/ranking tables,
paired timing/stability tables, runnable commands and exact source/input receipts.
Extend the existing plate entrypoints and concise documentation; preserve old
packets and the cleaned branch. Follow [the running guide](RUNNING_PARALLEL_PLATES.md),
[storage policy](../../../DATA_LOCATION.md) and the established rtx-mid /
GENESIS_IMAGE_VER=1_26 gs-srun workflow; new runtime output stays on data.

## Completion

Completion requires executed, reproducible, qualified error comparisons for
DPR, SCR and an admitted, explicitly identified Shell configuration on the
simple physical benchmark, with a usable ranking or uncertainty-supported ties,
plus matched cost and stability evidence. Report original Shell's reliability
separately if repair was required, and disclose every changed physical model.

A new report, submitted jobs, same-traction agreement, a failure diagnosis or
an unsupported-quantity declaration alone does not complete this goal.
If a remaining blocker prevents qualified comparison, leave the goal explicitly
in progress and retain the failing evidence and next concrete experiment.
Never fabricate a ranking to satisfy the deliverable.
