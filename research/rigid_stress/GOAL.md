# Agent goal: native Quadrants rigid stress recovery

## Active next stage: load-model accuracy and physical convergence, 2026-10-14

Follow [docs/LOAD_MODEL_PHYSICAL_VALIDATION.md](docs/LOAD_MODEL_PHYSICAL_VALIDATION.md).
Quantify how the pressure, componentwise and aggregate-source load models affect
internal stress, then establish credible mesh/geometry/contact-integration and
intrinsic-distance convergence on representative convex and nonconvex shapes.
Use independent open-source FEM and resolved deformable-contact references.
Keep the full hollow egg/Franka application and add rounded convex controls,
U/cavity/narrow-gap and thin-neck shapes with interior hotspots.

The completed low-mesh performance milestone is the starting point. Physical
convergence is now authorized and is the priority. The earlier performance-first
deferral applies to that completed phase and does not block this stage.
Keep production numerics in Quadrants, preserve fast/robust model options and
the qualified native feature, and report load-model error separately from
elastic numerical error. Commit and push reproducible progress and findings.
The previous completion records do not establish this stage's acceptance.

### Immediate evidence gates

The next result must resolve the measured spatial/contact discrepancy rather
than add another same-matrix agreement check. Complete at least three useful
body/contact levels with fixed physical inputs, separated volume/contact axes,
common-point tensors, raw extrema and traction/island diagnostics. Apply the
declared observable criteria only to a credible tail; retain unexplained growth
as unresolved. Reference numerical acceleration must first preserve all original
states, full corner tensors and original force budgets. Then complete matched
egg/Franka and U contact references, resolved-load/native replay versus feedback,
remaining applicable load-model sensitivity and physically verified warmed costs.
These are execution priorities within the complete linked objective.

The [complete fine-history and linearization checkpoint](docs/PHYSICAL_FULL_SPATIAL_20261015.md)
independently audits all80 precision-aware8433-tet states at the unchanged force
budgets. Three complete480/3577/8433 histories still change23.94% in tensor
and49.66% in energy over ten common times; the final raw peak reaches2.023GPa. Resolved-load independent
linear FEM at32.5ms differs5.17% in volume-weighted tensor and3.00% in raw peak
on the finest mesh. All six actual native CUDA replays pass complete residuals
and agree with independent linear corner tensors to1.44e-10 relative difference. These qualify a complete
short reference and quantify model limitations, not physical convergence or the
full original application. Every linked A/B/C/model/feedback/cost gate remains.

The [matched-spatial checkpoint](docs/PHYSICAL_MATCHED_SPATIAL_20261015.md)
closes three uniform reference levels: final full-tensor change remains49.32%
despite0.00866% contact-force change. Three independently balanced32.5ms states
retain23.94% final tensor/49.66% energy changes and a2.044GPa raw peak;18 active
nodes do not establish a physical tail. Local finite strain reaches0.20 despite
less than0.8% wall-relative objective displacement. All original25 fine states
retain complete arrays and corner/quadrature tensors to4.86e-11 relative change
under the precision correction. Full histories, contact/geometry/model/feedback
and verified warmed accuracy/cost remain required. Editable matched maps pass QA.

The earlier [stencil-precision/original-field checkpoint](docs/PHYSICAL_STENCIL_PRECISION_20261015.md)
independently recovers rejected8433-tet step26 to9.611e-13N at unchanged physical
inputs and force budgets. Its expanded audit identifies18 active body nodes and
0.00283025mm² lumped support at that one recovered state. The80-step re-solve and its audit
remain live/scheduled. Original ground-supported startup80-state global fields,
397440 interior/corner certificate checks and four actual native CUDA replays
are qualified. Full3.3s history, physical contact/space/geometry tails, model/
feedback and verified warmed accuracy/cost remain required. The uncached30720
producer times out before saving dynamic states; it is not completed evidence.

The earlier [original-step recovery/fourth-U checkpoint](docs/PHYSICAL_CONTACT_RECOVERY_20261015.md)
independently verifies original step866 and all865 preceding physical states at
the unchanged timestep and force budgets. Direct continuation accepts867/868
but rejects869; fixed-dt residual continuation then recovers869, independently
audited together with all preceding states. Artificial continuation stages are
only Newton iterates. Complete3.3s history remains unfinished. Four U levels reduce tensor changes to
7.38%, with4.07% displacement change despite0.0901% peak change. Three uniform
egg meshes retain two active nodes and area-quartering. Residual-globalized history,
fifth-U and nonlinear refinement jobs are running/scheduled, not accepted results. The
complete physical-contact/model/feedback/accuracy-cost scope remains active.

The earlier [moving-reference checkpoint](docs/PHYSICAL_MOVING_REFERENCE_20261015.md)
independently audits280 moving time-step states and160 corresponding feedback
states, preserves all175 original accepted fields and qualifies a complete280-state
initial-guess comparison. A read-only observer verifies330 original macro controls
and360 added hold controls without changing576 original captures. The first0.1s with original initial state
and real ground support passes independent80-state audits;865 original-history
states also pass independent audits, but the3.3s attempt rejects step866 and
remains incomplete. Narrow-band level15 rejects step26
at its original force gate, while level12 retains two nodes and a280.5MPa peak.
Priority remains actual physical contact tails and complete original history/
model/feedback/accuracy-cost acceptance. These controls do not complete the goal.

Initial progress is in [PHYSICAL_VALIDATION_PROGRESS_20261014.md](docs/PHYSICAL_VALIDATION_PROGRESS_20261014.md).
It establishes independent scikit-fem assembly/boundary integration, analytic
free-body fields and seven initial full3D geometry assets.
The [independent-gauge/control update](docs/PHYSICAL_CONTROLS_20261014.md) adds
8 CPU/8 CUDA gauge/probe passes and a successful five-level shell curve. Its
finest absolute tensor error is 1.017%, but the tensor tail remains 3.966%.
The best normal-contact control accepts 11/12, with one unresolved configuration.
Neither control establishes completion of the required full physical matrix.
The [finite-load matrix and material-sheet repair](docs/PHYSICAL_LOAD_MODELS_20261014.md)
adds343 independently rechecked native fields,33CPU/54CUDA stress-module passes
and8CPU/8CUDA continuous-boundary-transfer passes. Six shapes have three useful
levels each, but no complete initial3% physical tail passes. Pressure has measured
cross-sheet compression/friction failures. The finer analytic-shell tensor tail
is0.879%; supported sticking controls converge after resolving contact-entry
increments. Full physical refinement, separated error axes, radius/profile
sensitivity, changing histories, warmed accuracy/cost and resolved egg/Franka/
interior-hotspot comparisons remain required. This stage remains active.
The [separated-error update](docs/PHYSICAL_ERROR_AXES_20261014.md) adds fixed
polyhedron volume studies, independent quadrature bounds,150 additional native
quadrature/port fields, radius sweeps and three force-controlled contact controls.
The fixed-U raw peak grows on seed reentrant creases. Finer CAD reference sweeps
are running; full-U coupled attempts remain unqualified. These findings retain
the full physical-validation objective and its unfinished completion criteria.
The [finer FEM reference update](docs/PHYSICAL_FINE_REFERENCES_20261014.md)
adds180 independent six-level fields on six shapes,90 full-field peak/work
reconstructions and10/30 complete last-change diagnostics below the initial3%
target. Current-source contact controls pass, but full free-body U attempts
remain unqualified and expose weak rigid/contact modes. Full egg reference
tasks and the original physical/contact/model/cost completion criteria remain
active; these diagnostic passes are not stage completion.
The [profile/history update](docs/PHYSICAL_PROFILE_HISTORY_20261014.md) completes
the seven-shape200-field FEM sweep and distinguishes ten numerical tail
diagnostics from seven finest-traction-admissible ones. Fixed original wrenches
permit29 admissible alternative profiles with substantial capsule stress
differences. A native PCG release repair passes8CPU/8CUDA regressions and39
changing-contact fields, with one unsupported pressure moment retained. Complete
current-source checks pass41CPU/62CUDA cases with21 explicit CUDA-only skips.
The5669-tetrahedron native U sequence also passes; two finer8-port initializations
fail allocation. Extended native settings, converged full egg/Franka/interior-hotspot
contact and the original physical operating-range acceptance remain active.
The [native port-axis update](docs/PHYSICAL_NATIVE_PORT_AXIS_20261014.md) adds204
qualified fields and six unsupported pressure moments, with identical-input
port comparisons and75 unchanged probes. Declared2/4-port settings extend the
numerical U range to40824 tetrahedra. The combined audit retains282 qualified
fields and eight unsupported moments. Remaining physical tails, reference
traction limitations and full resolved-contact completion criteria stay active.
The [initial resolved-U update](docs/PHYSICAL_RESOLVED_U_20261014.md) obtains two
full3D free-body contact pilots and validates their resolved nodal loads against
actual native full tensors. The coarse fixed stencil lacks support for the
audited gravity wrenches, while its weakest Schur mode is a common-translation
gauge. Body strains are small;5–6% local pad strains require a nonlinear check.
Finer U and full-egg references are running. Contact convergence, finite-footprint
replay, feedback separation and actual Franka matching remain required.
The [resolved-model/refinement update](docs/PHYSICAL_RESOLVED_MODELS_20261014.md)
closes the finer U reference and actual native replay, adds guarded identical
source-wrench finite-footprint comparisons, and qualifies reference-only cache/
translation-gauge options. The two useful contact levels still change44.6% at
fixed tensor probes; finer pads reach7.5% local strain. The original egg job
times out during initialization. Third-level/separated-axis U jobs and a longer
full-egg retry are scheduled. The [acceptance ledger](docs/PHYSICAL_ACCEPTANCE_LEDGER.md)
retains the complete A/B/C, geometry, convergence, feedback, cost and publication
requirements; neither these numerical passes nor scheduled work completes them.
The [nonlinear reference qualification](docs/PHYSICAL_NONLINEAR_REFERENCE_20261014.md)
checks independent P2 neo-Hookean derivatives, objectivity, analytic compression,
full part-block assembly and whole-volume nonlinear peak bounds. Saved linear
U positions expose a finite-rotation assumption in the body as well as local
pad strain. Same-protocol pad-only/all-part nonlinear contact experiments are
running from separately hashed executable snapshots, without modifying the live
canonical references. These checks are preliminary reference evidence; solved
contact convergence, objective frame/native replay, egg/Franka matching, feedback
and the complete original acceptance matrix remain required.
The [objective replay and actual Franka inputs](docs/PHYSICAL_OBJECTIVE_FRANKA_20261014.md)
close both nonlinear U pilots, bound the nonlinear global peak and obtain an
objective native-versus-nonlinear tensor difference3.09337e-5 at one spatial
level. Separated contact axes retain44.9–45.1% body-probe changes. Actual
Franka same-substep captures preserve all12 finger geoms, runtime mass/COM,
effective friction0.6 and original PD actuation at three integration intervals.
Sampled angular motion and rank5 point-wrench support prevent claiming static
application acceptance. Matched deformable Franka contact, full contact tails,
the live third-level/egg references, feedback and accuracy/cost remain required.
The [actual Panda mechanics update](docs/PHYSICAL_FRANKA_MECHANICS_20261014.md)
qualifies all12 corresponding rigid-finger geoms and repairs one numerical
contact floor using identical held rows and unchanged tolerance. Direct and
least-squares free-body release remain unqualified. Actual full-arm matching,
contact convergence, feedback and every original physical acceptance item
remain active; small strains or narrow peak bounds are not equilibrium proof.
The [contact regularization and inertial reference update](docs/PHYSICAL_CONTACT_REGULARIZATION_20261015.md)
measures default rigid overlap comparable to wall thickness and separates its
contact-only load sensitivity at fixed dt/weld parameters. Three solved
actual-finger inertial pilots meet all original dynamic gates and common-time
tensor/velocity/energy diagnostics; one actual native nodal replay passes.
These are one coarse spatial/contact setting. Matched full-arm/support/history,
physical contact/mesh tails, model/feedback separation and the original full
accuracy/cost acceptance remain required.
The [finer contact and full-arm input update](docs/PHYSICAL_FRANKA_CONTACT_TAIL_20261015.md)
adds decreasing force tails at fixed actual weld parameters and full original
Franka controller/geometry input audits. Fixed-polyhedron body/contact refinement
retains physical held points. Sparse inertia is equation- and array-identical in
the tested coarse cold protocol; fine spatial studies remain live. The full-egg
lagged initializer fails with exact retained evidence. The complete original
physical-contact/model/feedback/accuracy-cost objective remains active.

The [articulated inertia and spatial-probe update](docs/PHYSICAL_ARTICULATED_SPATIAL_20261015.md)
qualifies common material-point comparisons, full-arm contact-law force tails
and physical mass separated from implicit damping. The first480/3840 comparison
changes full tensor probes56.2% despite0.0071% resultant-force change; it does not
establish convergence. Matched editable stress maps retain raw peak information.
Fixed-contact coarse controls preserve all original arrays; fine separated
spatial studies remain live.
The complete original physical-contact/model/feedback/accuracy-cost scope remains.

The [independent contact-discretization checkpoint](docs/PHYSICAL_CONTACT_DISCRETIZATION_20261015.md)
adds three solved collision refinements with fixed bulk unknowns and original
force budgets. Coarse-body tensors change only0.002419%/0.005042%, while two
active nodes retain area-quartering and approximately fourfold force/area density
growth. This is unresolved traction concentration, not physical contact acceptance.
P2/current-friction pullback and an independent cached-Newton numerical control
are qualified. The next evidence gate is separated volume/contact resolution and
a credible spatial/contact tail; full egg/U/application matching, model/feedback
separation and verified warmed accuracy/cost remain required.

The [local-refinement/reference update](docs/PHYSICAL_LOCAL_REFINEMENT_20261015.md)
qualifies all80 finer3840-tet cached states and full corner tensors at4.522e-11
relative difference. Four fixed-polyhedron local meshes retain volume, centre
and boundary area while resolving the load neighbourhood more finely. A rejected
custom ordering candidate and three other diagnostics remain visible. Coupled
and fixed-contact local studies, a finer original-protocol reference and cached
U studies are scheduled/running; none counts as physical tail acceptance.
The first completed480/1099 local comparison changes tensors43.03%, with
raw peak562kPa on both coupled and fixed-contact paths. Two active body nodes
retain concentrated force/area and the peak changes contact neighbourhood.
The complete original application/model/feedback/accuracy-cost goal remains.

The [three-level local/articulated update](docs/PHYSICAL_LOCAL_TAIL_ARTICULATED_20261015.md)
closes480/1099/3993-tet coupled/fixed-contact studies: tensor tails remain26.8%
and raw peaks grow to2.971MPa, so no physical stress/contact tail passes. A full
MuJoCo9-joint backend now matches all576 actual mass/bias/pose/Jacobian inputs
and qualifies its effort response with the original armature/equality. Cached
U final fields and744 force gates also agree with the original reference.
The next application gate is the actual full-arm/FEM coupled solve and its
complete force residuals, followed by the unchanged convergence/model/feedback/
physical accuracy-cost completion requirements.

The [coupled full-arm/U checkpoint](docs/PHYSICAL_COUPLED_FRANKA_U_20261015.md)
solves80 dynamic steps of independent actual-Panda/FEM contact and an explicit
fixed-arm support control. Independent audits reconstruct all160 balances at
the original1e-9 gates. The third U level reduces tensor change44.6%→12.2%,
still unresolved. Matched original approach/angular history, temporal and
spatial/contact/regularization tails, feedback/model separation and physically
verified warmed cost remain required. Scheduled temporal/fourth-U studies count
as scheduled work only; the complete linked physical objective remains active.

The [feedback/time checkpoint](docs/PHYSICAL_FEEDBACK_TIME_20261015.md) independently
audits280 full-arm temporal states and160 corresponding rigid/deformable states
at original budgets. At one coarse setting, maximum tensor time changes decrease
0.0111%→0.00507%; resolved-load feedback reaches0.0192%, with actual native
nodal replays verified. These close controlled short-hold diagnostics only.
Original history, spatial/contact/regularization tails, contact-law equivalence,
finite-footprint/model effects and physically verified warmed costs remain open.
The complete linked physical-validation objective remains unchanged.

The [contact-limit checkpoint](docs/PHYSICAL_CONTACT_LIMIT_20261015.md) completes
the fourth original local level: tensor changes remain32.75% and peaks reach
30.8MPa with two active nodes. Independent geometry confirms retained vertex
normal cones; an explicit fixed-point control has no finite-energy H1 load limit.
Three full-arm contact-distance holds meet original budgets and show small
regularization tails at one coarse setting. Qualified50um numerical marking
bands now place13/25 trace samples inside the activation scale; finer solved
patches remain unverified. Priority is actual contact/volume/geometry tails and
the unchanged original history/model/feedback/physical-cost completion criteria.

## Previous nonconvex implementation brief, 2026-10-11

Implement the contact mapping contract in
[docs/NONCONVEX_CONTACT_REQUIREMENTS.md](docs/NONCONVEX_CONTACT_REQUIREMENTS.md).
Preserve surface provenance, orient normals from the material boundary, select
a finite footprint along surface topology, and fit vector tractions preserving
the original six-component wrench and local friction constraints. Support
concave surfaces, cavities, narrow gaps and changing contacts without convexity
or symmetry assumptions.

Integrate this into the existing rigid stress feature using Quadrants for the
complete production numerical path. Keep the Franka/full hollow egg workload,
full/max output options, fixed elastic operators and global peak calculation.
Use small nonconvex fixtures for contact checks and inexpensive meshes for
detailed profiling and aggregate throughput optimization. Validate same-mesh
CPU consistency and lifecycle semantics; physical convergence remains a later,
separate study. Choose the distance algorithm, scheduling and tuning from
evidence rather than copying the Python prototype or fixing its port count.

This is a new requirement after the milestone below. Its prior completion
record does not establish native nonconvex-contact acceptance. Record and push
validated progress, including unfinished work and unavailable device checks.

## Nonconvex extension completion record, 2026-10-14

Native nonconvex contact mapping, lifecycle/field validation, detailed
optimization and measured performance acceptance are complete. Production
numerical source `76537c2f` is unchanged during final closeout. See the
[final report](docs/NATIVE_NONCONVEX_RESULTS_20261014.md) and
[requirement-to-evidence audit](docs/NATIVE_NONCONVEX_CLOSEOUT_20261014.md).

Current-source coverage is135 CUDA passes and114 CPU passes/21 explicitly
CUDA-only skips. This combines the retained complete initial suite with10
fresh CPU/CUDA passes in the sole changed test module. The two original CUDA
saved-edge failures came from checking elastic equilibrium against a
separately fitted CPU contact load. The corrected test independently balances
the actual native loads, retaining the complete CPU K/gauge residual and all
independent force/moment/cone/nodal/full-field/peak gates. Original failures,
RHS differences and source provenance remain published. Engine tolerances
stay fixed. Final-source464 dynamic fields and160 original-wrench/
independent-relief fields also pass CPU FP64.

Six matched four-scope widths, B1024/2048/4096/8192/16384/32768, complete
with original banks,600 warmup and three1280-step repeats per scope.
The one-GPU copied2048-condition curve reaches150133.237 live env-step/s
at B32768, with only2.623% gain for its last33.3% batch increase.
Separately allocated B36864 live/policy trials complete at154343.359/
154176.223 env-step/s, with zero invalid steps, overflows or failed lifts.
B40960 reaches the unchanged Jacobian address guard. B38912 fails warmup
during complete sample-position growth with CUDA out of memory. Their
diagnosed failures have no accepted rates. This establishes a practical
device/workload endpoint, with no global-optimum or integer-maximum claim.

Final-source same-GPU max/full/max measurements complete for live/policy
at B1024 and B8192. Full fields add107520 bytes/environment. Measured rate
costs are0.017%/0.772% and0.739%/0.745% respectively. Live poses and all
per-environment rigid/grasp/contact/descent outcomes match exactly. Policy
evolves independently from observations, so cross-process poses/peaks remain
rollout diagnostics. Same-state numerical accuracy uses the unchanged strict
tests and independent CPU fields.

The final closeout manifest verifies6997 entries/6997 exact originals and
two complete large-JSON reconstructions. The previous97 checkpoints verify
43120 entries/43119 exact originals/one exact archived allocation-log prefix
and36 complete reconstructions. The prefix and both hashes are explicitly
recorded.389 restored identity files match their original SHA256/size.
The combined allocated audit passes all98 publications:50117 entries,
50116 exact originals/one exact prefix and38 complete reconstructions.
Every manifest artifact is explicitly checked for Git inclusion.
The final source/test/docs/evidence deliverables are committed and pushed
to `Roushelfy/Genesis:rigid-stress-recovery`, with remote HEAD checked at
delivery. Runtime data and large staging remain in the declared data root.

Physical mesh/quadrature convergence and complete RL training remain later,
separate studies. The following dated checkpoints preserve their original
observations. This completion record supersedes their pending/running items.

## Archived extension status, 2026-10-13

The extension remains active. Pushed numerical source `eba3aeb2` qualifies
the four selected layout/capture/Hessian/diagnostic optimizations with35CPU/
35CUDA cases,96fixture fields,464dynamic fields and160original-wrench/
independent-relief fields. Its same-GPU B16384 complete live pair improves
106145.289 to113913.060 valid env-step/s (1.073180556x).

The [warp sampler follow-up](docs/NATIVE_WARP_COUNTERS_20261013.md) passes
35prototype CUDA cases in808.30s and a complete B8192/623105 same-GPU pair:
107515.913 to117664.460 valid env-step/s (1.094391115x), with identical banks,
rigid positions and per-environment contact/grasp counters, and zero invalid
steps/overflows/failed lifts. Numerical commit `76537c2f` removes the
prototype flag and duplicate path. Its CUDA kernel arguments/body are
AST-identical to the qualified prototype; the CPU numerical route is unchanged.
Clean-source35CUDA passes810.81s. The independent B16384/734001 pair
improves115452.687 to135495.178 valid env-step/s (1.173599173x), with zero
failures and exact rigid counters/positions. The complete two-cohort,
prototype/clean/source-equivalence qualification is published.

Final-source B1024/2048/4096/8192 four-scope measurements are verified and
published. Final-source464dynamic fields and160original-wrench/independent-
relief fields pass CPU FP64. The initial sampler now costs22.906ms per
instrumented B16384 step; warm fit costs25.026ms. These measurements load the original baseline banks;
they hold timestep, load law, sampling and precision budgets fixed. Wider
four-scope runs and the copied2048-condition scaling curve are running.
A complete constraint-interior/first-accept work audit gives no reasonable
material gain expectation for an unconditional Gram prepass or deferred
scatter-verification/replay path: first acceptance covers13.8% of samples,
replay covers86.2%, and additional interior evaluations are3.8% of total
sample-evaluation work. These are work proxies, not wall-time bounds. The
current fit remains selected. Wider scope/copy publications and the final
profile/cost plateau report remain required.
Neither the primitive counter probe nor an unfinished run establishes goal
completion. Physical mesh/quadrature convergence stays outside this phase.

## Extension development record, 2026-10-12–13

The chronological entries below retain the evidence available at each
checkpoint. Their older pending items are superseded by the current status
above and the later linked qualification reports.

The extension remains active. Numerical baseline `6a861556` contains
qualified contact repairs and selected seed/layout/port/scatter optimizations.
The subsequent [cold cooperative retry](docs/NATIVE_COOPERATIVE_RETRY_20261012.md)
is selected with a same-GPU1.036986834x live gain,23 CPU/23 CUDA production
cases and320 independent dynamic fields passing unchanged final budgets.
The subsequent [eight-sample-lane retry](docs/NATIVE_SAMPLE_LANE_RETRY_20261012.md)
has a separate paired1.044883994x live gain,23 CUDA production cases and160
dynamic independent fields passing. [Complete-sample compaction](docs/NATIVE_COMPACT_TRACTION_PROFILES_20261012.md)
passes31 CPU/31 CUDA cases and96 fields, saving15.715GB native bytes atB8192
on a separated seed cohort, with a small throughput regression. Its fused
capacity follow-up remains unselected. [Current-step position reuse](docs/NATIVE_CACHED_FAN_POSITIONS_20261012.md)
passes31 CPU/31 CUDA cases and96 fields, with an isolated1.074361448x live
gain on counted storage. Combined B2048/B8192 comparisons versus production
full storage are qualified below.
The [combined storage/tail checkpoint](docs/NATIVE_CONTACT_TAIL_STORAGE_20261012.md)
now qualifies1.022885722x B2048/1.027915774x separated-seedB8192 with14.116GB
native savings. Parallel fan sampling qualifies1.017149258x live and160
additional dynamic fields, reducing hard fan calls to4.3–4.4ms. Cooperative
refit and scalar sample layout pass their matrices but have only approximately
0.3% effects within repeat variation and remain unselected. The cleaned
production combination is now [qualified](docs/NATIVE_SELECTED_STORAGE_FANS_20261012.md)
with31 CPU/31 CUDA cases,96 independent fields and a same-GPU B2048 ratio
1.003100208 over the counted-position/serial-fan prototype, within repeat
variation. Its hard-tail improvement is retained; gains are not additive.
[Exploratory wide live probes](docs/NATIVE_WIDE_COUNTED_POSITIONS_20261012.md)
reach107641.889 valid env-step/s atB16384 and104753.768 atB32768, with
zero failures, on separate GPU/seed cohorts predating the clean combination.
Final-source wider scopes/seed measurements, bottleneck decisions and plateau
are still pending.
[Current production baselines](docs/NATIVE_WIDE_BASELINE_PROFILE_20261012.md)
now verify six matched four-scope widths throughB32768,464 captured full
fields and160 additional original-wrench/CPU-inertia-relief fields. B16384
live/policy rates are105910.718/105892.927 valid env-step/s with all outcomes
retained and zero failures. B32768 live/policy also pass104316.704/104146.207
valid env-step/s. Clean combined qualification, sparse diagnostic clearing and warm-Hessian precision
decisions are pending; no new completion is claimed.
The copied-cohort curve now passes unchanged workloads atB8192/16384/24576/32768,
peaking at108645.547 valid env-step/s atB16384 for production92ec3e0c.
[Valid-source-only initialization](docs/NATIVE_VALID_SOURCE_CAPTURE_20261012.md)
qualifies1.009451355x B8192/1.011652788x separated-seedB16384 and31CPU/31CUDA
cases/96 fields, selected for clean integration. This baseline platform does
not complete the pending final candidate/combined-source work.
[Coalesced cached coordinates](docs/NATIVE_COALESCED_POSITIONS_20261012.md)
qualify1.017995104x B16384/1.015946367x separated-seedB8192,31CPU/31CUDA/96
fields and eight exact cached/direct geometry comparisons, selected for clean
integration. Isolated FP32-Gaussian cost is below0.2% of the fenced step and
does not justify a production change. Sparse diagnostic/warm-Hessian matrices
both pass31CPU/31CUDA; live decisions and clean combined qualification remain.
[Warm Hessian and diagnostic clearing](docs/NATIVE_WARM_HESSIAN_DIAGNOSTICS_20261012.md)
now qualify two cohorts each: Hessian1.033683688x/1.020015526x, diagnostic
clearing1.009038163x/1.010380685x, with31CPU/31CUDA/96fields for each candidate
and zero invalid/failed outcomes. The squared-distance gate has no gain and is
not selected. Four-candidate clean code passes3CPU/3CUDA hard/lifecycle smoke
and33CPU combined cases;33CUDA/live comparison, dynamic oracles, inactive-walk
decision and final-source four-scope/scaling remain pending.
[Clean combined qualification](docs/NATIVE_COMBINED_QUALIFICATION_20261013.md)
now records a same-GPU1.073180556x complete B16384 live gain, zero invalid/
failed outcomes and exact rigid counters/positions. The first33CUDA matrix
has32passes/1over-strict repeat-precision failure, fully preserved. Independent
equal-history/forced-FP64 diagnostics identify its cause; budget-aware isolated
checks add original wrench and complete CPU/gauge residuals and pass2CUDA.
The expanded35CPU/35CUDA matrix now passes570.217/818.219s with96 independent
fixture fields;464dynamic fields and160original-wrench/independent-relief
fields also pass unchanged budgets. Final-source scopes/scaling and the
sampler atomic-cost decision remain pending. The inactive-walk exit has only1.000350925x with overlapping
repeat ranges and is not selected. No extension completion is claimed.
Production source `eba3aeb2` and its full qualification are pushed. Synthetic
warp counter costs pass six complete index/count/max checks and motivate a
private sampler prototype, whose3CUDA hard/growth cases pass157.13s. Its35CUDA
matrix and full same-GPU B8192 live pair remain pending; no production gain
or selection is claimed from the primitive cost alone.
[Matched B1024 four scopes](docs/NATIVE_MATCHED_SCOPES_20261012.md) include timed
grasp outcomes, resets and environment work. Contact graphs, fastcache and
standalone device-frontier maximum have no selected dynamic end-to-end benefit;
see [the ablation ledger](docs/NATIVE_CONTACT_SAMPLING_ABLATIONS_20261012.md).
[Dynamic profiling and320 independent GPU field samples](docs/NATIVE_DYNAMIC_PROFILE_20261012.md)
pass unchanged budgets and identify a0.74-1.49s serial fit tail. Cold retry
removes the largest unrefined tail; refinement tails remain. Baseline B4096
and B8192 matched four scopes now pass. Final-source wider scopes, separated
seeds, actual-sample memory compaction, remaining candidate decisions and a
throughput plateau are still required before extension completion.

## Completion record, 2026-10-11

The low-shell native feature, contact repair, unchanged-budget numerical and
lifecycle validation, measured optimization plateau and reproducible four-scope
GPU evidence are complete. See [the final report](docs/NATIVE_RESULTS_20261011.md)
and [generated measurement tables](docs/FINAL_MEASUREMENTS_20261011.md).
Production engine is `f3e92c62`, standalone timing preparation is `1f22f6a2`;
52 GPU cases, 31 CPU cases and 432 independent CPU FP64 full-field samples pass.
All 2004 publication artifacts and their originals pass verification. The
requirements below are preserved as the original brief; physical convergence
remains a separate later study.

## Internal stopping budget, 2026-10-10 follow-up

Match internal contact fitting termination to the actual error budget. The
historical 2e-12 normalized dual threshold is not an acceptance requirement.
An internal relaxation must pass unchanged final contact force/moment,
complete equilibrium including all gauge rows, and independent CPU FP64
stress consistency checks before selection. Sampling/physical infeasibility
must still be distinguished from numerical stagnation.

## Additional output requirement, 2026-10-10

Add a build-time option selecting maximum-only or complete stress-field
output, with maximum-only remaining the default. Full mode exposes the
complete device-side tensor/von Mises field and preserves the existing
maximum observation. Validate same-state numerical agreement, lifecycle
semantics and matched throughput/memory costs for both modes on the low mesh.
See [docs/OUTPUT_MODES.md](docs/OUTPUT_MODES.md) for the required contract.

## Current acceptance, 2026-10-10 follow-up

Prioritize legal-contact robustness before further optimization. The actual
B=2048 apex contact is infeasible under the Q10 constant-ratio sampled pad
law (132 eligible samples, transverse hull gap 10.9 micrometres). Improve the
declared load representation or local integration and turn that fixture into
successful recovery. Distinguish sampling, fitting failure and physical
incompatibility. Keep force, moment, frames and friction constraints explicit.
Vary actual combined contact coefficients in the live workload.

Use only the low-resolution full hollow shell in this phase. Rebuild a matched
baseline after any load-model change. Profile and evaluate applicable batch
tiling/reuse, scatter/reduction, scheduling, device conditions, failure packing,
shared operators, precision/correction, history and global peak candidates.
Record a measured result or concrete cost/profile reason for each decision.

Final GPU evidence covers effective rigid-only, recovery-only, live and policy
rates, batch steps/s, numerical failures, grasp outcomes, tail work and memory.
Use changing contacts, different seeds, partial resets and repeated trajectories
at B=1024,2048 and larger batches. Resume peak exploration after contact repair.
The earlier copied-condition scan was stopped at the user's request and is
preliminary historical evidence. Policy inference defines rollout scope only.

Complete and push only after contact, numerical and lifecycle checks pass and
measured optimization reaches a plateau with no unresolved actionable major
bottleneck. Missing resources must leave their tests explicitly unverified.

## Implementation objective

Implement auxiliary fixed-geometry linear-elastic stress recovery as an
optional, native feature of Genesis's existing rigid solver in
`Roushelfy/Genesis`, branch `rigid-stress-recovery`. Return the global maximum
von Mises stress per environment and maximize measured aggregate environment
transitions/s, including parallel scenes and matched policy inference.

Read [docs/NATIVE_QUADRANTS_PLAN.md](docs/NATIVE_QUADRANTS_PLAN.md) first, then
the relevant math/reference evidence and Genesis's development conventions.
This goal and that document replace the previous standalone GPU architecture
and convergence-first acceptance sequence. The existing research implementation
is a numerical reference and source of measurements, not the final feature.

Use Quadrants for all production numerical computation, including contact/load
mapping, elastic operators, recovery solves/preconditioners, residuals, history,
selection and maximum-stress reduction. Integrate configuration, state,
substep execution, resets and public observation into the existing rigid
solver lifecycle. Independent offline CPU FP64 oracles and ordinary Python
configuration/asset handling remain useful; CuPy, raw CUDA, Triton, Torch
numerical kernels, cuDSS and cuSPARSE do not implement the production feature.
Consult [upstream PR #3372](https://github.com/Genesis-Embodied-AI/genesis-world/pull/3372)
for native integration, Quadrants and profiling patterns. Choose the numerical
method and API that best fit this feature; a separate Mochi solver is not needed.

Use a Franka Panda gripping a full hollow egg-shaped rigid link. Only reference
geometry is fixed relative to contacts: positions, counts, footprints,
directions and symmetry remain unrestricted. Include normal and tangential
frictional forces and all support contacts. Preserve enabled contact moments,
or explicitly reject unsupported modes. Smooth grasp evolution can help reuse;
contact changes, slip, release and partial resets must remain correct. Material
and mass are declared fixed inputs to any shared operator. Rigid dynamics stay
unchanged by this auxiliary stress calculation.

Prioritize performance work on inexpensive low-resolution full-shell meshes.
First establish same-mesh numerical consistency, obtain detailed profiling,
and iteratively remove the actual throughput and memory bottlenecks. Explore
the documented candidates and additional opportunities identified by profiling;
retain improvements based on end-to-end measurements, not isolated timings.
Keep accuracy, timestep, sampling and the load model fixed within comparisons.

Before detailed profiling and the applicable performance optimizations have
been completed and documented, do not run high-resolution physical mesh or
quadrature convergence studies. Numerical consistency checks continue throughout:
same-input FP64 comparisons, complete residual including gauge rows, global
peak error, contact wrench/frame consistency, batching and reset independence.
Low-resolution results are performance-development results, not physically
converged egg stresses. Physical convergence is a later, separate stage.

Deliver native engine code, a normal Genesis example and tests, reproducible
low-resolution benchmarks, a detailed stage profile, optimization/ablation
evidence and a concise account of remaining bottlenecks. Measure rigid-only,
recovery, live rigid-plus-stress and policy-plus-rigid-plus-stress scopes.
Report actual hardware, precision, mesh, batch size, memory, numerical errors
and aggregate transitions/s. Commit and push to this task branch. Work
autonomously; use evidence and engineering judgment to choose implementation
details. If device access is unavailable, identify the unmeasured parts.
