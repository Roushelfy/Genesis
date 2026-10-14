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

Initial progress is in [PHYSICAL_VALIDATION_PROGRESS_20261014.md](docs/PHYSICAL_VALIDATION_PROGRESS_20261014.md).
It establishes independent scikit-fem assembly/boundary integration, analytic
free-body fields and seven initial full3D geometry assets.
The [independent-gauge/control update](docs/PHYSICAL_CONTROLS_20261014.md) adds
8 CPU/8 CUDA gauge/probe passes and a successful five-level shell curve. Its
finest absolute tensor error is 1.017%, but the tensor tail remains 3.966%.
The best normal-contact control accepts 11/12, with one unresolved configuration.
Neither control establishes completion of the required full physical matrix.
The complete load-model sensitivity, physical matrix and deformable-contact
comparisons remain required. This stage remains active.

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
