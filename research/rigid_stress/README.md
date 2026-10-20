# Native rigid stress recovery

The feature reconstructs elastic stress from changing rigid contacts inside
Genesis's existing rigid solver, entirely in Quadrants. It supports normal and
friction loads, raw global maximum/full fields, batch isolation and partial resets.
Use `RigidStressOptions.dpr(...)` for pressure and `.scr(...)` for
intrinsic/body_source_prior. Manual sampling uses `update_mode="manual"` and
`link.request_stress_update()`; every-step sampling remains the default.
See the [API](docs/NATIVE_API.md), [usage](docs/RUNNING.md) and
[source-prior model](docs/BODY_SOURCE_PRIOR_MODEL.md).

## Current goal and verified scope

The [2026-10-20 next goals](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md)
add automatic physical contact-region prediction, distinct SCR failure handling
and continued native throughput optimization. CPU investigation and production
tuning proceed alongside completion of the existing fixed-radius comparison;
these are goals, not newly validated results.

The [first CPU contact-region study](docs/AUTOMATIC_CONTACT_PROGRESS_20261020.md)
now measures distributed-source double spreading: the retained2mm DPR baseline
places about42% of normal load on the inner boundary. Conservative outer P2
transfer preserves force/moment/work and changes complete stress by8.70% on
the same linear elastic mesh. This is a transfer contrast, with inspected
pressure maps; automatic native recovery and final physical errors remain open.

The [native transfer and source-input study](docs/RESOLVED_PAD_AND_SOURCE_INPUTS_20261020.md)
adds400-step Quadrants resolved-pressure conformance and complete1000-step
DPR/SCR bounded-roundoff variants with independent mechanics/cost admission.
Original diverse SCR conflicts remain rejected; env330 additionally has a
force-only continuous certificate. The [final fixed-radius comparison](docs/QUALIFIED_FRICTION_RESULTS_20261020.md)
now includes qualified fine references,54 dimensional errors,18 conditional
rankings, matching costs and four visually inspected figures.

The [active goal](docs/QUALIFIED_STRESS_RANKING_GOAL.md) is qualified DPR/SCR/Shell
stress-error comparisons on a free full hollow egg between parallel plates.
The [accuracy/throughput protocol](docs/ACCURACY_AND_THROUGHPUT_PROTOCOL.md)
also requires independent GPU module and actual native scene measurements now.
The goal remains in progress. Full Franka physical-reference qualification is
outside this goal; its low-resolution scene remains useful for performance.

| Track | Verified result and remaining work |
| --- | --- |
| Physical normal loads | [Qualified normal fields](docs/NORMAL_PAD_QUALIFICATION_20261019.md), [common peaks](docs/COMMON_PEAK_BATCH_RESULTS_20261019.md), [81 dimensional errors](docs/NORMAL_ABSOLUTE_ERRORS_20261019.md) and [inspected spatial maps](docs/NORMAL_SPATIAL_COST_20261019.md). Ranks depend on metric and reference uncertainty. Fixed250um Shell fields are not mesh-independent thin-shell accuracy. |
| Same prescribed load | [C2 reference](docs/PRESCRIBED_REFERENCE_QUALIFICATION_20261019.md) passes25 qualification checks; [three-method comparison](docs/PRESCRIBED_METHOD_RANKING_20261019.md) supplies27 errors,15 signed tensor splits and12 complete-resultant checks. This is static prescribed-load validation. |
| Physical load/response separation | [250um Shell controls](docs/SHELL_PAD_CONTROL_SEPARATION_20261019.md) pass151 aggregation checks. Continuous, own-pad static, pressure replay and same-source solid fields remain distinct; their signed tensor differences retain cross terms. |
| Sustained friction | [Final qualified comparison](docs/QUALIFIED_FRICTION_RESULTS_20261020.md):251 reference checks,36 comparisons/18 envelopes,218 error-audit checks for54 dimensional errors/18 conditional rankings,100 matching-cost checks and four inspected final figures. DPR/SCR complete tangent errors are about7.94%/7.99% versus25.98% repaired Shell; raw Mises peak-magnitude error favors Shell. Baseline every-step failures and separately admitted1000-step bounded-roundoff variants remain distinct. |
| Research workflow cost | [24 normal timing cases/12 process-memory profiles](docs/COMPLETE_PAD_COST_20261019.md) and [friction keyframe costs](docs/FRICTION_KEYFRAME_COST_REUSE_20261019.md) include their documented research work. [Matched normal associations](docs/MATCHED_PAD_ERROR_COST_20261019.md) pass478 checks for24 configurations/120 applicable metrics; final friction associations remain pending. These are separate from production throughput. |
| Native production/module work | The [source-bound scene summary](docs/NATIVE_PERFORMANCE_SUMMARY_20261019.md) passes14,506 checks for61 primary configurations and20 independent-process memory profiles, preserving observed sampling gaps, exact conformance and source identities. The [minimal MPR repair](docs/NATIVE_CONTACT_CONSISTENCY_20261019.md), original motion failures, diverse SCR certificates and [sparse-output checks](docs/NATIVE_OBSERVATION_CADENCE_20261019.md) remain explicit. Four current-source B8 module modes retain replay/170 binding checks; the independent collector passes14 controls and all24 profile query-gap targets. Final physical-error/throughput consolidation remains open; collider speeds inherit no physical-pad errors. |

The [completion checklist](docs/QUALIFIED_STRESS_COMPLETION_CHECKLIST.md) and
[execution record](docs/QUALIFIED_STRESS_PROGRESS_20261019.md) retain outstanding
requirements and failures. A passing endpoint, numerical-core agreement or a
new report cannot close missing physical-reference or complete-history checks.

## Reproduce and interpret measurements

Follow the [parallel-plate running guide](docs/RUNNING_PARALLEL_PLATES.md).
It links source pins, data identities, captured task recipes and audit commands.
Use the established `rtx-mid` / `GENESIS_IMAGE_VER=1_26 gs-srun` workflow and
[storage policy](../../DATA_LOCATION.md); runtime data and logs stay on data.
Keep expensive refinements opt-in and never hot-edit an active engine source.

Stress-module replay excludes physical dynamics. Native production timing
compares pure rigid with max/full stress on identical evolving scenes, retaining
required runtime validity checks and GPU output. Full host readback and policy
inference require their own accounting. Default few-collider/6mm source timings
cannot inherit the qualified full-centroid/2mm pad errors. Source, geometry,
load representation, precision, cadence and solver settings must match before
joining error and speed. See the [protocol](docs/ACCURACY_AND_THROUGHPUT_PROTOCOL.md).

## Earlier accepted work and limitations

The [pressure comparison](docs/PRESSURE_COMPARISON_RESULTS_20261018.md),
[Shell PR22 comparison](docs/SHELL_COMPARISON_RESULTS_20261018.md),
[practical throughput study](docs/PRACTICAL_THROUGHPUT_RESULTS_20261017.md) and
[application study](docs/PRACTICAL_APPLICATION_RESULTS_20261016.md) retain their
own workloads, numerical acceptance and model limits. Their different rates
are not interchangeable current accuracy/cost recommendations.
The [Franka/QIPC report](docs/QIPC_REFERENCE_RESULTS_20261018.md) and
[bounded plate report](docs/PARALLEL_PLATE_RESULTS_20261018.md) preserve failed
reference/contact qualification and useful independent controls; they are not
accepted universal physical ground truth. Neither recovery route calibrates an
absolute physical fracture threshold.

Tests and useful reference modules retain their imports. The
[cleanup report](docs/BRANCH_CLEANUP_RESULTS.md), [archive](archive/README.md)
and immutable [evidence packets](evidence/) preserve earlier source identities,
negative results and restoration recipes without reopening historical goals.
