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
| Sustained friction | [250um reference](docs/FRICTION_WALL250_20261019.md) passes mechanics and wall controls; the [first125um/two-wall parent](evidence/20261019-qualified-ranking/canonical125-surface-terminal-v1/README.md) completes400 steps and passes10 strict history/momentum/endpoint checks; the [250→125um surface component](docs/FRICTION_SPATIAL250_20261019.md) passes65 checks with at most1.69784% field/0.45060% peak changes; [the complete seven-parent/36-comparison pipeline](docs/FRICTION_REFERENCE_QUALIFICATION_PIPELINE_20261019.md) is queued; fine qualification, [54-row dimensional/ranking audit](docs/STVK_METHOD_RANKING_PIPELINE_20261019.md) and [final figures](docs/FRICTION_FIGURE_PLAN.md) remain open. [Renderer controls](docs/STVK_FIGURE_RENDERING_20261019.md) pass15 checks/four inspected previews; final drawing independently requires matching fine admission. [Terminal endpoint continuation](docs/REFERENCE_ENDPOINT_CONTINUATION_20261019.md) now recovers three actually preempted parents from200/200/400 saved steps. Periodic output passes33 coarse equivalence checks; three fine input controls pass45, lineage26 and independent state/control reinspection99. [Partial-level recovery](evidence/20261019-qualified-ranking/solid-pad-partial-step-recovery-v1/README.md) now passes actual300+100 and225+175 coarse controls, each with34 unchanged-limit conformance and49 two-link ancestry checks. Three CPU continuations run in disjoint node pools; full fine admission remains pending. Both native every-step modes retain initial-detachment failures; requested-keyframe admission does not admit every-step stress. |
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
