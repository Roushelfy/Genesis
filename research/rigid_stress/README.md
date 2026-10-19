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
| Sustained friction | [250um reference](docs/FRICTION_WALL250_20261019.md) passes mechanics and wall controls; [the complete seven-parent/36-comparison pipeline](docs/FRICTION_REFERENCE_QUALIFICATION_PIPELINE_20261019.md) is queued; fine qualification, [54-row dimensional/ranking audit](docs/STVK_METHOD_RANKING_PIPELINE_20261019.md) and [final figures](docs/FRICTION_FIGURE_PLAN.md) remain open. [Renderer controls](docs/STVK_FIGURE_RENDERING_20261019.md) pass15 checks/four inspected previews; final drawing independently requires matching fine admission. [Terminal endpoint continuation](docs/REFERENCE_ENDPOINT_CONTINUATION_20261019.md) passes33 coarse equivalence/15 provenance checks as a resource fallback; live fine tasks are unchanged. Both native every-step modes retain initial-detachment failures; requested-keyframe admission does not admit every-step stress. |
| Research workflow cost | [24 normal timing cases/12 process-memory profiles](docs/COMPLETE_PAD_COST_20261019.md) and [friction keyframe costs](docs/FRICTION_KEYFRAME_COST_REUSE_20261019.md) include their documented research work. [Matched normal associations](docs/MATCHED_PAD_ERROR_COST_20261019.md) pass478 checks for24 configurations/120 applicable metrics; final friction associations remain pending. These are separate from production throughput. |
| Native production/module work | [Original scene/module report](docs/NATIVE_PRODUCTION_PERFORMANCE_20261019.md) retains failed motion conformance and admitted replay measurements. The [minimal MPR repair](docs/NATIVE_CONTACT_CONSISTENCY_20261019.md) passes349 B1/B8/B32 conformance checks and three focused regressions; all15 repaired-source timing receipts pass. B128/B512 and B2048/B8192 each add159 checks and total process-memory profiles; diverse DPR passes while diverse SCR retains certified infeasible loads. [B8/B128 sparse-output timings](docs/NATIVE_OBSERVATION_CADENCE_20261019.md) pass521 source/motion/output/reset checks. Final physical-error/throughput consolidation remains open. |

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
