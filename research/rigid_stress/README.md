# Native rigid stress recovery

The latest [dimensional normal-error audit](docs/NORMAL_ABSOLUTE_ERRORS_20261019.md)
adds81 absolute/relative metrics across the nine admitted normal comparisons,
including complete-column membrane and bending RMS in N/m and N. Fine friction
qualification, prescribed-load convergence and final cross-case delivery remain
open; this reporting completion does not close the active goal.

The feature is part of the existing native rigid solver. It reconstructs
conditional elastic stress from changing rigid contacts, including friction,
and supports raw global max/full output, complete numerical checks, environment
isolation and partial resets. Production numerical work remains in Quadrants.

The active [qualified stress-ranking goal](docs/QUALIFIED_STRESS_RANKING_GOAL.md)
requires actual DPR / SCR / Shell error rankings, reference uncertainty,
matched speed and stability evidence on the full-egg parallel-plate benchmark.
It authorizes minimal, separately labeled SCR/Shell/reference repairs while
preserving original baselines. The previous bounded plate campaign diagnosed
failures; it did not finish physical accuracy qualification. Diagnostic closure
or unsupported quantities alone cannot complete the current goal.
Full Franka physical-reference qualification remains out of scope.

The [current execution record](docs/QUALIFIED_STRESS_PROGRESS_20261019.md)
tracks the verified SCR repair and explicitly identified coupled Shell controls.
The [normal-pad qualification](docs/NORMAL_PAD_QUALIFICATION_20261019.md) now
passes its reference-change gates and reports conditional metric-specific
comparisons. Independent friction refinement and matched primary cost/batching
qualification are tracked in the updates below. Partial successes do not close
the goal.

The [common-peak/batch update](docs/COMMON_PEAK_BATCH_RESULTS_20261019.md)
adds qualified common peaks, actual B4 motion/source conformance, B1 every-step
full/max primary times and an inspected force/closure figure. The sparse/B4
cost matrix and spatial deliverables are updated in the later checkpoint below.

The subsequent [normal spatial/cost update](docs/NORMAL_SPATIAL_COST_20261019.md)
completes the inspected normal stress/actual-pressure maps and B1 every-step/
sparse full/max cost matrix. Fine sustained-friction spatial qualification,
the complete B4/memory matrix and final cross-case comparison remain open.

The [friction resolution/memory checkpoint](docs/FRICTION_RESOLUTION_MEMORY_20261019.md)
adds true250um Shell sustained-friction histories, exact nonlinear-reference
caching controls and separately admitted B1/B4 full/max process memory.
Shell timestep sensitivity is small, but500-to250um complete-field changes
are large; the fixed250um normal comparison is not mesh-independent Shell
accuracy. Independent friction spatial qualification remains in progress.

The [complete normal cost matrix](docs/COMPLETE_PAD_COST_20261019.md) now joins
24 admitted B1/B4 every-step/sparse full/max timings and12 distinct process-memory
profiles, with591 verified audit inputs. Separately admitted synchronized phase
profiles retain their own scope and do not replace primary wall times.

The [friction reference/observation checkpoint](docs/FRICTION_REFERENCE_OBSERVATION_20261019.md)
adds a complete500um nonlinear trajectory and an admitted Shell1000-step
friction cost. Spatial changes still exceed target, and DPR every-step
observation rejects lagged tangential forces at zero-normal-force sources.
The failure is retained; earlier endpoint-only fields do not admit that run.

The [keyframe cost and response-reuse update](docs/FRICTION_KEYFRAME_COST_REUSE_20261019.md)
adds admitted1000-step friction costs with full-field observation every200 steps
for all three methods. Both native every-step configurations fail during initial
contact; actual SCR inputs have a separately certified surface-cone incompatibility.
Optional reference response reuse passes complete-history and field controls.
Fine nonlinear spatial qualification and the final friction comparison remain open.

The [common-sampling control update](docs/FRICTION_COMMON_CONTROLS_20261019.md)
adds24 passed checks for coarse nonlinear time/contact controls at q9/q25,
including eight independently rehashed endpoint fields. Normal material-model
contrasts at0.3/1N and fine spatial/method comparisons remain pending;
fixed tasks and unchanged acceptance bands are retained separately.

The [250um nonlinear checkpoint](docs/FRICTION_SPATIAL250_20261019.md) adds a
completed two-wall trajectory with passing mechanical/momentum checks and
q9/q25 surface-refinement comparisons. The largest field change remains2.749%,
above the2% target. Audited plotting inputs retain complete histories and actual
contact forces; final fine-reference qualification and friction figures remain
pending. See the [figure contract](docs/FRICTION_FIGURE_PLAN.md).

The [reconstruction/material update](docs/NATIVE_RECONSTRUCTION_MATERIAL_20261019.md)
adds exact native quadrature and10-state independent integration/core agreement,
measured footprint broadening and SCR plate-side redistribution, plus admitted
500um normal material controls at0.3/1N. Prescribed-C2 spatial qualification is
still incomplete. The [completion checklist](docs/QUALIFIED_STRESS_COMPLETION_CHECKLIST.md)
keeps the remaining comparisons, absolute-error tables and final figures explicit.

The [plate report](docs/PARALLEL_PLATE_RESULTS_20261018.md) records verified
mechanical equilibria, the SCR cancellation-sensitive admission failure,
original Shell contact-split imbalance and unresolved QIPC point-contact fields.
It does not qualify a three-way physical accuracy ranking. See the
[requirement audit](docs/PARALLEL_PLATE_AUDIT.md),
[running guide](docs/RUNNING_PARALLEL_PLATES.md) and
[closeout packet](evidence/20261018-parallel-plates/closeout-v1/README.md).

`RigidStressOptions.dpr(...)` and `.scr(...)` select the existing pressure and
intrinsic/body_source_prior routes. Manual stress sampling is available through
`update_mode="manual"` and `link.request_stress_update()`; ordinary every-step
sampling remains the default. See [API](docs/NATIVE_API.md).

The preceding [Franka/QIPC report](docs/QIPC_REFERENCE_RESULTS_20261018.md),
[running guide](docs/RUNNING_QIPC_REFERENCE.md),
[progress](docs/QIPC_REFERENCE_PROGRESS.md) and
[sparse checkpoint](evidence/20261018-qipc-reference/sparse-keyframes-v1/README.md)
retain useful implementation, cost, failed reference qualification and
repeatability evidence. Those historical results are not accepted physical
ground truth; reuse their controls and data for the simpler active benchmark.
Existing Franka throughput remains separately labeled engineering evidence.

The [shell PR22 comparison](docs/SHELL_COMPARISON_RESULTS_20261018.md) is complete
within its [declared scope](docs/SHELL_SOLVER_COMPARISON_GOAL.md). It pins both
implementations, defines compatible full fields, executes analytic/same-load
and contact/Franka controls, and measures full-field cost on matched hardware.
Its [running guide](docs/RUNNING_SHELL_COMPARISON.md),
[audit](docs/SHELL_COMPARISON_AUDIT.md) and
[evidence packet](evidence/20261018-shell-comparison/comparison-v1/README.md)
preserve exact source/input/hardware identities, failures and editable figures.

At B4096, full-output native pressure/tuned/intrinsic achieve239264/255089/99024
numerically valid aggregate env-step/s. Original shell32/64 measure36282/24885
diagnostic env-step/s; no accuracy-qualified shell grasp setting was established.
Shell deformation feedback remains an additional capability, while its contact,
time integration and state-history limitations are documented. The native
production engine and previous accepted evidence remain unchanged.

The current [pressure/intrinsic comparison](docs/PRESSURE_COMPARISON_RESULTS_20261018.md)
is complete within its declared scope. Its [running guide](docs/RUNNING_PRESSURE_COMPARISON.md)
and [compact evidence packet](evidence/20261017-pressure-comparison/comparison-v1/README.md)
provide current paired measurements, configuration choices, model limits,
source/input/hardware identities and inspected editable figures.

At B16384 on RTX PRO6000 Blackwell, the default max/history0 pressure route
measures296859 live/296297 policy valid aggregate env-step/s, versus
113933/113971 for intrinsic body_source_prior. Separate matched pressure tuning
confirms6.68%/6.38% throughput gains, with about21.8GiB sampled process peaks.
These are inference rollouts; raw repeats, matched rigid overhead, memory,
max/full and capacity evidence are in the report.

Choose pressure as a declared fast proxy where local-normal and topology
limitations are acceptable. Prefer intrinsic body_source_prior when those
constraints matter. Both retain substantial footprint/reference uncertainty;
neither calibrates an absolute physical damage threshold. The matched two-model
comparison and separate radius/reward sensitivity support conditional choices.

Start with [API](docs/NATIVE_API.md), [running instructions](docs/RUNNING.md),
[source-prior model](docs/BODY_SOURCE_PRIOR_MODEL.md) and [goal/status](GOAL.md).
[Application results](docs/PRACTICAL_APPLICATION_RESULTS_20261016.md) retain
independent FEM/R/L/M and feedback limits. The preceding
[throughput study](docs/PRACTICAL_THROUGHPUT_RESULTS_20261017.md) retains distinct
code/history comparisons and separately tuned intrinsic capacity. Its older
rates are not substitutes for the current paired pressure measurements.

Tests and useful reference modules remain at their original import paths.
[Cleanup results](docs/BRANCH_CLEANUP_RESULTS.md) and [archive](archive/README.md)
preserve historical identities, failed cases and unresolved research without
reviving old campaigns. The original seven canonical packets remain unchanged.
