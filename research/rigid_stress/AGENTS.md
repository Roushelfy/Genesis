# Working on native rigid stress recovery

**Active user goal, 2026-10-28:** implement [automatic DPR fast path, defaults and examples](docs/AUTOMATIC_DPR_FAST_PATH_DEFAULTS_GOAL_20261028.md).
Read [GOAL.md](GOAL.md) first. Diagnose volume edge/q16 failures, evaluate practical
stress-error budgets and fit/load tolerances, and prefer a qualified cheap path
with selective repair over global q128. This user goal explicitly permits
evidence-backed changes to pressure-fit/load acceptance; older instructions to
keep every such threshold unchanged do not prohibit this work. Preserve strict
controls and do not weaken unrelated elasticity/material/lifecycle checks.

Make automatic-radius + Hertz-pressure DPR the actual API/factory/workload/CLI
default, preserving explicit manual/legacy and SCR alternatives. Volume remains
the backend default; retain Q4 loading by default for shell and show the qualified
planar recipe explicitly. Add and run simple 3D/shell, max/full and batched Franka
examples; [the current examples](docs/RUNNING_AUTOMATIC_DPR.md) are explicit
configurations pending that migration. Keep native production numerics in Quadrants,
the mature load model, arbitrary contacts/friction and existing planar winners.
Use focused low-model accuracy, actual profiles and matched full-path throughput.
Avoid reopening unchanged prior experiments or turning this goal into a broad
convergence/archival campaign. Standing preservation/storage policies still apply.

## Previous completed planar goal

**Completed user goal, 2026-10-27:** native [planar loading DPR](docs/PLANAR_LOADING_DPR_NATIVE_GOAL_20261027.md).
Read [GOAL.md](GOAL.md), [the final report](docs/PLANAR_LOADING_RESULTS_20261027.md),
[the execution ledger](docs/PLANAR_LOADING_PROGRESS_20261027.md) and
[the 29-clause packet](evidence/20261027-planar-loading-v1/README.md).

Preserve the original Q4 elastic model, original-geometry Hertz sizing,
source/friction moments, surface-ball pressure law, material identity, original
acceptance and lifecycle. Native floating preparation/runtime stay in Quadrants;
CPU79720355 remains independent. Measured engine is6dedc7957. Q4 remains the
public default; the complete measured-workflow planar recipe explicitly selects
k32/q10/radial/blocks/cap128/FP64/history0 and process-local QD_GRAPH=0 before
imports. The opt-in k8/cap32 defaults have a separate fidelity scope.

All168 primary processes/48 blocks/16 cohorts,28 memory workers,14 profiles and26
focused regressions pass. B32768 complete policy max/full gains are2.033/1.987;
B4 frozen-module slowdown, greater planar memory, prior negatives and sharp-edge
continuum limits remain explicit. Policy-max growth10.0479906723% fails the
frozen10% plateau criterion; the optional65536 memory condition fails, so no
probe or universal throughput maximum is claimed. The packet's independent
audit preserves all168 original repeats/vectors and recomputes paired timings.

The requested goal is closed. Retain useful production changes and restoration
pins for removed experiments. Do not reopen unchanged prior Atlas/H/FP32/screen/
history or expensive continuum campaigns without new user scope or a concrete
changed-workload/profile reason. All original twelve unfinished files and33 old/
31 new frozen sources remain preserved; standing policies below still apply.

**Previous delivery, 2026-10-27:** the matched CPU12 GPU follow-up is complete.
Read [GOAL.md](GOAL.md), [the specification](docs/ATLAS_CPU12_GPU_PARITY_GOAL_20261026.md),
[the matched GPU report](docs/CPU12_GPU_RESULTS_20261026.md),
[the native backend report](docs/CPU12_NATIVE_RESULTS_20261026.md) and
[the thirteen-cell packet](evidence/20261026-cpu12-gpu-parity-v1/README.md).

CPU/GPU equivalence is positive for the actual mother/P/B/operators/loads and
all banks/selectors, with original reduced residual and lifecycle contracts.
The CPU construction and bounded finer-coarse controls fail existing spatial
M/Q budgets. Cooperative/packed solves are validated internal improvements;
the matched complete module is slower than the existing optimized inverse
mother. Native twelve-center shell/volume have independent numerical/spatial,
bounded tuning, complete workflow, profile and memory decisions. Shell Q and
nonconvex remote hotspots retain spatial negatives. Volume passes the specified
four/64-condition trajectories but its complete workflow remains slower.

Keep the optimized mother default and the mature DPR pressure/fallback/angular
paths. Native experimental code is pinned at `94ec782fd`; restore it explicitly
from the report for replay. Retain reusable matched-QD numerical code and the
validated cooperative factor limit. Twenty focused regressions pass. The
measured workflow is policy inference with a seeded random MLP and scripted
residual controller, not policy training. Preserve original twelve unfinished
files, old twenty-three engines, this campaign's frozen sources and all prior
scoped evidence. The specification permits tuned negatives; do not reopen the
same comparison without new user scope. Standing policies below still apply.

**Previous delivery, 2026-10-26:** the real local-refinement follow-up is complete.
Read [GOAL.md](GOAL.md), [the measured report](docs/LOCAL_REFINEMENT_RESULTS_20261026.md)
and [the six-cell packet](evidence/20261026-local-refinement-v1/README.md).
Both backends have actual smaller `T_i=[P,B_i]`/`K_i`, native numerical controls,
combined spatial qualification, bounded tuning, component increments, complete
Franka max/full timing, heterogeneous/lifecycle checks and memory evidence.

The local-space implementation has no recommended full-path regime in the
measured fixtures. Source aggregation and one recovery are beneficial internal
components; their atlas dependency was removed from production. Runnable code
is pinned at `822ad2854`, including the validated cooperative selector.
Production `genesis`/`examples` match `4bc4ada25`; twenty focused cleanup
regressions pass. Preserve twelve original unfinished files, frozen sources,
prior pressure/fallback/angular improvements and H/FP32/screen/history results.

The [review/specification](docs/ATLAS_LOCAL_REFINEMENT_REVIEW_AND_GOAL.md) and
[CPU source guide](docs/ATLAS_CPU_REFERENCE_GUIDE.md) remain provenance for the
delivered goal. Its first three families are closed for this tested mechanism;
avoid unchanged reruns or restarting it without a new user goal. Numerical
acceptance in a declared reduced space remains separate from spatial error.
The full mother residual is a spatial diagnostic for an approximate result.
The result does not qualify arbitrary nonconvex surrogates, continuum sharp-edge
peaks or every possible reduced-space construction. Standing policies below apply.

## Historical candidate delivery and standing policies

**Latest delivery, 2026-10-26:** the native candidate integration/tuning goal
is complete for both shell and volume. Read [GOAL.md](GOAL.md),
[the final report](docs/NATIVE_CANDIDATE_RESULTS_20261025.md) and
[the fourteen-cell requirement audit](evidence/20261025-native-candidates-v1/metadata/requirements-final.json).
All seven candidates have native numerical, tuning, module/complete-scene and
memory evidence in each backend. Their incremental production paths were
removed after tuned negative results; runnable source pins remain in Git.
Preserve the earlier shared pressure preparation, Q4 projection reuse and
exact fallback compaction. Optional angular/q128 integration now admits and
passes independent field checks for the retained 64-condition volume bank;
its cost and lower-degree failures remain separate from scheduling results.

Keep FP64/history-off recommendations, original residual/load/lifecycle
contracts and the small maintained public configuration. The measured
four-condition fixture uses edge/q16; the heterogeneous volume qualification
uses angle/q128. These numerical controls do not qualify continuum sharp-edge
peaks or reopen high-resolution physical convergence. Follow the standing
storage, source-preservation and engineering policies below.

## Previous deliveries

**Latest delivery, 2026-10-25:** the GPU throughput follow-up is complete.
Read [GOAL.md](GOAL.md), [the measured report](docs/GPU_THROUGHPUT_20261025.md)
and [the requirement audit](evidence/20261025-gpu-throughput-v1/metadata/requirements-final.json).
The integrated winners are shared shell-pressure face preparation, fixed Q4
projection reuse and large-batch exact failed-environment compaction. Keep
FP64/q16/history-off recommendations, original acceptance/lifecycle behavior
and all retained negative evidence. The heterogeneous volume trajectory still
rejects the same frame-221 pressure fit as its baseline; this workload has no
qualified complete-cycle volume rate. Few-mesh/common-mother GPU controls remain
conditional planar prototypes, not validated arbitrary curved native atlases.
No implicit physical-reference or atlas integration task is reopened by this
delivery. Standing storage, source preservation and engineering policies below
remain in force.

**Latest delivery, 2026-10-24:** the optional native thin-shell DPR backend,
focused numerical/lifecycle tests, matched throughput/profiling and targeted
global physical checks are complete. Read [GOAL.md](GOAL.md) and
[the final report](docs/SHELL_DPR_NATIVE_20261024.md).
Keep mature DPR loading; production numerical work stays in the existing
rigid solver in Quadrants. Preserve volume FEM, arbitrary moving/asymmetric
convex/nonconvex contacts, normal/friction eccentricity, max/full and lifecycle
behavior in future changes. The delivered sequence was coarse integration,
CPU/native consistency, detailed performance tuning, then targeted physical
checks. Shell in-plane results do not certify local three-dimensional contact
peaks; that unresolved quantity is not a hidden completion claim.

This closes the shell integration priority. The volume-DPR deliverable is
also complete. Older enrichment, SCR-repair and physical-ranking directions
below remain historical, not active follow-up tasks. The atlas CPU study is
supplementary research, not an implicit native implementation request.
Keep historical evidence and existing storage/reproduction policies.

## Historical directions and standing engineering policies

**Immediate user goal, 2026-10-23:** maximize throughput of the now-integrated
automatic-radius + Hertz-pressure DPR path. Follow
[the optimization goal](docs/DPR_AUTOMATIC_RADIUS_GOAL_20261023.md#immediate-goal-maximize-native-automatic-dpr-throughput).
Prioritize contained-disk analytical P2 integration, contact-face/geometry and
refined-moment reuse, duplicate/zero-work elimination, fit/scheduling and batch
tuning. Measure max/full complete-scene gains against matched manual DPR and
pure rigid. Preserve the load model and numerical checks; use low-resolution,
focused validation without waiting for broad physical convergence campaigns.

**Highest-priority user correction, 2026-10-23:** follow
[automatic-radius DPR](docs/DPR_AUTOMATIC_RADIUS_GOAL_20261023.md) and
[GOAL.md](GOAL.md). Keep the mature DPR and its prefactorized FEM; add a cheap
contact-radius/support estimate. Stop further analytic particular-field,
boundary-traction, combined-stress/probe and associated singular-peak expansion.
Keep earlier evidence as historical. Existing normal/friction handling and DPR
checks remain; do not make new friction solvers or sharp-edge reference
qualification prerequisites for this minimal feature. Deliver and measure the
actual automatic-radius DPR on this branch. This overrides conflicting older
instructions below, including broad sharp-edge and enrichment research.

**Latest geometry direction, 2026-10-22:** proceed with sharp edges first.
This supersedes the preceding fillet/chamfer dimension wait. Continue matched
sharp-box collision/pressure geometry, independent physical checks and cheap
native implementation. Report sampling depth/resolution and edge convergence
with stress peaks; retain the actual-geometry inspection for a later variant.
Unknown fillet dimensions do not block the current work.

**Highest-priority user direction, 2026-10-20:** follow the
[cheap hard-contact goal](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md)
and [GOAL.md](GOAL.md). Prioritize inexpensive Hertz/thin-shell approximations
and measured native throughput. Do not assume or further develop a deformable
gripper pad. Keep the coupled pressure/Newton-PCG path optional unless independently
verified physical accuracy and measured cost justify it; no default heavy
fallback or prerequisite integration. Preserve prior pad evidence as historical
scope, not hard-contact accuracy. This overrides earlier pad/coupled-integration
priorities while retaining practical stress-error budgets and strict baselines.

**Latest throughput follow-up:** execute the
[contact-input optimization plan](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md#immediate-priority-reduce-contact-input-overhead).
Profile before attributing costs; reuse collision provenance, continuous-contact
local caches and fixed-geometry preprocessing. Measure fusion, grouping and
batch-scaling improvements with equivalent validity/numerics and complete
max/full scene accounting. Preserve nonconvex identity and history/reset
invalidation. Keep research audits outside production timing and low-resolution
performance work active without waiting for broad physical convergence sweeps.

**Latest user direction, 2026-10-19:** separate accuracy verification from
performance timing. Read [the protocol](docs/ACCURACY_AND_THROUGHPUT_PROTOCOL.md).
Run low-resolution native module and end-to-end throughput tuning now; do not
wait for fine FEM convergence. Exclude research-only audits/exports from
production timing, keep required runtime acceptance, and bind accuracy and
speed to matching configurations. Preserve every-step friction failures.

**Execution-agent handoff, 2026-10-18:** finish qualified DPR / SCR / Shell
stress-error rankings on the simple full-egg parallel-plate benchmark. Read
[the active goal](docs/QUALIFIED_STRESS_RANKING_GOAL.md) before further work.
The bounded diagnostic study is complete; the physical comparison is not.
Its unsupported-quantity clause cannot close this new goal.

Read [GOAL.md](GOAL.md),
[docs/QUALIFIED_STRESS_RANKING_GOAL.md](docs/QUALIFIED_STRESS_RANKING_GOAL.md),
[README.md](README.md) and [docs/RUNNING_PARALLEL_PLATES.md](docs/RUNNING_PARALLEL_PLATES.md).
Repair SCR's two rejection mechanisms, establish a spatially resolved physical
reference and obtain admitted Shell fields. Necessary minimal engine repairs
in isolated pinned variants are authorized; keep original baselines and label
all variants. Preserve complete-field keyframes, measured admission and
uncertainty, short profiling/CPU controls and continuous mechanical history.
Full Franka qualification remains outside this goal. Assess running jobs at
a safe boundary without hot-editing captured sources or discarding evidence.

Reuse the preceding [shell report](docs/SHELL_COMPARISON_RESULTS_20261018.md),
[running guide](docs/RUNNING_SHELL_COMPARISON.md) and
[requirement audit](docs/SHELL_COMPARISON_AUDIT.md). Prior pressure/intrinsic
evidence and cleanup remain accepted within scope. Pin each source; preserve
the unmodified external baselines and distinguish diagnostic/repaired variants.
Historical campaigns do not create unrelated active tasks.

Follow [DATA_LOCATION.md](../../DATA_LOCATION.md). The latest successful data
probe restored data as the destination for new plate runs. Keep private caches
node-local and preserve existing outputs in place. The historical authorized
home fallback remains bounded at4GB; it is not the preferred destination and
does not authorize an unbounded home-data sweep.

Follow root CLAUDE.md and CODING_GUIDELINES.md when editing engine code.
The feature belongs to Genesis's existing rigid solver; all production numerical
work remains in Quadrants. Preserve arbitrary convex/nonconvex changing contacts,
normal force and friction, global max/full output, numerical acceptance,
environment isolation, history invalidation and partial resets.

Keep active entrypoints short. Follow dependencies before moving research code,
tests, assets or wrappers. Preserve reproducible current results and historical
evidence through [the archive index](archive/README.md); do not duplicate large
artifacts or rewrite shared Git history. Validate affected imports, commands,
links and focused tests. Unchanged code does not require old expensive sweeps.

Routine reversible cleanup, tests, commits and updates to this task branch are
authorized. Preserve unrelated work and report unavailable hardware checks.
