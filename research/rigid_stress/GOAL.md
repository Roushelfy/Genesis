# Native rigid stress recovery: local-refinement atlas follow-up

**Completed, 2026-10-26:** genuine shared-coarse/local Galerkin spaces,
per-feature source aggregation and merged recovery have native implementation,
numerical controls and separate shell/volume decisions. Read
[the measured report](docs/LOCAL_REFINEMENT_RESULTS_20261026.md),
[the execution ledger](docs/LOCAL_REFINEMENT_PROGRESS_20261026.md) and
[the six-cell evidence packet](evidence/20261026-local-refinement-v1/README.md).

Smaller feature operators pass original numerical, common-mother, rank and
virtual-work controls. Bounded spatial tuning, complete Franka max/full
accuracy/cost, heterogeneous/lifecycle controls, profiling and memory show no
useful complete-path gain on the measured models. Aggregation and one recovery
have useful internal gains, but depend on the unretained atlas. Experimental
code remains runnable at `822ad2854`; production code matches `4bc4ada25`.
Twenty cleanup regressions pass. Earlier pressure/fallback/angular winners
and the scoped H/FP32/screen/history negatives remain preserved.

The [review and delivered specification](docs/ATLAS_LOCAL_REFINEMENT_REVIEW_AND_GOAL.md)
and [CPU source guide](docs/ATLAS_CPU_REFERENCE_GUIDE.md) explain this follow-up.
Its real smaller spaces, separate numerical/spatial gates and tuned negative
results close the three reopened families in both backends. This does not
qualify arbitrary nonconvex spatial surrogates or continuum sharp-edge peaks,
and does not rule out other reduced-space constructions. Avoid unchanged reruns
unless a new user goal supplies new scope.

## Historical candidate delivery, scoped by the current review

**Completed, 2026-10-26:** all fourteen native candidate/backend decisions are
delivered. Native integration, numerical controls, targeted tuning, module and
complete Franka comparisons, memory and production cleanup are complete.
None of the seven incremental candidates provides a reproducible useful gain
in the measured regimes; their runnable native code and negative evidence are
retained. The maintained engine preserves the prior pressure/fallback winners
and adds the independently qualified optional angular/q128 pressure repair.
Read [the final report](docs/NATIVE_CANDIDATE_RESULTS_20261025.md),
[the seven-by-two ledger](docs/NATIVE_CANDIDATE_PROGRESS_20261025.md) and
[the requirement audit](evidence/20261025-native-candidates-v1/metadata/requirements-final.json).
Same-model checks do not qualify continuum sharp-edge peaks; broad physical
convergence remains outside this delivered integration/throughput goal.

**Delivered specification, 2026-10-25:** take all seven GPU candidates through
real native integration, targeted tuning and matched end-to-end measurement
for both midsurface shell and three-dimensional volume FEM. Track the seven
candidates separately for each backend; neither backend's result closes the
other. Each has its own mother space, gauge/relief, load maps and outputs.
Follow [the integration goal](docs/NATIVE_CANDIDATE_INTEGRATION_GOAL_20261025.md).

The curved atlas mapping and common gauge/inertia relief are now explicit
implementation work. Simplified planar, H-only or recovery-only prototypes
cannot close these items. Connect each candidate to real native inputs,
required acceptance/state, max/full output and lifecycle. Keep measured
winners; revert production candidates only after a reasonably tuned native
comparison supports the negative result.

Keep all production numerical preparation/runtime in Quadrants. Preserve
the mature DPR load model and arbitrary moving/asymmetric convex/nonconvex
contacts, friction, surface sides, batching, volume/shell choice and lifecycle.
Use low-resolution numerical consistency and full Franka throughput first.
The [earlier candidate report](docs/GPU_THROUGHPUT_20261025.md) is the baseline
and historical evidence, with its unintegrated candidates still open under
this stronger goal.

---

## Previous delivery: GPU candidate evaluation

**Completed, 2026-10-25:** the applicable candidates, integrated pressure and
exact-fallback improvements, focused regressions, paired module/scene timings,
memory and negative results are delivered. Read [the report](docs/GPU_THROUGHPUT_20261025.md)
and [the requirement audit](evidence/20261025-gpu-throughput-v1/metadata/requirements-final.json).
The heterogeneous volume bank retains the same frame-221 pressure-fit rejection
before and after optimization; no successful rate is claimed for that workload.
Curved-mesh atlas mapping and continuum contact peaks remain unqualified.

**Delivered specification, 2026-10-25:** implement and evaluate the applicable GPU
optimizations in [the follow-up goal](docs/GPU_THROUGHPUT_FOLLOWUP_GOAL_20261025.md).
Maximize useful multi-environment throughput in the existing rigid solver,
with production numerical preparation and runtime entirely in Quadrants.

Cover shared load preparation and multi-RHS grouping; few feature meshes per
contact with continuous transitions; common-mother recovery and max/full
fusion; complete/packed response operators with independently tested FP32 and
other supported precisions; conservative global max screening; history
prediction, failed-environment compaction, buffer rotation and cached Kq;
and applicable condensation/reduction controls. Measure each candidate's
incremental value against the already optimized native implementation.
Prioritize the actual profile, including pressure integration and fitting.

Preserve mature DPR loads, arbitrary moving/asymmetric convex/nonconvex
contacts, normal/friction eccentricity, surface sides, volume/shell choice,
max/full output, numerical acceptance, batching and lifecycle behavior.
Use coarse meshes for numerical consistency and throughput tuning before
new high-resolution physical convergence work. Deliver usable improvements,
paired module/Franka scene benchmarks, memory, recommendations and a complete
candidate ledger; document negative and inapplicable outcomes.

This explicit follow-up supersedes the previous completion/atlas-only scope
for the candidates above. Earlier delivered results remain accepted within
their recorded scope.

---

## Previous delivered goal: native thin-shell DPR integration

**Completion, 2026-10-24:** the native implementation, focused regressions,
complete-scene/module throughput, profiles, memory, configuration comparisons
and targeted global physical checks are delivered. Read
[the final report](docs/SHELL_DPR_NATIVE_20261024.md) and
[the requirement audit](evidence/20261024-native-shell-dpr-v1/metadata/requirements-final.json).
The task is complete within that scope; local three-dimensional contact peaks
remain explicitly unqualified. The requirements below record the delivered goal.

**User direction, 2026-10-24:** implement an optional thin-shell elastic-recovery
backend inside the existing rigid stress feature, entirely in Quadrants.
Follow [the active goal](docs/SHELL_DPR_INTEGRATION_GOAL_20261024.md).

Keep mature native automatic-radius + finite-pressure DPR and its numerical
checks. Preserve arbitrary changing convex/nonconvex and asymmetric contacts,
normal/friction forces including eccentric moments, max/full outputs, batching,
isolation and lifecycle behavior. Keep volume FEM available.

The gripper box keeps ideal sharp edges for the current implementation, as
confirmed by the user. Fillet/chamfer dimensions are not a prerequisite; any
reported edge peak must retain its sampling and convergence limitations.

Start from the completed [CPU study](shell_cpu/README.md) and
[algorithm notes](shell_cpu/ALGORITHM.md). First integrate a low-resolution
native path and verify numerical consistency. Prioritize detailed profiling
and all applicable throughput optimization on the full Franka/hollow-egg
workflow, across small and useful large batches. Keep high-resolution physical
convergence campaigns behind this performance phase.

Report shell in-plane membrane/bending stress honestly: the CPU tiny-contact
egg peak is unresolved and does not certify a three-dimensional maximum.
Preserve the native DPR load law; the main CPU projected-Hertz variant is not
an exact native transfer replacement.

## Previous delivery

The volume-DPR throughput task is complete within its measured numerical and
performance scope. Retain [its report](docs/DPR_THROUGHPUT_20261023.md) as the
baseline and [its requirements](docs/DPR_AUTOMATIC_RADIUS_GOAL_20261023.md) as
historical context. Earlier conflicting enrichment, SCR-repair, physical
ranking and sharp-edge campaigns do not override the active shell goal.

Follow existing storage and reproduction policies. Deliver usable native code,
focused tests, matched module/scene timings, memory, recommended configurations
and explicit accuracy limitations; documentation alone does not complete the goal.

