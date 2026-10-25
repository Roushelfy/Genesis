# Native rigid stress recovery: GPU throughput follow-up

**Active user goal, 2026-10-25:** implement and evaluate the applicable GPU
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

