# Active goal: native thin-shell DPR integration and throughput

**User direction, 2026-10-24:** implement an optional thin-shell elastic-recovery
backend inside the existing rigid stress feature, entirely in Quadrants.
Follow [the active goal](docs/SHELL_DPR_INTEGRATION_GOAL_20261024.md).

Keep mature native automatic-radius + finite-pressure DPR and its numerical
checks. Preserve arbitrary changing convex/nonconvex and asymmetric contacts,
normal/friction forces including eccentric moments, max/full outputs, batching,
isolation and lifecycle behavior. Keep volume FEM available.

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
