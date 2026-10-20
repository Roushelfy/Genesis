# Active goal: cheap hard-contact stress estimates and maximum throughput

Follow [the current hard-contact goal](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md).

**Latest user priority, 2026-10-20:** implement inexpensive Hertz/thin-shell
contact approximations first. Do not assume a deformable gripper pad or extend
the spring-pad study. Expensive automatic region/contact solves are optional
reference work unless they show very accurate physical stress and a worthwhile
measured accuracy/cost advantage. They are not a prerequisite or default fallback.

Target a rigid Franka gripping a hollow egg. Only rest shape/material is fixed;
preserve arbitrary moving convex/nonconvex and asymmetric normal/friction contacts,
global maximum/full output, batching, isolation and partial resets. Keep the
feature inside the Genesis rigid solver, entirely in Quadrants.

Run compact CPU comparisons and low-resolution native performance work in
parallel. Predict a physically motivated pressure/support estimate cheaply,
retain boundary/load fidelity and justified engineering stress-error budgets,
and measure actual module and paired pure-rigid/max/full scene throughput.
Do not infer physical accuracy from numerical-core agreement or global wrench
balance. Match error and speed to the actual configuration.

Reuse the completed fixed-radius/soft-pad comparison as historical evidence,
without inheriting its errors for hard contact or rerunning completed matrices.
Keep the separate [accuracy/throughput protocol](docs/ACCURACY_AND_THROUGHPUT_PROTOCOL.md)
and [engineering error judgments](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md#error-judgments-and-practical-acceptance-budgets).
The [prior ranking goal](docs/QUALIFIED_STRESS_RANKING_GOAL.md) retains its original
scope; its soft-pad/coupled-integration requirements do not override this update.

Deliver a useful cheap native estimator, applicable regimes, complete/peak stress
errors and stability, batch scaling and measured overhead with recommended settings.
Preserve original baselines, failures and source identities. Full Franka physical
reference qualification is not revived; use small discriminating hard-contact
controls and retain the Franka scene for throughput. Keep active entrypoints and
documentation concise. Follow existing data storage/reproduction instructions.
