# Active goal: qualified stress-error rankings

**2026-10-20 error-budget clarification:** follow the
[error judgments and practical acceptance budgets](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md#error-judgments-and-practical-acceptance-budgets).
Prioritize DPR boundary/load fidelity and CPU stress-sensitivity measurements for
nearby feasible SCR loads. Exact infeasibility is not an engineering-error rank.
Preserve strict results while evaluating explicitly labelled approximations whose
load and stress corrections are bounded; do not blindly retain1e-8 or loosen tests.

**2026-10-20 priority update:** execute the
[contact-region, SCR robustness and throughput goals](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md).
Finish the existing fixed-radius comparison while independently testing automatic
physical contact regions on CPU and optimizing low-resolution native throughput.
Prioritize DPR for practical integration; diagnose certified SCR incompatibility
separately from detachment roundoff/input-law failures. Migrate useful validated
contact models into the existing Quadrants rigid feature. Keep current baselines,
running captures and all previous accuracy/performance requirements.

Follow [docs/QUALIFIED_STRESS_RANKING_GOAL.md](docs/QUALIFIED_STRESS_RANKING_GOAL.md).

**2026-10-19 priority update:** follow the
[separate accuracy/throughput protocol](docs/ACCURACY_AND_THROUGHPUT_PROTOCOL.md).
Advance native production throughput and tuning now, independently of costly
reference qualification. Measure contact-replay module throughput and actual
pure-rigid / rigid-plus-max / rigid-plus-full scene throughput separately.
Join errors and speed only for matching configurations; retain production
validity checks and all failed every-step friction observations.

Finish measured DPR / SCR / Shell stress-error rankings, uncertainty, speed
and stability on the simple full-hollow-egg parallel-plate benchmark. Repair
SCR's balanced-load acceptance and shear fit, obtain an admitted Shell
configuration with consistent contact/elastic coupling, and establish a
spatially resolved finite-area-contact reference using QIPC/independent FEM.
Keep original external baselines and label every repaired variant.

The preceding [bounded plate report](docs/PARALLEL_PLATE_RESULTS_20261018.md)
localizes the blockers but does not establish physical accuracy. Its
unsupported-quantity completion clause is superseded for this stage.
Completion requires actual qualified comparisons, with ties where uncertainty
prevents separation, and matched admitted cost/stability evidence. Diagnosis
or another closeout report alone is insufficient; remaining blockers keep
the goal in progress.

Retain full-spatial fields at selected equilibria, short profiling and CPU
controls before expensive runs, same-load versus contact-error separation,
and exact sources/inputs/failure records. Distinguish every-step stress cost
from sparse observation. Production recovery stays in the existing rigid
solver and Quadrants, preserving arbitrary moving convex/nonconvex contacts,
friction, max/full output, batching and partial resets.

Reuse [plate reproduction instructions](docs/RUNNING_PARALLEL_PLATES.md) and
the existing diagnostic packet. Full Franka physical qualification remains
out of scope; keep its useful throughput/application evidence separately.
