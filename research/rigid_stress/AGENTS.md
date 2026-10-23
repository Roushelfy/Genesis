# Working on native rigid stress recovery

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
