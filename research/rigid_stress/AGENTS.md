# Working on native rigid stress recovery

**Latest user direction, 2026-10-20:** follow the
[automatic contact-region and throughput goal](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md).
Finish current fixed-radius rankings, run a compact CPU comparison of physical
region predictors and shell/pad compliance, address the distinct current SCR
failure classes, and profile/tune native throughput in parallel. Prefer DPR for
first integration, keep production in Quadrants and preserve changing nonconvex
normal/friction contacts. Validated recovery infrastructure is available; focus
on actual results instead of expanding generic orchestration/audit frameworks.

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
