# Working on native rigid stress recovery

**Execution-agent handoff,2026-10-18:** the primary accuracy benchmark is now
two parallel rigid plates compressing the hollow egg. Read
[the new goal](docs/PARALLEL_PLATE_COMPARISON_GOAL.md) before further long jobs.
Start with common-force quasi-static loading, then bounded friction controls.
Full Franka reference/controller/convergence work is no longer a completion
gate; keep its existing results and useful engineering cost evidence.

Read [GOAL.md](GOAL.md), [docs/PARALLEL_PLATE_COMPARISON_GOAL.md](docs/PARALLEL_PLATE_COMPARISON_GOAL.md),
[README.md](README.md) and [docs/RUNNING.md](docs/RUNNING.md) first.
Compare DPR, SCR and original PR22 Shell against qualified QIPC solid FEM.
Preserve occasional full-spatial keyframes, short-window stage profiling,
measured tolerance selection, continuous mechanical history and failure checks.
Assess already running work at a safe boundary without hot-editing captured
sources or discarding evidence. Do not restart broad Franka qualification sweeps.

Reuse the preceding [shell report](docs/SHELL_COMPARISON_RESULTS_20261018.md),
[running guide](docs/RUNNING_SHELL_COMPARISON.md) and
[requirement audit](docs/SHELL_COMPARISON_AUDIT.md). Prior pressure/intrinsic
evidence and cleanup remain accepted within scope. Pin each source; preserve
the unmodified external baselines and distinguish diagnostic/repaired variants.
Historical campaigns do not create unrelated active tasks.

Follow the storage exception in [DATA_LOCATION.md](../../DATA_LOCATION.md):
the user explicitly permits this task's existing home fallback within its4GB
persistent-output bound while data remains unwritable. New submissions must
record `--allow-home-output`; private caches stay node-local. This exception
does not alter other projects or authorize an unbounded home-data sweep.

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
