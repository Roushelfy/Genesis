# Working on native rigid stress recovery

Read [GOAL.md](GOAL.md), [docs/SHELL_SOLVER_COMPARISON_GOAL.md](docs/SHELL_SOLVER_COMPARISON_GOAL.md),
[README.md](README.md) and [docs/RUNNING.md](docs/RUNNING.md) first.
The completed stage compares our full stress outputs with Kashu shell PR22;
read [its report](docs/SHELL_COMPARISON_RESULTS_20261018.md),
[running guide](docs/RUNNING_SHELL_COMPARISON.md) and
[requirement audit](docs/SHELL_COMPARISON_AUDIT.md). No new campaign is active.
Unify mechanical observables, pin both sources, and separate contact instability,
discretization/solve error and physical model differences. Reproduce issues
before claiming them. Keep an original external baseline and label instrumentation
or repairs separately. Prior pressure/intrinsic evidence and cleanup remain
accepted within scope; historical campaigns do not create new active tasks.

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
