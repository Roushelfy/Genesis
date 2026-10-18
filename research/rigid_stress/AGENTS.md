# Working on native rigid stress recovery

**Execution-agent handoff,2026-10-18:** the user changed the observation and
execution priorities. Read the user-update section of
[the active goal](docs/QIPC_REFERENCE_COMPARISON_GOAL.md) before further long
submissions: only a few predeclared full-spatial-stress keyframes are needed;
first run short-contact-window timing profiles and tolerance comparisons,
then full-length experiments with sparse stress computation/export. Preserve
continuous mechanics/contact history and failure checks. Dense stress norms
and global temporal peaks are no longer required. Acknowledge this update in
the progress record at the next safe boundary; assess existing jobs without
hot-editing their captured source or automatically discarding their results.

Read [GOAL.md](GOAL.md), [docs/QIPC_REFERENCE_COMPARISON_GOAL.md](docs/QIPC_REFERENCE_COMPARISON_GOAL.md),
[README.md](README.md) and [docs/RUNNING.md](docs/RUNNING.md) first.
The active task compares DPR (pressure), SCR (intrinsic + body_source_prior)
and Kashu shell against qualified cuda-graph-qipc reference grasps. Deliver
common-reference error, speed/memory and stability evidence, with clearer
configuration names. Separate prescribed robot motion from matched capped
actuation, and qualify reference uncertainty before ranking accuracy.

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
