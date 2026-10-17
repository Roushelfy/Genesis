# Runtime data locations

The unselected `research/rigid_stress/evidence/20261014-physical/five-level-thickness-v1`
publication contained redundant direct copies of the same ten large fields
already represented as exact byte chunks. Its publication process was terminal
before copying it with metadata and symlink preservation to
`runs/20261014-physical-validation/publication-attempt-five-level-thickness-v1-unbounded-extras`
under the data root below. All192 files (3312301696 bytes) were SHA256-verified
before removing only that untracked publication from the checkout. Original
scientific runs, field arrays and publication staging remain intact. The selected
`five-level-thickness-v2` avoids redundant direct extras and retains every original
field through exact reconstructed chunks. The complete old/new paths, active-job
check, per-file hashes and old publisher source are retained in
`runs/20261014-physical-validation/meta/thickness-publication-large-extra-correction-v1.json`.

The rigid stress recovery worktree uses
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery` for runs,
generated collision assets, video, numerical replays and caches. Its optional
environment is `/mnt/data/zhaofeng/venvs/rigid-stress-recovery`.

`research/rigid_stress/scripts/cluster_env.sh` sets these locations explicitly.
Override `RIGID_STRESS_DATA_ROOT` and `RIGID_STRESS_ENV` for another machine.
Dependencies reuse `/mnt/data/zhaofeng/pip_cache`.

The practical large-batch throughput stage uses
`runs/20261017-practical-throughput/` for immutable source snapshots, Slurm
submission records, matched scope measurements, intrusive traces, targeted
tests and metadata summaries. It reuses the eight-condition bank and the
qualified native application snapshot under
`runs/20261014-physical-validation/`; neither input is migrated or rewritten.
Per-job stdout/stderr, temporary files, CUDA/Genesis/Quadrants/Numba caches and
test artifacts are explicitly placed under the data root. The scientific
environment remains `/mnt/data/zhaofeng/venvs/rigid-stress-physical`.

The compact baseline scaling figure packet is tracked under
`research/rigid_stress/evidence/20261017-throughput/baseline-scaling-v1/`.
It contains editable/exported figures, source CSV, exact plotting inputs/source
and validation records. Full traces, raw candidate arrays, failed checker
versions and long-running capacity experiments remain in the data-side run
root above. No runtime output or dependency cache was moved into the checkout.

Small source, configuration, documentation and selected published evidence
remain in this worktree. No existing runtime directories were migrated or
replaced, and compatibility symlinks remain in place. Raw run directories are
listed in `research/rigid_stress/docs/PROGRESS_20261009.md`.

The nonconvex extension uses `runs/20261011-nonconvex/` for its original
qualifications, matched scopes, profiles, scaling and capacity trials.
Final regression, output-mode comparisons, analysis, archive staging and
verification use `runs/20261014-closeout/`. These paths are relative to the
data root above. The selected immutable publication is
`research/rigid_stress/evidence/20261011-nonconvex/`, including
`native-nonconvex-closeout-20261014-v1`. Complete raw JSON and caches stay on
the data mount. Published large JSON uses ordered compressed byte ranges
with full-file reconstruction hashes.

The new load-model/physical-validation stage uses
`runs/20261014-physical-validation/` for independent FEM qualifications,
mesh assets, fine sweeps, reference setup and failed attempts. Its optional
reference environment is `/mnt/data/zhaofeng/venvs/rigid-stress-physical`,
configured through `PHYSICAL_VALIDATION_ENV`. It inherits the existing
validated base site's dependencies via a recorded `.pth` and adds pinned
reference packages without migrating or editing that base environment.

The load-model matrix, refined shell, friction controls and publication tools
remain under `runs/20261014-physical-validation/`. Selected immutable
deliverables are `load-model-checkpoint-v2`, `load-refinement-figures-v1` and
`egg-solver-comparison-v1` under `research/rigid_stress/evidence/20261014-physical/`.
Runtime caches and long job logs continue to use the data mount.

The unselected initial publication at
`research/rigid_stress/evidence/20261014-physical/load-model-checkpoint-5` was
copied with metadata/symlink preservation to
`runs/20261014-physical-validation/publication-attempt-load-model-checkpoint-5`
under the data root. Its writer job712467 was terminalCOMPLETED before copying.
All4321 files were SHA256-verified before removing the untracked source copy.
No tracked deliverable or active writer was moved. Exact old/new absolute paths
and validation are recorded in that run root's `publication-attempt-migration.json`.
