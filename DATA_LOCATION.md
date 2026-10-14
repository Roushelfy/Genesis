# Runtime data locations

The rigid stress recovery worktree uses
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery` for runs,
generated collision assets, video, numerical replays and caches. Its optional
environment is `/mnt/data/zhaofeng/venvs/rigid-stress-recovery`.

`research/rigid_stress/scripts/cluster_env.sh` sets these locations explicitly.
Override `RIGID_STRESS_DATA_ROOT` and `RIGID_STRESS_ENV` for another machine.
Dependencies reuse `/mnt/data/zhaofeng/pip_cache`.

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
