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
