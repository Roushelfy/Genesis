#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-refinement
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/boundary-penalty-continuous-v1"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=("$task_root/probe_boundary_penalty_continuous_v1.py" "$task_root/run_boundary_penalty_continuous_v1.sh" "$task_root/audit-policy-failure-inputs-v1/results/original-environment-24.npz" "$task_root/boundary-penalty-certificate-v1/cpu/results/native-ray.npz" examples/rigid/assets/hollow_egg/level1/elastic.npz)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_boundary_penalty_continuous_v1.py" "$task_run/results" > "$task_run/diagnostic.log" 2>&1 || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
