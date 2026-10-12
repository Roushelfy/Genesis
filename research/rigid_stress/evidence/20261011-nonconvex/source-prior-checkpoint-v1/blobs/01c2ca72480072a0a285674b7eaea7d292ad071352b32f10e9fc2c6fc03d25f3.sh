#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-refinement
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/source-prior-cloud-comparison-v1"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=("$task_root"/source-prior-fields-v3/{cpu,cuda}/results/*-cloud-0.npz "$task_root/compare_source_prior_clouds_v1.py" "$task_root/run_compare_source_prior_clouds_v1.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
"$RIGID_STRESS_ENV/bin/python" "$task_root/compare_source_prior_clouds_v1.py" > "$task_run/comparison.log" 2>&1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
