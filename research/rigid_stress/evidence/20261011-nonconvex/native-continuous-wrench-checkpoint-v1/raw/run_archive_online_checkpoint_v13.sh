#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/archive-native-continuous-wrench-checkpoint-v1"
mkdir -p "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
hostname > "$task_run/node.txt"
"$RIGID_STRESS_ENV/bin/python" "$task_root/archive_online_checkpoint_v13.py" > "$task_run/archive.log" 2>&1
