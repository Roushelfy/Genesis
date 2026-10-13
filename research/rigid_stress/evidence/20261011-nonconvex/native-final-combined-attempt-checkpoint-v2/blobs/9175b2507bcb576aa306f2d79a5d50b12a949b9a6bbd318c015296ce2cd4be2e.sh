#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
export PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$task_root/archive_native_final_combined_attempt_v2.py" > "$task_root/native-final-combined-attempt-publish-v2.log" 2>&1
