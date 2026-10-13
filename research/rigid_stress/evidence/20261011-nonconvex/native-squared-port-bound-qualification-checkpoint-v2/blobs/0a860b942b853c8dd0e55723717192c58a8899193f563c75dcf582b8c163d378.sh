#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_family=${1:?Family required}
export PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$task_root/archive_native_final_candidate_v2.py" "$task_family" > "$task_root/native-${task_family}-publish-v2.log" 2>&1
