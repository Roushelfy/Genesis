#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_batch=${1:?Batch required}
task_seed=${2:?Seed required}
task_revision=${3:?Numerical revision required}
"$RIGID_STRESS_ENV/bin/python" "$task_root/archive_native_final_matched_scopes_v1.py" "$task_batch" "$task_seed" --revision "$task_revision" > "$task_root/native-final-matched-scopes-b${task_batch}-${task_seed}-publish-v1.log" 2>&1
