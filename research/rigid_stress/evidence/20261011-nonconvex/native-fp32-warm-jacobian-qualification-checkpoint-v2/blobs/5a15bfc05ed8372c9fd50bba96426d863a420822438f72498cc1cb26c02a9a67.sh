#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_family=${1:?Family required}
"$RIGID_STRESS_ENV/bin/python" "$task_root/check_native_final_memory_pairs_v4.py" "$task_family" --output "$task_root/native-${task_family}-qualified-pairs-v4.json" > "$task_root/native-${task_family}-check-pairs-v4.log" 2>&1
