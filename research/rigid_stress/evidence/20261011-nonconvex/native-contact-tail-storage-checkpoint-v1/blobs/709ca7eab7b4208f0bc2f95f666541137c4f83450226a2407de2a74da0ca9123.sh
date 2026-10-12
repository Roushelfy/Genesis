#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
export PYTHONUNBUFFERED=1
for task_family in compact-cached-positions parallel-refined-fan cooperative-refined-fan sample-contiguous-current; do
    "$RIGID_STRESS_ENV/bin/python" "$task_root/check_native_pairs_v1.py" "$task_family" --output "$task_root/native-${task_family}-qualified-pairs-v1.json" > "$task_root/native-${task_family}-check-v1.log" 2>&1
done
