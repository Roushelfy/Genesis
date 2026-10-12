#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
"$RIGID_STRESS_ENV/bin/python" "$task_root/check_native_wrench_online_v1.py" > "$task_root/check-native-wrench-online-v1.log" 2>&1
