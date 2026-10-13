#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
"$RIGID_STRESS_ENV/bin/python" "$task_root/audit_native_first_warm_accept_cost_v1.py" > "$task_root/native-first-warm-accept-cost-v1.log" 2>&1
