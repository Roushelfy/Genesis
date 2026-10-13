#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
"$RIGID_STRESS_ENV/bin/python" "$task_root/archive_native_fit_interior_cost_v2.py" > "$task_root/native-fit-interior-cost-publish-v2.log" 2>&1
