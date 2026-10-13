#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
"$RIGID_STRESS_ENV/bin/python" "$task_root/clean_native_sampler_warp_v1.py" > "$task_root/native-sampler-warp-clean-v1.log" 2>&1
