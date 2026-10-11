#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
for task_backend in cpu cuda; do
    RIGID_STRESS_RUN_DIR="$task_root/intrinsic-regression-v3-$task_backend" bash research/rigid_stress/scripts/check_nonconvex_contact.sh "$task_backend"
done
