#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
RIGID_STRESS_RUN_DIR="$task_run/checkpoint-v1-cpu" bash research/rigid_stress/scripts/check_nonconvex_geometry.sh cpu
RIGID_STRESS_RUN_DIR="$task_run/checkpoint-v1-cuda" bash research/rigid_stress/scripts/check_nonconvex_geometry.sh cuda
echo 'CHECKPOINT_COMBINED_V1_COMPLETE'
