#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
RIGID_STRESS_RUN_DIR="$task_root/intrinsic-regression-v4-cpu" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu &
task_cpu_pid=$!
RIGID_STRESS_RUN_DIR="$task_root/intrinsic-regression-v4-cuda" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cuda &
task_cuda_pid=$!
task_failed=0
wait "$task_cpu_pid" || task_failed=1
wait "$task_cuda_pid" || task_failed=1
exit "$task_failed"
