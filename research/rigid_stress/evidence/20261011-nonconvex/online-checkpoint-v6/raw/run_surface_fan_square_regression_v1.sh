#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/surface-fan-square-regression-v1"
mkdir -p "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$task_root/extract_patch_fixtures_v1.py" > "$task_run/fixture.log" 2>&1
RIGID_STRESS_RUN_DIR="$task_run/cpu" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu > "$task_run/cpu-launch.log" 2>&1 &
task_cpu_pid=$!
RIGID_STRESS_RUN_DIR="$task_run/cuda" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cuda > "$task_run/cuda-launch.log" 2>&1 &
task_cuda_pid=$!
task_failed=0
wait "$task_cpu_pid" || task_failed=1
wait "$task_cuda_pid" || task_failed=1
exit "$task_failed"
