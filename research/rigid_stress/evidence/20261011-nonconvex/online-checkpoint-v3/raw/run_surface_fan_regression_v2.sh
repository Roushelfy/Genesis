#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/surface-fan-regression-v2"
mkdir -p "$task_root/tmp"
export TMPDIR="$task_root/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/extract_surface_fan_fixture_v1.py" > "$task_root/fixture.log" 2>&1
RIGID_STRESS_RUN_DIR="$task_root/cpu" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu > "$task_root/cpu-launch.log" 2>&1 &
task_cpu_pid=$!
RIGID_STRESS_RUN_DIR="$task_root/cuda" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cuda > "$task_root/cuda-launch.log" 2>&1 &
task_cuda_pid=$!
task_failed=0
wait "$task_cpu_pid" || task_failed=1
wait "$task_cuda_pid" || task_failed=1
exit "$task_failed"
