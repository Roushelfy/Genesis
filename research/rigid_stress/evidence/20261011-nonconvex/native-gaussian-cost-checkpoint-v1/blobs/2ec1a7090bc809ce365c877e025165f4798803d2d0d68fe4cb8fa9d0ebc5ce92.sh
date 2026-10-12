#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/native-gaussian-precision-cost-v1"
test ! -e "$task_run"
mkdir -p "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
cp "$task_root/probe_gaussian_precision_cost_v1.py" "$task_root/run_gaussian_precision_cost_v1.sh" "$task_run/"
sha256sum "$task_root/probe_gaussian_precision_cost_v1.py" "$task_root/run_gaussian_precision_cost_v1.sh" > "$task_run/source.sha256"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_gaussian_precision_cost_v1.py" --capture "$task_root/native-selected-dynamic-v1/b16384-623102/result.instrumented.npz" --output "$task_run/result.json" > "$task_run/benchmark.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "$task_root/probe_gaussian_precision_cost_v1.py" "$task_root/run_gaussian_precision_cost_v1.sh" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
