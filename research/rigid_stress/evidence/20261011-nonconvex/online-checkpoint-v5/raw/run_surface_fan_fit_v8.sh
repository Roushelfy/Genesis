#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/surface-fan-fit-v8"
mkdir -p "$task_run/tmp" "$task_run/source" "$task_run/cpu" "$task_run/cuda"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/cpu/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/cpu/genesis"
"$RIGID_STRESS_ENV/bin/python" "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/diagnose_native_surface_fans_v8.py" --backend cpu --run-dir "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/online-surface-fan-gjk-patch-v3/results/failure" --edge-subdivisions 16 --anchor-fan --packed --output "$task_run/cpu/diagnosis.json" > "$task_run/cpu.log" 2>&1 &
task_cpu_pid=$!
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/cuda/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/cuda/genesis"
"$RIGID_STRESS_ENV/bin/python" "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/diagnose_native_surface_fans_v8.py" --backend cuda --run-dir "$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/online-surface-fan-gjk-patch-v3/results/failure" --edge-subdivisions 16 --anchor-fan --packed --output "$task_run/cuda/diagnosis.json" > "$task_run/cuda.log" 2>&1 &
task_cuda_pid=$!
task_failed=0
wait "$task_cpu_pid" || task_failed=1
wait "$task_cuda_pid" || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
