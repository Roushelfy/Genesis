#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/surface-fan-oracle-v3"
mkdir -p "$task_root/tmp" "$task_root/source"
export TMPDIR="$task_root/tmp" PYTHONUNBUFFERED=1
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py tests/rigid/{test_stress_traction,test_stress_footprint,test_stress_surface_fan}.py tests/rigid/assets/nonconvex_mesh64_tick147.npz)
sha256sum "${task_sources[@]}" > "$task_root/source.sha256"
cp --parents "${task_sources[@]}" "$task_root/source"
hostname > "$task_root/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_root/hardware.csv"
export QD_OFFLINE_CACHE_FILE_PATH="$task_root/cache/cpu/quadrants" GS_CACHE_FILE_PATH="$task_root/cache/cpu/genesis"
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cpu --tb=short -q tests/rigid/test_stress_traction.py tests/rigid/test_stress_surface_fan.py > "$task_root/cpu.log" 2>&1 &
task_cpu_pid=$!
export QD_OFFLINE_CACHE_FILE_PATH="$task_root/cache/cuda/quadrants" GS_CACHE_FILE_PATH="$task_root/cache/cuda/genesis"
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cuda --tb=short -q tests/rigid/test_stress_traction.py tests/rigid/test_stress_surface_fan.py > "$task_root/cuda.log" 2>&1 &
task_cuda_pid=$!
task_failed=0
wait "$task_cpu_pid" || task_failed=1
wait "$task_cuda_pid" || task_failed=1
sha256sum "${task_sources[@]}" > "$task_root/source-after.sha256"
cmp "$task_root/source.sha256" "$task_root/source-after.sha256"
exit "$task_failed"
