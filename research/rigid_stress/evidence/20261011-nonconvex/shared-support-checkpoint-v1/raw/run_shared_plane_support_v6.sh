#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/shared-plane-support-v6"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py tests/rigid/{test_stress_surface_fan,test_stress_traction}.py research/rigid_stress/{wrench,mechanics,field_cpu}.py "$task_root/diagnose_shared_plane_support_v6.py" "$task_root/diagnose_native_surface_fans_v11.py" "$task_root/traction_union_candidate_v1.py" "$task_root/cpu_union_oracle_candidate_v1.py")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
mkdir -p "$task_run/cpu" "$task_run/cuda"
(export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cpu/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cpu/cache/genesis"; "$RIGID_STRESS_ENV/bin/python" "$task_root/diagnose_shared_plane_support_v6.py" --backend cpu --output "$task_run/cpu/results" > "$task_run/cpu/probe.log" 2>&1) &
task_cpu_pid=$!
(export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cuda/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cuda/cache/genesis"; "$RIGID_STRESS_ENV/bin/python" "$task_root/diagnose_shared_plane_support_v6.py" --backend cuda --output "$task_run/cuda/results" > "$task_run/cuda/probe.log" 2>&1) &
task_cuda_pid=$!
task_failed=0
wait "$task_cpu_pid" || task_failed=1
wait "$task_cuda_pid" || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
if [[ "$task_failed" == 0 ]]; then
    "$RIGID_STRESS_ENV/bin/python" "$task_root/compare_shared_support_backends_v3.py" > "$task_run/compare.log" 2>&1 || task_failed=1
fi
exit "$task_failed"
