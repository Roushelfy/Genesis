#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-paths-v1/$SLURM_JOB_ID/quadrants"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-paths-v1/$SLURM_JOB_ID/genesis"
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis; print(genesis.__file__)'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cpu --tb=short -q tests/rigid/test_stress_nonconvex.py -k surface_paths > "$task_run/paths-cpu-v1.log" 2>&1
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cuda --tb=short -q tests/rigid/test_stress_nonconvex.py -k surface_paths > "$task_run/paths-cuda-v1.log" 2>&1
echo 'PATHS_V1_COMPLETE'
