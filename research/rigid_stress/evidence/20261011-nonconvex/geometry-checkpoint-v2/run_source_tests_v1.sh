#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-source-v1/$SLURM_JOB_ID/quadrants"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-source-v1/$SLURM_JOB_ID/genesis"
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis; print(genesis.__file__)'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
sha256sum genesis/engine/solvers/rigid/stress/{data,model,surface,topology,paths,provenance}.py tests/rigid/test_stress_nonconvex.py > "$task_run/source-source-v1.sha256"
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cpu --tb=short -q tests/rigid/test_stress_nonconvex.py -k source_provenance > "$task_run/source-cpu-v1.log" 2>&1
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cuda --tb=short -q tests/rigid/test_stress_nonconvex.py -k source_provenance > "$task_run/source-cuda-v1.log" 2>&1
echo 'SOURCE_V1_COMPLETE'
