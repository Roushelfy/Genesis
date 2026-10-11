#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-boundary-v2/$SLURM_JOB_ID/quadrants"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-boundary-v2/$SLURM_JOB_ID/genesis"
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis; print(genesis.__file__)'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cpu --tb=short -q tests/rigid/test_stress_nonconvex.py > "$task_run/boundary-cpu-v2.log" 2>&1
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cuda --tb=short -q tests/rigid/test_stress_nonconvex.py > "$task_run/boundary-cuda-v2.log" 2>&1
set +e
PYTHONPATH="$task_run:$PWD" "$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cpu --tb=short -q -p boundary_com_mutation tests/rigid/test_stress_nonconvex.py -k material_normals > "$task_run/boundary-com-mutation-v2.log" 2>&1
task_mutation_status=$?
set -e
if [[ "$task_mutation_status" != 1 ]]; then
    echo "Unexpected mutation status: $task_mutation_status"
    exit 1
fi
echo 'BOUNDARY_V2_COMPLETE'
