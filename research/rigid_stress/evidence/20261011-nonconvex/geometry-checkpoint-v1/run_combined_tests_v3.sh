#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-combined-v3/$SLURM_JOB_ID/quadrants"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-combined-v3/$SLURM_JOB_ID/genesis"
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis; print(genesis.__file__)'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
sha256sum genesis/engine/solvers/rigid/stress/{data,model,surface,topology,paths}.py tests/rigid/test_stress_nonconvex.py > "$task_run/combined-source-v3.sha256"
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cpu --tb=short -q tests/rigid/test_stress_nonconvex.py > "$task_run/combined-cpu-v3.log" 2>&1
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cuda --tb=short -q tests/rigid/test_stress_nonconvex.py > "$task_run/combined-cuda-v3.log" 2>&1
set +e
PYTHONPATH="$task_run:$PWD" "$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cpu --tb=short -q -p path_euclidean_mutation tests/rigid/test_stress_nonconvex.py -k surface_paths > "$task_run/path-euclidean-mutation-v3.log" 2>&1
task_mutation_status=$?
set -e
if [[ "$task_mutation_status" != 1 ]]; then
    echo "Unexpected mutation status: $task_mutation_status"
    exit 1
fi
echo 'COMBINED_V3_COMPLETE'
