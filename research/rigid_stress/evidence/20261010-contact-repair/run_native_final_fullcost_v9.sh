#!/usr/bin/env bash
set -euo pipefail
task_batch=${1:?batch required}
task_seed=${2:?seed required}
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery-postgraph-dev
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=f3e92c6267357143b859b24a27a3a67731586667
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_final_fullcost_v9/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_final_fullcost_genesis_v9/$SLURM_JOB_ID"
for task_scope in live policy; do
    task_modes=(max full)
    if [[ "$task_scope" == policy ]]; then task_modes=(full max); fi
    for task_mode in "${task_modes[@]}"; do
        "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/measured_sweep.py --envs "$task_batch" --seeds "$task_seed" --scopes "$task_scope" --steps 2400 --warmup 900 --repetitions 3 --source-revision "$RIGID_STRESS_SOURCE_REVISION" --output-mode "$task_mode" --output "$task_run/final-native-fullcost-v9-$task_scope-$task_mode-b$task_batch-seed$task_seed" > "$task_run/final-native-fullcost-v9-$task_scope-$task_mode-b$task_batch-seed$task_seed.log" 2>&1
    done
done
