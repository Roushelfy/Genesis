#!/usr/bin/env bash
set -euo pipefail
task_batch=${1:?batch required}
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery-postgraph-dev
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=f3e92c6267357143b859b24a27a3a67731586667
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_final_profile_v9/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_final_profile_genesis_v9/$SLURM_JOB_ID"
"$RIGID_STRESS_ENV/bin/python" examples/speed_benchmark/rigid_stress.py --scope profile --envs "$task_batch" --seed 623001 --varied --warmup 900 --steps 200 --compact-log --output "$task_run/native-final-snapshot-profile-b$task_batch-v9.json" > "$task_run/native-final-snapshot-profile-b$task_batch-v9.log" 2>&1
for task_scope in live policy; do
    for task_mode in max full; do
        "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/profile_native_rollout.py --scope "$task_scope" --envs "$task_batch" --seed 623001 --warmup 900 --steps 1200 --output-mode "$task_mode" --output "$task_run/native-final-rollout-profile-$task_scope-$task_mode-b$task_batch-v9.json" > "$task_run/native-final-rollout-profile-$task_scope-$task_mode-b$task_batch-v9.log" 2>&1
    done
done
