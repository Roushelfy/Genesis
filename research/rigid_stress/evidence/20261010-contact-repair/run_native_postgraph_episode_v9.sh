#!/usr/bin/env bash
set -euo pipefail
task_group=${1:?small or large required}
task_seed=${2:?seed required}
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery-postgraph-dev
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=460e87f5-native-postgraph-production-v9
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_postgraph_episode_v9/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_postgraph_genesis_episode_v9/$SLURM_JOB_ID"
case "$task_group" in
    small) task_batches=(1024 2048) ;;
    large) task_batches=(32768 61440) ;;
    *) exit 2 ;;
esac
task_inputs=()
for task_batch in "${task_batches[@]}"; do
    for task_scope in live policy; do
        task_name="native-postgraph-episode-$task_scope-b$task_batch-seed$task_seed-v9"
        "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/native_grasp_outcomes.py --scope "$task_scope" --envs "$task_batch" --seed "$task_seed" --warmup 900 --steps 2400 --output "$task_run/$task_name.json" > "$task_run/$task_name.log" 2>&1
        task_inputs+=("$task_run/$task_name.json")
    done
done
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/summarize_native_audits.py --output "$task_run/native-postgraph-episode-$task_group-seed$task_seed-v9-summary.json" "${task_inputs[@]}" > "$task_run/native-postgraph-episode-$task_group-seed$task_seed-v9-summary.log" 2>&1
