#!/usr/bin/env bash
set -euo pipefail
task_batch=${1:?batch required}
task_seed=${2:?seed required}
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
export RIGID_STRESS_DATA_ROOT PYTHONUNBUFFERED=1
export RIGID_STRESS_SOURCE_REVISION=1f22f6a231657c731b13d5b24438d05b52abdbe8-pipeline-entry-v1
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_pipeline_entry_v1/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/native_pipeline_entry_genesis_v1/$SLURM_JOB_ID"
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/probe_native_pipeline_entry.py --envs "$task_batch" --seed "$task_seed" --output "$task_run/native-pipeline-entry-b$task_batch-seed$task_seed-v1.json" > "$task_run/native-pipeline-entry-b$task_batch-seed$task_seed-v1.log" 2>&1
