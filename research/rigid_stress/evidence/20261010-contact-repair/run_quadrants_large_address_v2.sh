#!/usr/bin/env bash
set -euo pipefail
task_mode=${1:?scalar or vector required}
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/large_address_v2/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/large_address_genesis_v2/$SLURM_JOB_ID"
"$RIGID_STRESS_ENV/bin/python" "$task_run/probe_quadrants_large_address_v2.py" --mode "$task_mode" --output "$task_run/quadrants-large-address-$task_mode-v2.json" > "$task_run/quadrants-large-address-$task_mode-v2.log" 2>&1
