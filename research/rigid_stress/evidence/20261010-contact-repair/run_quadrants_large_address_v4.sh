#!/usr/bin/env bash
set -euo pipefail
task_variant=${1:?explicit or default-i64 required}
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/large_address_v4/$SLURM_JOB_ID"
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/large_address_genesis_v4/$SLURM_JOB_ID"
task_extra=()
if [[ "$task_variant" == default-i64 ]]; then task_extra+=(--default-i64); fi
"$RIGID_STRESS_ENV/bin/python" "$task_run/probe_quadrants_large_address_v4.py" --mode vector "${task_extra[@]}" --output "$task_run/quadrants-large-address-vector-$task_variant-v4.json" > "$task_run/quadrants-large-address-vector-$task_variant-v4.log" 2>&1
