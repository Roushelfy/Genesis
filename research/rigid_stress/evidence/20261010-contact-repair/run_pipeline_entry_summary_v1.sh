#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" "$task_run/summarize_pipeline_entry_v1.py" > "$task_run/native-pipeline-entry-complete-v1-summary.log" 2>&1
