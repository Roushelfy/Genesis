#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/summarize_postgraph.py --run "$task_run" --output "$task_run/postgraph-composition-final-v9-summary.json" > "$task_run/postgraph-composition-final-v9-summary.log" 2>&1
