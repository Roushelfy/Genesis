#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/render_final_native.py --summary "$task_run/native-final-comprehensive-complete-v9-v10-summary.json" --output "$task_run/native-final-complete-v9-v10-tables.md" > "$task_run/native-final-complete-v9-v10-tables.log" 2>&1
