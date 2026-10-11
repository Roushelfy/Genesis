#!/usr/bin/env bash
set -euo pipefail
task_label=${1:?summary label required}
shift
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/summarize_final_native.py --run "$task_run" --version v9 --recovery-version v10 --output "$task_run/native-final-comprehensive-$task_label-v9-v10-summary.json" "$@" > "$task_run/native-final-comprehensive-$task_label-v9-v10-summary.log" 2>&1
