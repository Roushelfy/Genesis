#!/usr/bin/env bash
set -euo pipefail
task_label=${1:?summary label required}
shift
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp"
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/summarize_final_native.py --run "$task_run" --output "$task_run/native-final-comprehensive-$task_label-v8-summary.json" "$@" > "$task_run/native-final-comprehensive-$task_label-v8-summary.log" 2>&1
