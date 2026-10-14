#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_revision=${1:?Numerical revision required}
"$RIGID_STRESS_ENV/bin/python" "$task_root/archive_native_final_copied_curve_v1.py" --revision "$task_revision" > "$task_root/native-final-copied-curve-publish-v1.log" 2>&1
