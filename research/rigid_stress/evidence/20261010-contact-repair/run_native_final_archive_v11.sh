#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/archive_final_native.py --run "$task_run" --output research/rigid_stress/evidence/20261010-contact-repair --revision 1f22f6a231657c731b13d5b24438d05b52abdbe8 --checkpoint v11 --summary "$task_run/native-final-comprehensive-complete-v8-summary.json" --require-complete > "$task_run/native-final-v11-archive.log" 2>&1
