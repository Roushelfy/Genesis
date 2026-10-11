#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/archive_final_native.py --run "$task_run" --output research/rigid_stress/evidence/20261010-contact-repair --revision c5d7f160 --checkpoint v13 --summary "$task_run/native-final-comprehensive-complete-v9-v10-summary.json" --require-complete > "$task_run/native-final-v13-archive.log" 2>&1
