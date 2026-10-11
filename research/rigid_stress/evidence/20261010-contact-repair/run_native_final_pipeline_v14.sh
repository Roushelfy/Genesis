#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
exec > "$task_run/native-final-pipeline-v14.log" 2>&1
task_summary="$task_run/native-final-comprehensive-complete-v14-v9-v10-summary.json"
task_tables="$task_run/native-final-complete-v14-v9-v10-tables.md"
test ! -e "$task_summary"
test ! -e "$task_tables"
echo 'Summarizing complete v9/v10 matrix with pinned preparation identities'
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/summarize_final_native.py --run "$task_run" --version v9 --recovery-version v10 --output "$task_summary" --require-complete > "$task_run/native-final-comprehensive-complete-v14-v9-v10-summary.log" 2>&1
echo 'Rendering complete v9/v10 tables'
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/render_final_native.py --summary "$task_summary" --output "$task_tables" > "$task_run/native-final-complete-v14-v9-v10-tables.log" 2>&1
echo 'Publishing closed final v14 evidence'
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/archive_final_native.py --run "$task_run" --output research/rigid_stress/evidence/20261010-contact-repair --revision 881ab76a --checkpoint v14 --summary "$task_summary" --require-complete > "$task_run/native-final-v14-archive.log" 2>&1
echo 'Final summary, tables and original/publication hash verification completed'
