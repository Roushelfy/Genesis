#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/final-throughput-pdf-v1
cd "$task_run"
export TMPDIR="$task_run/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$task_run/cache/matplotlib" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_finish() {
 task_status=$?
 trap - EXIT
 set +e
 printf '%s\n' "$task_status" > program-returncode.txt
 sha256sum source/* input/* > source-after.sha256
 cmp source-before.sha256 source-after.sha256
 task_source_status=$?
 /usr/bin/python3 - <<'INVENTORY' > source-inventory-after.json
from pathlib import Path
import json
expected={line.split('  ',1)[1] for line in Path('source-before.sha256').read_text().splitlines()}
actual={str(f) for folder in ['source','input'] for f in Path(folder).rglob('*') if f.is_file()}
print(json.dumps({'extra_files':sorted(actual-expected),'missing_files':sorted(expected-actual),'passed':actual==expected},indent=2))
assert actual==expected
INVENTORY
 task_inventory_status=$?
 if test "$task_status" = 0; then
  if test "$task_source_status" != 0; then task_status=$task_source_status
  elif test "$task_inventory_status" != 0; then task_status=$task_inventory_status; fi
 fi
 printf '%s\n' "$task_status" > returncode.txt
 exit "$task_status"
}
trap task_finish EXIT
for task_history in 4 0; do
 if test "$task_history" = 4; then task_family=packed-history-surface; else task_family=history0; fi
 set +e
 /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python source/scaling.py --summary "input/history$task_history.json" --style-helper source/style.py --rate-family "$task_family-four-scopes-b" --memory-family "$task_family-memory-four-scopes-b" --series-label "selected engine, history$task_history" --output "results/history$task_history-figure" --export-pdf > "render$task_history.log" 2>&1
 task_plot_status=$?
 printf '%s\n' "$task_plot_status" > "render$task_history-program-returncode.txt"
 set -e
 if test "$task_plot_status" != 0; then exit "$task_plot_status"; fi
done
set +e
/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python source/comparison.py --history4-summary input/history4.json --history0-summary input/history0.json --style-helper source/style.py --output results/comparison-figure --export-pdf > render-comparison.log 2>&1
task_plot_status=$?
printf '%s\n' "$task_plot_status" > render-comparison-program-returncode.txt
exit "$task_plot_status"
