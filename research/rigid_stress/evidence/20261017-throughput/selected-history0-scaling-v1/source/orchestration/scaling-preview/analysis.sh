#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/final-throughput-figure-preview-v2
cd "$task_run"
export TMPDIR="$task_run/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$task_run/cache/matplotlib" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_finish() {
 task_status=$?
 trap - EXIT
 set +e
 printf '%s\n' "$task_status" > program-returncode.txt
 sha256sum source/* > source-after.sha256
 cmp source-before.sha256 source-after.sha256
 task_source_status=$?
 if test "$task_status" = 0; then
  if test "$task_source_status" != 0; then task_status=$task_source_status; fi
 fi
 printf '%s\n' "$task_status" > returncode.txt
 exit "$task_status"
}
trap task_finish EXIT
for task_history in 4 0; do
 /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python source/summarize.py --root /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput --sacct scheduler.txt --history "$task_history" --output "results/history$task_history.json" > "summary$task_history.log" 2>&1
 /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python source/audit.py --summary "results/history$task_history.json" --contract /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/meta/baseline-contract-v1.json --history "$task_history" --output "results/audit$task_history.json" > "audit$task_history.log" 2>&1
 if test "$task_history" = 4; then
  task_family=packed-history-surface
 else
  task_family=history0
 fi
 /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python source/plot.py --summary "results/history$task_history.json" --style-helper source/style.py --rate-family "$task_family-four-scopes-b" --memory-family "$task_family-memory-four-scopes-b" --series-label "selected engine, history$task_history" --output "results/history$task_history-figure" > "render$task_history.log" 2>&1
done
