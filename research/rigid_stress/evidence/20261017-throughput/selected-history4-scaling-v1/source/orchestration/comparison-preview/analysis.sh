#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/final-throughput-comparison-preview-v3
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
 if test "$task_status" = 0; then
  if test "$task_source_status" != 0; then task_status=$task_source_status; fi
 fi
 printf '%s\n' "$task_status" > returncode.txt
 exit "$task_status"
}
trap task_finish EXIT
set +e
/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python source/plot.py --history4-summary input/history4.json --history0-summary input/history0.json --style-helper source/style.py --output results > render.log 2>&1
task_plot_status=$?
printf '%s\n' "$task_plot_status" > plot-program-returncode.txt
exit "$task_plot_status"
