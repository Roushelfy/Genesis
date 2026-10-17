#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/baseline-closed-scaling-pdf-v2
test ! -e "$task_run"
mkdir -p "$task_run/tmp"
cd /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/analysis-source/baseline-closed-scaling-pdf-v2
sha256sum plot.py style.py summary.json > "$task_run/source-before.sha256"
trap 'task_status=$?; sha256sum plot.py style.py summary.json > "$task_run/source-after.sha256"; if test "$task_status" = 0; then cmp "$task_run/source-before.sha256" "$task_run/source-after.sha256" || task_status=1; fi; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; exit "$task_status"' EXIT
export TMPDIR="$task_run/tmp" PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$task_run/cache/matplotlib" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python plot.py --summary summary.json --style-helper style.py --output "$task_run/results" --export-pdf > "$task_run/render.log" 2>&1
