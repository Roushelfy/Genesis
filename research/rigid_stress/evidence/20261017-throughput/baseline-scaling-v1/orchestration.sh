#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/baseline-scaling-figure-v2
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
cp /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/baseline-scaling-figure-preview-v2/source/plot_practical_throughput.py "$task_run/source/plot_practical_throughput.py"
cp /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/baseline-scaling-figure-preview-v2/source/nature_mpl_style.py "$task_run/source/nature_mpl_style.py"
cp /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/baseline-scaling-figure-preview-v2/source/summary.json "$task_run/source/summary.json"
cp /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/meta/baseline-scaling-figure-v2.sh "$task_run/orchestration.sh"
cd "$task_run"
sha256sum source/* orchestration.sh > source-before.sha256
trap 'task_status=$?; sha256sum source/* orchestration.sh > source-after.sha256; if test "$task_status" = 0; then cmp source-before.sha256 source-after.sha256 || task_status=1; fi; printf "%s\n" "$task_status" > returncode.txt; exit "$task_status"' EXIT
export TMPDIR="$task_run/tmp" MPLCONFIGDIR="$task_run/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python source/plot_practical_throughput.py --summary source/summary.json --style-helper source/nature_mpl_style.py --output results --export-pdf > render.log 2>&1
