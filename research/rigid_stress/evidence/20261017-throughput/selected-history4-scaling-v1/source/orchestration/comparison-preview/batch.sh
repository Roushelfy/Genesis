#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/final-throughput-comparison-preview-v3
task_upstream=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/final-throughput-figure-preview-v2
test ! -e "$task_run"
for task_exit in program-returncode.txt returncode.txt batch-returncode.txt; do
 test "$(cat "$task_upstream/$task_exit")" = 0
done
mkdir -p "$task_run/source" "$task_run/input" "$task_run/tmp"
cp /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/analysis-source/final-throughput-comparison-preview-v3/* "$task_run/source/"
cp "$task_upstream/results/history4.json" "$task_upstream/results/history0.json" "$task_run/input/"
cd "$task_run"
sha256sum source/* input/* > source-before.sha256
sacct -j 725302 --format=JobID,State,ExitCode,Elapsed,NodeList -P > upstream-scheduler.txt
set +u
source /mnt/data/shared/config/env.sh
set +e
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash "$task_run/source/analysis.sh"
task_status=$?
printf '%s\n' "$task_status" > batch-returncode.txt
exit "$task_status"
