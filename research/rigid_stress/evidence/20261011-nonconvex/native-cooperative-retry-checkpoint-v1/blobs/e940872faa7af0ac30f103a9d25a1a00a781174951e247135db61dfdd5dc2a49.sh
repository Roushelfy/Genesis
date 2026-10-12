#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
for task_mode in off on; do
    bash "$task_root/run_cooperative_retry_v1.sh" "$task_mode" live
done
