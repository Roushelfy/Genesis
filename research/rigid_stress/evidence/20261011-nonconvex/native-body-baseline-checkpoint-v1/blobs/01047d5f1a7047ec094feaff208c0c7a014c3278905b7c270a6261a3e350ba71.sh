#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
for task_scope in rigid recovery profile live policy; do
    bash "$task_root/run_native_body_baseline_v1.sh" 1024 copied "$task_scope" frontier 623101
done
