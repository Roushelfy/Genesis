#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
for task_scope in rigid live policy; do
    bash "$task_root/run_rollout_metrics_validation_v3.sh" "$task_scope"
done
