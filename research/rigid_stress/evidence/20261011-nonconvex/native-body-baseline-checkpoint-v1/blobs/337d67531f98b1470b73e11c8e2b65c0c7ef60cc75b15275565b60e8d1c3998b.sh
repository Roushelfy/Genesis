#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
for task_scope in live policy; do
    bash "$task_root/run_native_body_baseline_v1.sh" "$1" varied "$task_scope" frontier "$2"
done
