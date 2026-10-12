#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
for task_mode in off fast off-after; do
    bash "$task_root/run_native_contact_fast_live_v1.sh" "$task_mode" live
done
