#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
for task_mode in off both whole whole-fast; do
    bash "$task_root/run_native_contact_scheduling_live_v1.sh" "$task_mode" live
done
