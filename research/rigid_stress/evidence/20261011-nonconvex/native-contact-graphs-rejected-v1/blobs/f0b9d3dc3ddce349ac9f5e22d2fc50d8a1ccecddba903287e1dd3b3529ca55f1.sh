#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
for task_mode in off fast graph both; do
    bash "$task_root/run_native_contact_dispatch_v2.sh" "$task_mode" profile
done
