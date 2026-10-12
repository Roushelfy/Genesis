#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_compact_cache_fan_selected_live_v1.sh" baseline
bash "$task_root/run_compact_cache_fan_selected_live_v1.sh" candidate
