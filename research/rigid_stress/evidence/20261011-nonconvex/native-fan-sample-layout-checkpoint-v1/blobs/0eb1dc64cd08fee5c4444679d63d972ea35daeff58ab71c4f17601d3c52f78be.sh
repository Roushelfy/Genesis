#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_fan_sample_layout_v1.sh" off profile
bash "$task_root/run_fan_sample_layout_v1.sh" on profile
