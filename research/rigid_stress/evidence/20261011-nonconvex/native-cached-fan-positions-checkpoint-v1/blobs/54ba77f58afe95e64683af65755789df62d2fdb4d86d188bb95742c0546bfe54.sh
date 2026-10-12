#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_cached_fan_positions_v1.sh" off live 2048 623102
bash "$task_root/run_cached_fan_positions_v1.sh" on live 2048 623102
