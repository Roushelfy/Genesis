#!/usr/bin/env bash
set -euo pipefail
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
bash "$task_root/run_cooperative_refined_fan_v1.sh" off live 2048 623102
bash "$task_root/run_cooperative_refined_fan_v1.sh" on live 2048 623102
