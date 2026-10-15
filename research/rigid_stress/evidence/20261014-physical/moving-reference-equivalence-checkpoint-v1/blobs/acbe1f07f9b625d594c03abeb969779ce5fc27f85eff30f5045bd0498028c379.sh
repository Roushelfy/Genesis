#!/usr/bin/env bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID:?}" in
0) task_name=coarse; task_dt=0.0025; task_steps=40 ;;
1) task_name=middle; task_dt=0.00125; task_steps=80 ;;
2) task_name=fine; task_dt=0.000625; task_steps=160 ;;
esac
bash '/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_predicted_wide_franka.sh' '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation'/franka-predicted-motion-"$task_name"-v1 --capture '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-physical-mass-input-v2' --closure-run '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-regularization-0-pilot-v1' --dt "$task_dt" --steps "$task_steps" --captured-initial-velocity --check-jacobian

