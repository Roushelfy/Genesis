#!/usr/bin/env bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID:?}" in
0) task_run='/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-captured-motion-dt-coarse-v1'; task_dt=0.0025; task_steps=40 ;;
1) task_run='/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-captured-motion-dt-fine-v1'; task_dt=0.000625; task_steps=160 ;;
esac
bash '/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_articulated_franka.sh' "$task_run" --capture '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-physical-mass-input-v2' --closure-run '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-regularization-0-pilot-v1' --dt "$task_dt" --steps "$task_steps" --captured-initial-velocity --check-jacobian

