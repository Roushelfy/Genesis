#!/usr/bin/env bash
set -e
source /mnt/data/shared/config/env.sh
case "$SLURM_ARRAY_TASK_ID" in
  0) task_dhat=0.0000025 ;;
  1) task_dhat=0.00000125 ;;
  2) task_dhat=0.000000625 ;;
  *) exit 2 ;;
esac
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_articulated_franka.sh \
 /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-regularization-$SLURM_ARRAY_TASK_ID-hold-v1 \
 --capture /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-physical-mass-input-v2 \
 --closure-run /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-regularization-$SLURM_ARRAY_TASK_ID-pilot-v1 \
 --dhat "$task_dhat" --dt 0.00125 --steps 80

