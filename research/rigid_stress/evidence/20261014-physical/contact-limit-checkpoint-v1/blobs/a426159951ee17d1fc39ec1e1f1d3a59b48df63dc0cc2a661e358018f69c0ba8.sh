#!/usr/bin/env bash
set -e
source /mnt/data/shared/config/env.sh
case "$SLURM_ARRAY_TASK_ID" in
  0) task_level=0 ;;
  1) task_level=12 ;;
  2) task_level=15 ;;
  3) task_level=18 ;;
  *) exit 2 ;;
esac
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_localized_franka.sh \
 /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-tip-coupled-$task_level-v1 solve \
 --closure-run /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-ref-centred-v1 \
 --local-contact-run /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-spatial-cold-0-v1 \
 --local-levels "$task_level" --local-radius 0.00005 --body-refinement 0 --cold-closure \
 --control-steps 64 --dt 0.00125 --steps 80

