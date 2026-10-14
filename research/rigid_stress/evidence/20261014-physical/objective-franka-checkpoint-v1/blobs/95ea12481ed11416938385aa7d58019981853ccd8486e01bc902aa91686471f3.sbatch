#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
case "$SLURM_ARRAY_TASK_ID" in
  0) task_name=franka-input-s2-v1; task_substeps=2 ;;
  1) task_name=franka-input-s4-v1; task_substeps=4 ;;
  *) exit 2 ;;
esac
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/capture_franka_reference.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/"$task_name" --substeps "$task_substeps"

