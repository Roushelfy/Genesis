#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
case "$SLURM_ARRAY_TASK_ID" in
  0) task_command=(bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_grasp_contact_reference.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/grasp-u-spatial-y-l1-v1 --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.0025.npz --geometry u --pad-resolution 5 --gravity 0 -9.81 0 --body-rigid-coordinates --updated-friction) ;;
  1) task_command=(bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_grasp_contact_reference.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/grasp-u-spatial-y-l2-v1 --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.00175.npz --geometry u --pad-resolution 7 --gravity 0 -9.81 0 --body-rigid-coordinates --updated-friction) ;;
  2) task_command=(bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_grasp_contact_reference.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/grasp-u-spatial-zero-l1-v1 --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.0025.npz --geometry u --pad-resolution 5 --gravity 0 0 0 --body-rigid-coordinates --updated-friction) ;;
  *) exit 2 ;;
esac
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 "${task_command[@]}"
