#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
case "$SLURM_ARRAY_TASK_ID" in
  0) task_command=(bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/map_resolved_grasp.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/resolved-load-models-y-v5 --replay /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/grasp-u-replay-y-l1-v2 --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.0025.npz --radius 0.003 --ports 2) ;;
  1) task_command=(bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/map_resolved_grasp.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/resolved-load-models-zero-v5 --replay /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/grasp-u-replay-zero-l1-v2 --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.0025.npz --radius 0.003 --ports 2) ;;
  *) exit 2 ;;
esac
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 "${task_command[@]}"
