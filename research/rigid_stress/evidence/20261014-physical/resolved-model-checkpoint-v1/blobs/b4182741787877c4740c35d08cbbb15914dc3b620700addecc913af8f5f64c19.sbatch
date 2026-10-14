#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
case "$SLURM_ARRAY_TASK_ID" in
  0) task_command=(bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-reference-acceleration/research/rigid_stress/scripts/run_grasp_contact_reference.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/grasp-u-cache-only-v1 --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.0025.npz --geometry u --pad-resolution 5 --gravity 0 -9.81 0 --body-rigid-coordinates --updated-friction --reuse-empty-factor) ;;
  1) task_command=(bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-reference-acceleration/research/rigid_stress/scripts/run_grasp_contact_reference.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/grasp-u-centred-cache-v1 --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.0025.npz --geometry u --pad-resolution 5 --gravity 0 -9.81 0 --body-rigid-coordinates --updated-friction --reuse-empty-factor --centre-gauge) ;;
  *) exit 2 ;;
esac
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 "${task_command[@]}"
