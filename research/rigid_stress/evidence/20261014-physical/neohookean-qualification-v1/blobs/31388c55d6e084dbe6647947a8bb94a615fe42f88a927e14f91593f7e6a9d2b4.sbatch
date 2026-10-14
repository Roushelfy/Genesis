#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
case "$SLURM_ARRAY_TASK_ID" in
  0) task_name=grasp-u-neohookean-pads-y-l1-v1; task_body=linear ;;
  1) task_name=grasp-u-neohookean-all-y-l1-v1; task_body=neo_hookean ;;
  *) exit 2 ;;
esac
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_neohookean_grasp.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/"$task_name" --mesh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.0025.npz --geometry u --pad-resolution 5 --gravity 0 -9.81 0 --body-rigid-coordinates --updated-friction --reuse-empty-factor --centre-gauge --pad-material neo_hookean --body-material "$task_body"

