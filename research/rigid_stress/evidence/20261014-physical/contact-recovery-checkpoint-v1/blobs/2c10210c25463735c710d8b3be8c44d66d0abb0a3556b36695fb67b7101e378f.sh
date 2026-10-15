#!/usr/bin/env bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID:?}" in
0) task_mesh="/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.0025.npz"; task_pad=5; task_level=1 ;;
1) task_mesh="/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/matrix-mesh-u-v1/meshes/h0.00175.npz"; task_pad=7; task_level=2 ;;
2) task_mesh="/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/fine-matrix-mesh-u-v1/meshes/h0.00125.npz"; task_pad=9; task_level=3 ;;
*) exit 2 ;;
esac
exec bash /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_rigid_initialized_nonlinear.sh /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/u-nonlinear-rigid-init-spatial-${task_level}-v1 --mesh "$task_mesh" --geometry u --pad-resolution "$task_pad" --gravity 0 -9.81 0 --body-rigid-coordinates --updated-friction --reuse-empty-factor --centre-gauge --pad-material neo_hookean --body-material neo_hookean
