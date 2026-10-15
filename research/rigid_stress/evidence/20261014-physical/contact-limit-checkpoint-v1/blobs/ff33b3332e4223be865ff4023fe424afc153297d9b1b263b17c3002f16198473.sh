#!/usr/bin/env bash
set -e
source /mnt/data/shared/config/env.sh
case "$SLURM_ARRAY_TASK_ID" in
  0) task_channel=coupled; task_kind=fields; task_script=compare_spatial_franka ;;
  1) task_channel=frozen; task_kind=fields; task_script=compare_spatial_franka ;;
  2) task_channel=coupled; task_kind=tractions; task_script=audit_refined_contact_tractions ;;
  3) task_channel=frozen; task_kind=tractions; task_script=audit_refined_contact_tractions ;;
  *) exit 2 ;;
esac
task_inputs=()
for task_level in 0 3 6 9; do
  task_inputs+=("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-local-$task_channel-$task_level-v1")
done
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash "/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/$task_script.sh" \
 "/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-local-four-level-$task_channel-$task_kind-v1" --axis local_levels --run "${task_inputs[@]}"

