#!/usr/bin/env bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID:?}" in
0) task_name=middle; task_input=franka-captured-motion-fem-v1 ;;
1) task_name=coarse; task_input=franka-captured-motion-dt-coarse-v1 ;;
2) task_name=fine; task_input=franka-captured-motion-dt-fine-v1 ;;
esac
bash '/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/audit_wide_motion_state.sh' '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation'/franka-wide-state-"$task_name"-v1 --run '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation'/"$task_input"

