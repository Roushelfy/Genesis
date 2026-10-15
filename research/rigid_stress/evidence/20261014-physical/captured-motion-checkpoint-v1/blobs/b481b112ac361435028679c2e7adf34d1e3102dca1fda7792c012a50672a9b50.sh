#!/usr/bin/env bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID:?}" in
0) bash '/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_articulated_franka.sh' '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-captured-motion-fem-v1' --capture '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-physical-mass-input-v2' --closure-run '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-regularization-0-pilot-v1' --steps 80 --captured-initial-velocity --check-jacobian ;;
1) bash '/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/research/rigid_stress/scripts/run_rigid_egg_franka.sh' '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-captured-motion-rigid-v1' --capture '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-articulated-physical-mass-input-v2' --held-closure-run '/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation/franka-rigid-egg-coupled-pilot-v3' --steps 80 --captured-initial-velocity --check-jacobian ;;
esac

