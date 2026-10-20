#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/nearby-scr-loads-projection-v5/workload-${SLURM_ARRAY_TASK_ID}.sh
