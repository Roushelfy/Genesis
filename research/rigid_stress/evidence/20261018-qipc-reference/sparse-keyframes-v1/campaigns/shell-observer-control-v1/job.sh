#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash /mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/shell-observer-control-v1/workload-${SLURM_ARRAY_TASK_ID}.sh
