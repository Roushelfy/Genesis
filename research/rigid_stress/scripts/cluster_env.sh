#!/usr/bin/env bash
set -euo pipefail
RIGID_STRESS_DATA_ROOT=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
RIGID_STRESS_ENV=${RIGID_STRESS_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-recovery}
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GS_CACHE_FILE_PATH=$RIGID_STRESS_DATA_ROOT/cache/genesis
export QD_OFFLINE_CACHE_FILE_PATH=$RIGID_STRESS_DATA_ROOT/cache/quadrants
export NUMBA_CACHE_DIR=$RIGID_STRESS_DATA_ROOT/cache/numba
export CUPY_CACHE_DIR=$RIGID_STRESS_DATA_ROOT/cache/cupy
export PIP_CACHE_DIR=/mnt/data/zhaofeng/pip_cache
mkdir -p "$RIGID_STRESS_DATA_ROOT/cache"
