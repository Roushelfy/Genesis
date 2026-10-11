#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
failed=0
for backend in cpu cuda; do
    run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/sliding-footprint-v1-$backend"
    mkdir -p "$run/tmp" "$run/source"
    export TMPDIR="$run/tmp" PYTHONUNBUFFERED=1
    export QD_OFFLINE_CACHE_FILE_PATH="$run/cache/quadrants"
    export GS_CACHE_FILE_PATH="$run/cache/genesis"
    sources=(genesis/engine/solvers/rigid/stress/*.py tests/rigid/{test_stress_sliding_footprint,test_stress_footprint,test_stress_traction}.py tests/rigid/assets/nonconvex_sliding_tick*.npz)
    sha256sum "${sources[@]}" > "$run/source.sha256"
    cp --parents "${sources[@]}" "$run/source"
    hostname > "$run/node.txt"
    if [[ "$backend" == cuda ]]; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run/hardware.csv"
    fi
    "$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend "$backend" --tb=short -q tests/rigid/test_stress_sliding_footprint.py > "$run/tests.log" 2>&1 || failed=1
    sha256sum "${sources[@]}" > "$run/source-after.sha256"
    cmp "$run/source.sha256" "$run/source-after.sha256"
done
exit "$failed"
