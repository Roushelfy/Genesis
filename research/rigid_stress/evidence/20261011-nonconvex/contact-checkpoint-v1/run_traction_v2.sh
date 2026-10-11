#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/traction-v2"
mkdir -p "$run/tmp"
export TMPDIR="$run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$run/cache/genesis"
export QD_OFFLINE_CACHE_FILE_PATH="$run/cache/quadrants"
sources=(genesis/engine/solvers/rigid/stress/{traction,provenance,paths}.py tests/rigid/{test_stress_traction,test_stress_traction_limits,test_stress_nonconvex}.py)
sha256sum "${sources[@]}" > "$run/source.sha256"
mkdir -p "$run/source"
cp --parents "${sources[@]}" "$run/source"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run/hardware.csv"
hostname > "$run/node.txt"
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis; print(genesis.__file__)' > "$run/import.log" 2>&1
failed=0
"$RIGID_STRESS_ENV/bin/python" -m pytest tests/rigid/test_stress_traction.py tests/rigid/test_stress_traction_limits.py -n 0 --backend cpu --tb=short -q > "$run/cpu.log" 2>&1 || failed=1
"$RIGID_STRESS_ENV/bin/python" -m pytest tests/rigid/test_stress_traction.py tests/rigid/test_stress_traction_limits.py -n 0 --backend cuda --tb=short -q > "$run/cuda.log" 2>&1 || failed=1
sha256sum "${sources[@]}" > "$run/source-after.sha256"
cmp "$run/source.sha256" "$run/source-after.sha256"
exit "$failed"
