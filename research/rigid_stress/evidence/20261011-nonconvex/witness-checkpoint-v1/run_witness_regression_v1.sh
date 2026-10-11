#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
run="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex/witness-regression-v1"
mkdir -p "$run/tmp"
export TMPDIR="$run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-witness-regression-v1/$SLURM_JOB_ID/genesis"
export QD_OFFLINE_CACHE_FILE_PATH="$RIGID_STRESS_DATA_ROOT/cache/nonconvex-witness-regression-v1/$SLURM_JOB_ID/quadrants"
sources=(genesis/utils/array_class.py genesis/engine/solvers/rigid/collider/{collider,contact,gjk,mpr,narrowphase,witness}.py tests/rigid/test_stress_witness.py)
sha256sum "${sources[@]}" > "$run/source.sha256"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run/hardware.csv"
failed=0
"$RIGID_STRESS_ENV/bin/python" -m pytest tests/rigid/test_stress.py tests/rigid/test_stress_nonconvex.py -n 0 --backend cuda --tb=short -q > "$run/stress-cuda.log" 2>&1 || failed=1
"$RIGID_STRESS_ENV/bin/python" -m pytest tests/rigid/test_stress.py tests/rigid/test_stress_nonconvex.py -n 0 --backend cpu --tb=short -q > "$run/stress-cpu.log" 2>&1 || failed=1
"$RIGID_STRESS_ENV/bin/python" -m pytest tests/rigid/test_narrowphase.py::test_split_vs_monolithic_narrowphase tests/rigid/test_narrowphase.py::test_contact_patch_full_box_box_manifold tests/rigid/test_islands.py::test_hibernation_with_pruning -n 0 --backend cuda --tb=short -q > "$run/collision-cuda.log" 2>&1 || failed=1
sha256sum "${sources[@]}" > "$run/source-after.sha256"
cmp "$run/source.sha256" "$run/source-after.sha256"
exit "$failed"
