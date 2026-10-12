#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_backend=${1:?Backend required}
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/sample-contiguous-layout-v11-v1/$task_backend"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py tests/rigid/{test_stress_sources,test_stress_traction,test_stress_nonconvex,test_stress_footprint,test_stress_sliding_footprint,test_stress_surface_fan,test_stress_shared_support,test_stress_shared_lifecycle,test_stress_compatibility}.py tests/rigid/assets/nonconvex_{sliding,urdf,mesh64,plane_pair}_tick*.npz research/rigid_stress/{wrench,mechanics,field_cpu}.py "$task_root/traction_original_point_candidate_v10.py" "$task_root/traction_sample_cooperative_candidate_v2.py" "$task_root/probe_sample_layout_candidate_v11.py" "$task_root/run_sample_layout_candidate_v11.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
if [[ "$task_backend" == cuda ]]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
fi
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_sample_layout_candidate_v11.py" --test "$task_backend" > "$task_run/tests.log" 2>&1 || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
