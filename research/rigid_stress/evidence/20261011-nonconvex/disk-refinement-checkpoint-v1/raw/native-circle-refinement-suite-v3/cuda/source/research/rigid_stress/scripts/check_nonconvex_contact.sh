#!/usr/bin/env bash
set -euo pipefail
task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_repo"
source research/rigid_stress/scripts/cluster_env.sh
task_backend=${1:-cuda}
case "$task_backend" in
    cpu|cuda) ;;
    *) echo 'Backend must be cpu or cuda.' >&2; exit 2 ;;
esac
task_run=${RIGID_STRESS_RUN_DIR:-"$RIGID_STRESS_DATA_ROOT/runs/nonconvex-contact-${SLURM_JOB_ID:-local}-$task_backend"}
if [[ -e "$task_run/tests.log" || -e "$task_run/source.sha256" ]]; then
    echo 'Choose a new RIGID_STRESS_RUN_DIR to preserve existing evidence.' >&2
    exit 2
fi
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py genesis/engine/entities/rigid_entity/description.py tests/rigid/{test_stress_sources,test_stress_traction,test_stress_nonconvex,test_stress_footprint,test_stress_sliding_footprint,test_stress_surface_fan,test_stress_disk_refinement,test_stress_disk_torque,test_stress_shared_support,test_stress_shared_lifecycle,test_stress_compatibility}.py tests/rigid/assets/nonconvex_*.npz examples/rigid/assets/hollow_egg/level1/elastic.npz research/rigid_stress/{wrench,mechanics,field_cpu}.py)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis; print(genesis.__file__)' > "$task_run/import.log" 2>&1
hostname > "$task_run/node.txt"
if [[ "$task_backend" == cuda ]]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
fi
task_failed=0
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend "$task_backend" --tb=short -q tests/rigid/test_stress_nonconvex.py tests/rigid/test_stress_sources.py tests/rigid/test_stress_traction.py tests/rigid/test_stress_footprint.py tests/rigid/test_stress_sliding_footprint.py tests/rigid/test_stress_surface_fan.py tests/rigid/test_stress_disk_refinement.py tests/rigid/test_stress_disk_torque.py tests/rigid/test_stress_shared_support.py tests/rigid/test_stress_shared_lifecycle.py tests/rigid/test_stress_compatibility.py > "$task_run/tests.log" 2>&1 || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
