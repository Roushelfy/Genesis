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
task_run=${RIGID_STRESS_RUN_DIR:-"$RIGID_STRESS_DATA_ROOT/runs/nonconvex-witness-${SLURM_JOB_ID:-local}-$task_backend"}
if [[ -e "$task_run/tests.log" || -e "$task_run/source.sha256" ]]; then
    echo 'Choose a new RIGID_STRESS_RUN_DIR to preserve existing evidence.' >&2
    exit 2
fi
mkdir -p "$task_run/tmp" "$task_run/cache/quadrants" "$task_run/cache/genesis"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/utils/array_class.py genesis/engine/solvers/rigid/collider/{collider,contact,gjk,mpr,narrowphase,witness}.py tests/rigid/{test_stress_witness,test_stress_witness_manifold,test_stress_nonconvex}.py)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis; print(genesis.__file__)' > "$task_run/import.log" 2>&1
if [[ "$task_backend" == cuda ]]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/gpu.csv"
fi
task_tests=(tests/rigid/test_stress_witness.py tests/rigid/test_stress_witness_manifold.py)
if [[ "${RIGID_STRESS_CHECK_EXISTING:-0}" == 1 ]]; then
    task_tests+=(tests/rigid/test_stress.py tests/rigid/test_stress_nonconvex.py)
fi
task_failed=0
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend "$task_backend" --tb=short -q "${task_tests[@]}" > "$task_run/tests.log" 2>&1 || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
