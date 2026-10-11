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
task_run=${RIGID_STRESS_RUN_DIR:-"$RIGID_STRESS_DATA_ROOT/runs/nonconvex-producers-${SLURM_JOB_ID:-local}-$task_backend"}
if [[ -e "$task_run/source.sha256" ]]; then
    echo 'Choose a new RIGID_STRESS_RUN_DIR to preserve existing evidence.' >&2
    exit 2
fi
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py research/rigid_stress/{probe_nonconvex_producers,mechanics,field_cpu}.py research/rigid_stress/scripts/check_nonconvex_producers.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git rev-parse HEAD > "$task_run/revision.txt"
hostname > "$task_run/node.txt"
if [[ "$task_backend" == cuda ]]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
fi
task_failed=0
for task_case in primitive:cube decomposition:u sdf:u sdf:cavity; do
    task_producer=${task_case%:*}
    task_shape=${task_case#*:}
    "$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.probe_nonconvex_producers --shape "$task_shape" --producer "$task_producer" --backend "$task_backend" --output "$task_run/$task_producer-$task_shape/results" > "$task_run/$task_producer-$task_shape.log" 2>&1 || task_failed=1
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
