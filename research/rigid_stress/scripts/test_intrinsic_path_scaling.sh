#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_backend=${2:?cpu or gpu required}
shift 2
test "$task_backend" = cpu || test "$task_backend" = gpu
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 QD_OFFLINE_CACHE=0
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export NUMBA_CACHE_DIR="$task_run/cache/numba" MPLCONFIGDIR="$task_run/cache/matplotlib"
task_sources=()
while IFS= read -r task_source; do task_sources+=("$task_source"); done < <(
    python3 - <<'PY'
from pathlib import Path
files = [p for root in ('genesis', 'tests') for p in Path(root).rglob('*.py')]
files += list(Path('research/rigid_stress').glob('*.py'))
files += list(Path('tests/rigid/data').glob('stress_corner_cancellation.*'))
files += [Path('examples/rigid/assets/hollow_egg/level1/elastic.npz')]
files += list(Path('tests/rigid/assets').glob('nonconvex_plane_pair_tick*.npz'))
files += [Path('research/rigid_stress/scripts/test_intrinsic_path_scaling.sh'),
          Path('research/rigid_stress/scripts/cluster_env.sh'), Path('pyproject.toml')]
for path in sorted(set(files)):
    print(path)
PY
)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
git rev-parse HEAD > "$task_run/base-revision.txt"
git diff HEAD > "$task_run/source.patch"
printf '%s\n' "$task_backend" "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
"$task_env/bin/python" - <<'PY' > "$task_run/import-origin.json"
import json
from pathlib import Path
import genesis
assert Path(genesis.__file__).resolve().is_relative_to(Path.cwd())
print(json.dumps({'genesis':genesis.__file__},indent=2))
PY
if test "$task_backend" = gpu; then
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
else
    lscpu > "$task_run/hardware.txt"
fi
"$task_env/bin/python" -m pytest -n 0 --backend "$task_backend" -m required \
    tests/rigid/test_stress_footprint.py "$@" --maxfail=1 \
    --basetemp "$task_run/pytest-tmp" -o "cache_dir=$task_run/pytest-cache" \
    --junitxml "$task_run/junit.xml" > "$task_run/pytest.log" 2>&1
