#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_backend=${2:?cpu or gpu required}
task_fixture_source=${3:-}
test "$task_backend" = cpu || test "$task_backend" = gpu
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
trap 'task_status=$?; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; exit "$task_status"' EXIT
if test -n "$task_fixture_source"; then
    "$task_env/bin/python" - "$task_fixture_source" "$task_run" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
from research.rigid_stress.summarize_fine_references import closed_source
source,run=map(Path,sys.argv[1:])
closed_source(source)
assert int((source/'returncode.txt').read_text())==0
summary=json.loads((source/'results/summary.json').read_text())
assert summary['all_existing_component_budgets_passed']
assert all(row['raw_global_peak_existing_budget_passed'] for row in summary['rows'])
original=source/'results/regression_fixture.npz'
provenance=source/'results/regression_fixture_provenance.json'
target=Path('tests/rigid/data/stress_corner_cancellation.npz')
metadata=target.with_suffix('.json')
assert not target.exists() and not metadata.exists()
target.parent.mkdir(parents=True,exist_ok=True)
with np.load(original) as saved:
 arrays={key:saved[key].copy() for key in saved.files}
 expected=arrays['expected_corner_components_Pa']
 rounded=expected.astype(np.float64)
 fraction=float(np.max(abs(expected-rounded)/(1e-5+1e-9*abs(expected))))
 assert fraction<1e-6
 arrays['expected_corner_components_Pa']=rounded
np.savez_compressed(target,**arrays)
record={**json.loads(provenance.read_text()),'complete_qualification_summary':str(source/'results/summary.json'),'complete_qualification_summary_sha256':hashlib.sha256((source/'results/summary.json').read_bytes()).hexdigest(),'original_fixture_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'portable_fixture_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'expected_values_dtype':'float64','maximum_reference_conversion_existing_budget_fraction':fraction,'scope':'One unchanged tetrahedron and frozen nodal values extracted from complete96-digit independent polynomial verification; target conversion is explicitly bounded.'}
metadata.write_text(json.dumps(record,indent=2)+'\n')
(run/'fixture-preparation.json').write_text(json.dumps(record,indent=2)+'\n')
PY
fi
task_sources=(genesis/options/rigid_stress.py
    research/rigid_stress/{field_cpu,mechanics,peak_cpu,wrench,skfem_reference,native_load_samples,summarize_fine_references}.py
    research/rigid_stress/scripts/{test_corner_stress_regression,cluster_env}.sh
    tests/conftest.py tests/rigid/{test_stress,test_stress_corner_precision,test_stress_independent_fem,test_stress_nonconvex}.py
    tests/rigid/data/stress_corner_cancellation.{npz,json})
while IFS= read -r task_source; do task_sources+=("$task_source"); done < <(rg --files genesis/engine/solvers/rigid/stress -g '*.py' | sort)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$task_backend" "$task_fixture_source" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
if test "$task_backend" = gpu; then
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
else
    lscpu > "$task_run/hardware.txt"
fi
"$task_env/bin/python" - <<'PY' > "$task_run/import-origin.json"
import json,genesis
from pathlib import Path
assert Path(genesis.__file__).resolve().is_relative_to(Path.cwd())
print(json.dumps({'genesis':genesis.__file__},indent=2))
PY
"$task_env/bin/python" -m pytest -n 0 --backend "$task_backend" -m required \
    tests/rigid/test_stress.py tests/rigid/test_stress_corner_precision.py tests/rigid/test_stress_independent_fem.py \
    -k 'corner_cancellation or full_field or batched_recovery_against_full_fp64_direct or history_reuse_abrupt_load_and_reset or (independent_fem and uniform_strain and direct)' \
    --maxfail=1 --basetemp "$task_run/pytest-tmp" -o "cache_dir=$task_run/pytest-cache" \
    --junitxml "$task_run/junit.xml" > "$task_run/pytest.log" 2>&1
