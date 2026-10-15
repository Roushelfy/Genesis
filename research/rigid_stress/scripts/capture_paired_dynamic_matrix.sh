#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_parent=${2:?Closed paired source with saved initialization closure required}
shift 2
test ! -e "$task_run"
cmp "$task_parent/source.sha256" "$task_parent/source-after.sha256"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export REFERENCE_LINEAR_SOLVER_STATS_PATH="$task_run/linear_solver_stats.json"
export REFERENCE_METIS_STATS_PATH="$task_run/metis_factor_stats.json"
task_sources=(research/rigid_stress/capture_paired_dynamic_matrix.py research/rigid_stress/scripts/capture_paired_dynamic_matrix.sh research/rigid_stress/scripts/cluster_env.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_parent/source/." "$task_run/source/"
cp --parents research/rigid_stress/capture_paired_dynamic_matrix.py research/rigid_stress/scripts/capture_paired_dynamic_matrix.sh "$task_run/source"
cp "$task_parent/source.sha256" "$task_run/parent-source.sha256"
printf '%s\n' "$task_parent" > "$task_run/parent-source-path.txt"
mapfile -t task_parent_sources < <(sed 's/^[^ ]*  //' "$task_parent/source.sha256")
task_actual=("${task_parent_sources[@]}" research/rigid_stress/capture_paired_dynamic_matrix.py research/rigid_stress/scripts/capture_paired_dynamic_matrix.sh)
(cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source.sha256"
trap 'task_status=$?; (cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source-after.sha256"; sha256sum "${task_sources[@]}" > "$task_run/input-source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
lscpu > "$task_run/cpu.txt"
"$task_env/bin/python" - <<'PY' > "$task_run/environment.json"
import importlib.metadata,json,platform,sys
print(json.dumps({"python":sys.version,"platform":platform.platform(),"packages":{n:importlib.metadata.version(n) for n in ["numpy","scipy","scikit-fem","ipctk"]}},indent=2))
PY
(cd "$task_run/source"
    task_extra_path=""
    if test -f "$task_parent/dependency-path.txt"; then IFS= read -r task_extra_path < "$task_parent/dependency-path.txt"; fi
    export PYTHONPATH="$task_run/source${task_extra_path:+:$task_extra_path}"
    "$task_env/bin/python" -m research.rigid_stress.capture_paired_dynamic_matrix --source-run "$task_parent" --output "$task_run/results" "$@" > "$task_run/capture.log" 2>&1)
