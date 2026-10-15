#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_mode=${2:?qualify or solve}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{dynamic_franka_reference,qualify_dynamic_franka,franka_contact_reference,franka_rigid_mode_coordinates,audit_franka_reference,contact_reference,neohookean_reference,stable_neohookean_reference,skfem_reference,summarize_fine_references}.py
    research/rigid_stress/scripts/{dynamic_franka_reference,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$task_mode $*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
lscpu > "$task_run/cpu.txt"
"$task_env/bin/python" - <<'PY' > "$task_run/environment.json"
import importlib.metadata
import json
import platform
import sys
print(json.dumps({"python":sys.version,"platform":platform.platform(),
                  "packages":{name:importlib.metadata.version(name) for name in
                              ("numpy","scipy","scikit-fem","ipctk")}},indent=2))
PY
case "$task_mode" in
    qualify) task_module=qualify_dynamic_franka ;;
    solve) task_module=dynamic_franka_reference ;;
    *) exit 2 ;;
esac
"$task_env/bin/python" -m "research.rigid_stress.$task_module" --output "$task_run/results" "$@" \
    > "$task_run/reference.log" 2>&1
