#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{profile_reference_sensitivity,finite_traction_reference,load_model_validation,skfem_reference,native_load_samples,qualify_reference_scaling,spherical_shell_validation,validation_meshes}.py
    research/rigid_stress/scripts/{run_profile_sensitivity,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
lscpu > "$task_run/cpu.txt"
"$task_env/bin/python" - <<'PY' > "$task_run/environment.json"
import importlib.metadata
import json
import platform
import sys
print(json.dumps({"python": sys.version, "platform": platform.platform(),
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("numpy", "scipy", "scikit-fem", "gmsh", "pyamg", "ipctk", "quadrants")}}, indent=2))
PY
task_status=0
"$task_env/bin/python" -m research.rigid_stress.profile_reference_sensitivity --output "$task_run/results" "$@" \
    > "$task_run/validation.log" 2>&1 || task_status=$?
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
exit "$task_status"
