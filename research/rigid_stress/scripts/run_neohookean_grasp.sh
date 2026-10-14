#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/input-source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{grasp_contact_reference,contact_reference,neohookean_reference,skfem_reference,finite_traction_reference}.py
    research/rigid_stress/neohookean_grasp_driver.patch research/rigid_stress/scripts/{run_neohookean_grasp,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_run/input-source/." "$task_run/source/"
patch -p1 -d "$task_run/source" < research/rigid_stress/neohookean_grasp_driver.patch > "$task_run/patch.log"
touch "$task_run/source/research/__init__.py" "$task_run/source/research/rigid_stress/__init__.py"
task_actual=("${task_sources[@]}" research/__init__.py research/rigid_stress/__init__.py)
(cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source.sha256"
trap 'task_status=$?; (cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source-after.sha256"; sha256sum "${task_sources[@]}" > "$task_run/input-source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"; exit "$task_status"' EXIT
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
                               ("numpy", "scipy", "scikit-fem", "ipctk")}}, indent=2))
PY
(cd "$task_run/source"
    export PYTHONPATH="$task_run/source"
    "$task_env/bin/python" -c 'from pathlib import Path; from research.rigid_stress import grasp_contact_reference as driver; assert Path(driver.__file__).resolve().is_relative_to(Path.cwd()); print(driver.__file__)' \
        > "$task_run/driver-origin.txt"
    "$task_env/bin/python" -m research.rigid_stress.grasp_contact_reference --output "$task_run/solution" "$@" \
        > "$task_run/contact.log" 2>&1)
