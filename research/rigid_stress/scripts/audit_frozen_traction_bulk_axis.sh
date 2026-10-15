#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_qualification=${1:?Closed manufactured-observable qualification required}
shift
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{audit_frozen_traction_bulk_axis,audit_objective_frame,neohookean_reference,contact_reference,qualify_articulated_precision,sample_load_fields,skfem_reference,summarize_fine_references,validation_meshes}.py
    research/rigid_stress/scripts/{audit_frozen_traction_bulk_axis,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_run/input-source/." "$task_run/source/"
touch "$task_run/source/research/__init__.py" "$task_run/source/research/rigid_stress/__init__.py"
task_actual=("${task_sources[@]}" research/__init__.py research/rigid_stress/__init__.py)
(cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source.sha256"
trap 'task_status=$?; (cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source-after.sha256"; sha256sum "${task_sources[@]}" > "$task_run/input-source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
lscpu > "$task_run/cpu.txt"
"$task_env/bin/python" - <<'PY' > "$task_run/environment.json"
import importlib.metadata, json, platform, sys
print(json.dumps({"python":sys.version,"platform":platform.platform(),
                  "packages":{n:importlib.metadata.version(n) for n in ["numpy","scipy","scikit-fem","ipctk"]}},indent=2))
PY
(cd "$task_run/source"
    export PYTHONPATH="$task_run/source"
    printf '%s\n' "$task_qualification" > "$task_run/qualification-path.txt"
    "$task_env/bin/python" - "$task_qualification" <<'PYQUAL' > "$task_run/qualification.json"
import ast
import hashlib
import json
from pathlib import Path
import sys
from research.rigid_stress.summarize_fine_references import closed_source
q = Path(sys.argv[1])
closed_source(q)
assert int((q / "returncode.txt").read_text()) == 0
report = json.loads((q / "results/summary.json").read_text())
assert report["qualification_passed"]
relative = Path("research/rigid_stress/audit_frozen_traction_bulk_axis.py")
original = (q / "source" / relative).read_text()
current = relative.read_text()
def functions(source):
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
a, b = functions(original), functions(current)
for name in ["prolongation", "volume_squared"]:
    assert a[name] == b[name]
relative = Path("research/rigid_stress/skfem_reference.py")
assert (q / "source" / relative).read_bytes() == relative.read_bytes()
print(json.dumps({"qualification": str(q), "source_functions_verified": {name: hashlib.sha256(b[name].encode()).hexdigest() for name in ["prolongation", "volume_squared"]}, "unchanged_independent_FEM_source_verified": True}))
PYQUAL
    "$task_env/bin/python" -m research.rigid_stress.audit_frozen_traction_bulk_axis --output "$task_run/results" "$@" \
        > "$task_run/audit.log" 2>&1)
