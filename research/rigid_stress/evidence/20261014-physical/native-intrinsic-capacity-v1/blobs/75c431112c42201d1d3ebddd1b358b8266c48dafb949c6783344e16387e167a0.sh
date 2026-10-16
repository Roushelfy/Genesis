#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_qualification=${1:?Closed recording/current-friction qualification required}
shift
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/input-source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export REFERENCE_LINEAR_SOLVER_STATS_PATH="$task_run/linear_solver_stats.json"
export REFERENCE_ACCEPTED_CONTACT_STATES_ROOT="$task_run/accepted-contact-history"
task_sources=(research/rigid_stress/{cached_reference_newton,grasp_contact_reference,contact_reference,neohookean_reference,skfem_reference,finite_traction_reference}.py
    research/rigid_stress/{neohookean_grasp_driver,cached_reference_newton,rigid_nonlinear_approach}.patch research/rigid_stress/scripts/{run_rigid_initialized_nonlinear,cluster_env}.sh)
task_sources+=(research/rigid_stress/reference_contact_history.py research/rigid_stress/reference_contact_history.patch research/rigid_stress/scripts/run_recorded_nonlinear_contact.sh)
task_sources+=(research/rigid_stress/summarize_fine_references.py research/rigid_stress/fixed_contact_initialization_history.patch research/rigid_stress/scripts/run_fixed_history_barrier.sh)
task_sources+=(research/rigid_stress/common_u_initialization_anchors.patch research/rigid_stress/scripts/run_common_initialized_contact.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_run/input-source/." "$task_run/source/"
patch -p1 -d "$task_run/source" < research/rigid_stress/neohookean_grasp_driver.patch > "$task_run/patch.log"
patch -p1 -d "$task_run/source" < research/rigid_stress/cached_reference_newton.patch >> "$task_run/patch.log"
patch -p1 -d "$task_run/source" < research/rigid_stress/rigid_nonlinear_approach.patch >> "$task_run/patch.log"
patch -p1 -d "$task_run/source" < research/rigid_stress/reference_contact_history.patch >> "$task_run/patch.log"
patch -p1 -d "$task_run/source" < research/rigid_stress/fixed_contact_initialization_history.patch >> "$task_run/patch.log"
patch -p1 -d "$task_run/source" < research/rigid_stress/common_u_initialization_anchors.patch >> "$task_run/patch.log"
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
    printf '%s\n' "$task_qualification" > "$task_run/qualification-path.txt"
    "$task_env/bin/python" - "$task_qualification" "$task_run/tmp" <<'PYQUAL' > "$task_run/qualification.json"
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from research.rigid_stress.summarize_fine_references import closed_source
q, tmp = map(Path, sys.argv[1:])
closed_source(q)
assert int((q / "returncode.txt").read_text()) == 0
report = json.loads((q / "results/summary.json").read_text())
assert report["stages_checked"] == 744
assert report["all_retained_stages_independent_current_friction_and_original_force_gate"]
producer = Path(report["run"])
closed_source(producer)
assert int((producer / "returncode.txt").read_text()) == 0
recorded = json.loads((producer / "accepted-contact-history/manifest.json").read_text())
assert next(call for call in recorded["calls"] if call["phase"] == "contact_initialization")["load_steps"] == 604
for name in ["contact_reference", "neohookean_reference", "skfem_reference", "cached_reference_newton", "reference_contact_history"]:
    relative = Path("research/rigid_stress") / (name + ".py")
    assert (producer / "source" / relative).read_bytes() == relative.read_bytes()
relative = Path("research/rigid_stress/grasp_contact_reference.py")
expected = tmp / "qualified-driver" / relative
expected.parent.mkdir(parents=True)
expected.write_bytes((producer / "source" / relative).read_bytes())
subprocess.run(["patch", "-p1", "-d", str(tmp / "qualified-driver")], input=Path("research/rigid_stress/fixed_contact_initialization_history.patch").read_bytes(), stdout=subprocess.DEVNULL, check=True)
subprocess.run(["patch", "-p1", "-d", str(tmp / "qualified-driver")], input=Path("research/rigid_stress/common_u_initialization_anchors.patch").read_bytes(), stdout=subprocess.DEVNULL, check=True)
assert expected.read_bytes() == relative.read_bytes()
print(json.dumps({"qualified_history_audit": str(q), "qualified_producer": str(producer), "same_contact_bulk_solver_recording_sources_verified": True, "driver_diff_is_only_declared_initialization_count_and_anchor_options": True}))
PYQUAL
    "$task_env/bin/python" -m research.rigid_stress.grasp_contact_reference --output "$task_run/solution" "$@" \
        > "$task_run/contact.log" 2>&1)
