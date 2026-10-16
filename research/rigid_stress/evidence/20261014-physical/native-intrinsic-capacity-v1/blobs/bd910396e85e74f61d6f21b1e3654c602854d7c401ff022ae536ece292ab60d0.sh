#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/native-port-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{native_physical_sequence,summarize_native_port_axis}.py
    research/rigid_stress/scripts/{publish_native_port_axis,run_native_physical_sequence,summarize_native_port_axis}.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
git rev-parse HEAD > "$task_tools/base-revision.txt"
task_names=(native-port-u-l0-s4-v1 native-port-u-l0-s2-v1 native-port-u-l1-s4-v1
    native-port-u-l1-s2-v1 native-port-u-l2-s2-v1 native-pressure-u-l2-v1 native-port-axis-summary-v1)
TASK_PORT_EPOCH="$task_epoch" TASK_PORT_NAMES="${task_names[*]}" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["TASK_PORT_EPOCH"])
records = []
for name in os.environ["TASK_PORT_NAMES"].split():
    run = root / name
    before = (run / "source.sha256").read_bytes()
    assert before == (run / "source-after.sha256").read_bytes(), name
    for line in before.decode().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / relative).read_bytes()).hexdigest() == expected, (name, relative)
    records.append({"name": name, "returncode": int((run / "returncode.txt").read_text())})
summary = json.loads((root / "native-port-axis-summary-v1/results/summary.json").read_text())
print(json.dumps({"scope": "Numerical intrinsic-port and combined remeshing axes for native U histories, including pure-pressure independence",
                  "source_closures": records, "analysis": summary, "goal_completion": False}, indent=2))
PY
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_runs+=("$task_tools")
task_extra=("$task_epoch/native-port-axis-v1.sbatch" "$task_epoch/native-port-axis-v2.sbatch"
    "$task_epoch/native-port-axis-v1.json" "$task_epoch/native-port-axis-v1-launch-audit.json"
    "$task_epoch/native-port-axis-summary-v1.sbatch")
sacct -X -j 713572,713581,713607 -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
sacct -X -j 713572,713581,713607 -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
for task_job in 713572 713581 713607; do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/native-port-axis-checkpoint-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "${task_runs[@]}" --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Port resolution is an explicit numerical axis with fixed physical radius/friction/wrenches and unchanged complete acceptance tolerances. Same-mesh input bytes are checked. Combined remeshing retains changes in force application points and wrenches. Pure pressure requires no intrinsic-distance cache. Raw full fields, failures and source remain exact. Earlier baseline fields are in their existing publications and are not duplicated. Converged full egg/Franka/interior-hotspot contact and complete physical operating-range acceptance remain required."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
