#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/native-refinement-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/scripts/publish_native_refinement_checkpoint.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
git rev-parse HEAD > "$task_tools/base-revision.txt"
task_names=(pcg-final-stress-suite-gpu-v1 native-fine-u-l0-v1 native-fine-u-l1-v1 native-fine-u-l2-v1)
TASK_NATIVE_EPOCH="$task_epoch" TASK_NATIVE_NAMES="${task_names[*]}" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

root = Path(os.environ["TASK_NATIVE_EPOCH"])
records = []
for name in os.environ["TASK_NATIVE_NAMES"].split():
    run = root / name
    before = (run / "source.sha256").read_bytes()
    assert before == (run / "source-after.sha256").read_bytes(), name
    for line in before.decode().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / relative).read_bytes()).hexdigest() == expected, (name, relative)
    record = {"name": name, "returncode": int((run / "returncode.txt").read_text())}
    if (run / "results/summary.json").exists():
        summary = json.loads((run / "results/summary.json").read_text())
        record.update(tetrahedra=summary["tetrahedra"], mesh_sha256=summary["mesh_sha256"],
                      classifications=dict(Counter(row["classification"] for row in summary["rows"])),
                      warmed_costs=summary["warmed_costs"])
    elif (run / "results.xml").exists():
        suites = ET.parse(run / "results.xml").getroot().iter("testsuite")
        record["testsuites"] = [dict(suite.attrib) for suite in suites]
    else:
        assert "CUDA_ERROR_OUT_OF_MEMORY" in (run / "validation.log").read_text(), name
        record["classification"] = "intrinsic_distance_allocation_failure_before_numerical_qualification"
    records.append(record)
print(json.dumps({"scope": "Complete current-source CUDA stress suite and finer native U sequences, including allocation failures",
                  "source_closures": records, "goal_completion": False,
                  "limitations": "Observed allocations do not define a universal GPU capacity endpoint. Full physical tails and converged egg/Franka/interior-hotspot contact remain required."}, indent=2))
PY
sacct -X -j 713500,713501 -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
sacct -X -j 713500,713501 -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_runs+=("$task_tools")
task_extra=("$task_tools/jobs.txt" "$task_tools/job-commands.txt")
for task_job in 713500 713501; do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/native-refinement-closeout-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "${task_runs[@]}" --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Complete current-source CUDA suite passes62 cases. The5669-tetrahedron sequence accepts39 fields and retains one unsupported pressure moment. Two finer8-port intrinsic-distance initializations fail allocation before mapping/recovery acceptance. Source snapshots, terminal status and failed logs remain exact; this checkpoint does not complete the physical-validation goal."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
