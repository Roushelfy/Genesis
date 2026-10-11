#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_run="$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_revision=c5d7f160
exec > "$task_run/native-final-pipeline-v13.log" 2>&1
"$RIGID_STRESS_ENV/bin/python" - "$task_run" <<'PY'
import json
import sys
import time
from pathlib import Path

run = Path(sys.argv[1])
requirements = []
for batch in (1024, 2048, 32768, 49152, 61440):
    for seed in (510000, 623001):
        for scope in ("rigid", "live", "policy"):
            requirements.append((run / f"final-native-v9-b{batch}-seed{seed}/summary.json", scope))
        requirements.append((run / f"final-native-recovery-v10-b{batch}-seed{seed}/summary.json", "recovery"))
for batch in (1024, 32768):
    for seed in (510000, 623001):
        for scope in ("live", "policy"):
            for mode in ("max", "full"):
                requirements.append((run / f"final-native-fullcost-v9-{scope}-{mode}-b{batch}-seed{seed}/summary.json", scope))
deadline = time.monotonic() + 70 * 60
previous = None
last_notice = 0.0
while True:
    completed = 0
    for path, scope in requirements:
        try:
            source = json.loads(path.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        matches = [row for row in source["measurements"] if row["scope"] == scope]
        assert len(matches) <= 1, (path, scope)
        if matches:
            status = matches[0]["status"]
            assert not status.startswith("failed"), (path, scope, status)
            completed += status == "completed"
    now = time.monotonic()
    if completed != previous or now - last_notice >= 60:
        print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "Closed required timed cases", completed, "/", len(requirements), flush=True)
        previous, last_notice = completed, now
    if completed == len(requirements):
        break
    if now >= deadline:
        raise SystemExit("Timed cases remain incomplete; no final summary/publication was attempted.")
    time.sleep(20)
PY
echo 'Summarizing complete v9/v10 matrix'
bash "$task_run/run_native_final_summary_v9.sh" complete --require-complete
echo 'Rendering complete v9/v10 tables'
bash "$task_run/run_native_final_render_v9.sh"
echo 'Publishing closed final v13 evidence'
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/archive_final_native.py --run "$task_run" --output research/rigid_stress/evidence/20261010-contact-repair --revision "$task_revision" --checkpoint v13 --summary "$task_run/native-final-comprehensive-complete-v9-v10-summary.json" --require-complete > "$task_run/native-final-v13-archive.log" 2>&1
echo 'Final summary, tables and original/publication hash verification completed'
