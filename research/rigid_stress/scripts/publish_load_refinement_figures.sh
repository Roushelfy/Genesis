#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_tools="$task_epoch/load-figure-publication-tools-v1"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{plot_load_refinement,plot_physical_controls}.py
    research/rigid_stress/scripts/{publish_load_refinement_figures,plot_load_refinement}.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
TASK_LOAD_FIGURE_EPOCH="$task_epoch" "$task_env/bin/python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ["TASK_LOAD_FIGURE_EPOCH"])
for version in ("v1", "v2"):
    run = root / ("load-refinement-figures-" + version)
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    for line in (run / "source.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == expected
print(json.dumps({"scope": "Two editable six-panel refinement figures and exact CSV inputs; original label overlap corrected in v2",
                  "visual_qa": "Both selected v2 previews inspected; consistent units, raw peaks, complete tensor changes and explicit pressure-cone limitation",
                  "goal_completion": False}, indent=2))
PY
sacct -X -j 712520,712532 -o JobIDRaw,State,ExitCode,Elapsed,SubmitLine%8000 -n -P > "$task_tools/jobs.txt"
task_dest=research/rigid_stress/evidence/20261014-physical/load-refinement-figures-v1
"$task_env/bin/python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "$task_epoch/load-refinement-figures-v1" "$task_epoch/load-refinement-figures-v2" "$task_tools" \
    --extra "$task_epoch/load-refinement-figures-v1-712520_0.log" "$task_epoch/load-refinement-figures-v2-712532_0.log" \
    --summary "$task_tools/summary.json" --limitations "The curves combine volume and geometry changes. No complete3% tail passes. They show selected opposing loads; all five load cases are retained in the main checkpoint CSV. Narrow-gap pressure is numerically qualified but locally inadmissible. These figures do not establish separated refinement axes or deformable-grasp accuracy."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_env/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
