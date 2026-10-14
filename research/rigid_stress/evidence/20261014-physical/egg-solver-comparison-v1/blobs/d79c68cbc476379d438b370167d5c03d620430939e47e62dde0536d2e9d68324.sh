#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_tools="$task_epoch/egg-solver-publication-tools-v1"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/scripts/publish_egg_solver_comparison.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
TASK_EGG_SOLVER_EPOCH="$task_epoch" "$task_env/bin/python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path
import numpy as np
root = Path(os.environ["TASK_EGG_SOLVER_EPOCH"])
runs = [root / name for name in ("egg-pcg-diagnostic-v3", "egg-direct-diagnostic-v2")]
for run in runs:
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    for line in (run / "source.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == expected
cases = [run / "results/opposing" for run in runs]
assert (cases[0] / "frozen_inputs.npz").read_bytes() == (cases[1] / "frozen_inputs.npz").read_bytes()
snapshots = [json.loads((case / "snapshot.json").read_text())["models"][0] for case in cases]
assert all(row["accepted"] for row in snapshots)
with np.load(cases[0] / "pressure_field.npz") as pcg, np.load(cases[1] / "pressure_field.npz") as direct:
    np.testing.assert_allclose(pcg["raw_native_tensor_Pa"], direct["raw_native_tensor_Pa"], rtol=1e-4, atol=1e-3)
    difference = np.linalg.norm(pcg["raw_native_tensor_Pa"] - direct["raw_native_tensor_Pa"]) / np.linalg.norm(direct["raw_native_tensor_Pa"])
print(json.dumps({"scope": "Matched full-egg native PCG/direct pressure solutions; both have inadmissible local traction",
                  "tensor_relative_difference": float(difference), "snapshots": snapshots,
                  "cost_scope": "Independent cold allocations; not a matched-device warmed throughput comparison",
                  "goal_completion": False}, indent=2))
PY
sacct -X -j 712347,712448 -o JobIDRaw,State,ExitCode,Elapsed,SubmitLine%8000 -n -P > "$task_tools/jobs.txt"
sacct -j 712347,712448 -o JobIDRaw,State,MaxRSS,Elapsed -n -P > "$task_tools/memory.txt"
task_dest=research/rigid_stress/evidence/20261014-physical/egg-solver-comparison-v1
"$task_env/bin/python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "$task_epoch/egg-direct-diagnostic-v2" "$task_tools" \
    --extra "$task_epoch/egg-direct-diagnostic-v2-712347_0.log" "$task_epoch/egg-direct-diagnostic-v2-712347_0.err" \
    --summary "$task_tools/summary.json" --limitations "Both native solvers satisfy the unchanged numerical gates for identical frozen inputs. Pressure local traction still violates compression/friction. Direct allocation completes in58m20s, versus10m46s for the PCG cold allocation, including unrelated initialization/reference overheads and on separate allocations. No warmed throughput ratio or physical contact acceptance is inferred. PCG raw evidence is retained in the main load-model checkpoint."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_env/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
