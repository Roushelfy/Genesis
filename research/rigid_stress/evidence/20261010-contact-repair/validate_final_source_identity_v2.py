import copy
import hashlib
import json
from pathlib import Path

from research.rigid_stress.scripts.summarize_final_native import BENCHMARK_PREPARATION_HASHES, numerical_sources

run = Path(__file__).resolve().parent
rows = []
for batch in (1024, 2048, 32768, 49152, 61440):
    for seed in (510000, 623001):
        for scope in ("rigid", "recovery", "live", "policy"):
            directory = f"final-native-recovery-v10-b{batch}-seed{seed}" if scope == "recovery" else f"final-native-v9-b{batch}-seed{seed}"
            source = json.loads((run / directory / "summary.json").read_bytes())
            row = next(item for item in source["measurements"] if item["scope"] == scope)
            assert row["status"] == "completed"
            rows.append(row)
for batch in (1024, 32768):
    for seed in (510000, 623001):
        for scope in ("live", "policy"):
            for mode in ("max", "full"):
                directory = f"final-native-fullcost-v9-{scope}-{mode}-b{batch}-seed{seed}"
                source = json.loads((run / directory / "summary.json").read_bytes())
                assert len(source["measurements"]) == 1
                row = source["measurements"][0]
                assert row["status"] == "completed"
                rows.append(row)
reference = numerical_sources(rows[0], True)
assert len(rows) == 56
assert all(numerical_sources(row, True) == reference for row in rows)
changed_engine = copy.deepcopy(rows[0])
changed_engine["source_sha256"]["genesis/engine/solvers/rigid/stress/pipeline.py"] = "0" * 64
assert numerical_sources(changed_engine, True) != reference
changed_benchmark = copy.deepcopy(rows[0])
key = next(key for key in changed_benchmark["source_sha256"] if key.endswith("/examples/speed_benchmark/rigid_stress.py"))
changed_benchmark["source_sha256"][key] = "0" * 64
try:
    numerical_sources(changed_benchmark, True)
except AssertionError:
    pass
else:
    raise AssertionError("An unreviewed benchmark change was accepted.")
result = {
    "passed": True,
    "completed_cases": len(rows),
    "numerical_and_configuration_source_sha256": reference,
    "audited_benchmark_preparation_source_sha256": BENCHMARK_PREPARATION_HASHES,
    "unreviewed_engine_change_rejected": True,
    "unreviewed_benchmark_change_rejected": True,
    "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "summarizer_sha256": hashlib.sha256(Path("research/rigid_stress/scripts/summarize_final_native.py").read_bytes()).hexdigest(),
}
(run / "native-final-source-identity-v2.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
