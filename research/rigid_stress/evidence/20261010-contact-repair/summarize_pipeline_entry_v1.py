"""Read completed matched entry experiments on an allocated node."""

import hashlib
import json
from pathlib import Path

import numpy as np

run = Path(__file__).resolve().parent
comparisons = []
for batch in (1024, 32768):
    for seed in (510000, 623001):
        path = run / f"native-pipeline-entry-b{batch}-seed{seed}-v1.json"
        payload = path.read_bytes()
        source = json.loads(payload)
        assert source["passed"]
        rates = {}
        for row in source["rows"]:
            repeats = row["repeats"]
            assert all(item["invalid_environment_steps"] == 0 for item in repeats)
            rates[row["route"]] = {
                "rate": batch * sum(item["steps"] for item in repeats) / sum(item["seconds"] for item in repeats),
                "std": float(np.std([item["valid_env_steps_per_second"] for item in repeats], ddof=1)),
            }
        result = {
            "envs": batch,
            "seed": seed,
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rates": rates,
            "direct_body_gain_percent": 100 * (rates["direct_body"]["rate"] / rates["func_wrapper"]["rate"] - 1),
            "maximum_absolute_differences": source["maximum_absolute_differences"],
            "gpu_uuid": source["gpu_uuid"],
        }
        comparisons.append(result)
        print(json.dumps(result), flush=True)
for batch in (1024, 2048, 32768, 49152, 61440):
    for seed in (510000, 623001):
        path = run / f"final-native-recovery-v10-b{batch}-seed{seed}/summary.json"
        source = json.loads(path.read_bytes())
        assert len(source["measurements"]) == 1
        assert source["measurements"][0]["status"] == "completed"
        print("primed-recovery-completed", batch, seed, flush=True)
(run / "native-pipeline-entry-complete-v1-summary.json").write_text(json.dumps({"comparisons": comparisons, "passed": True}, indent=2) + "\n")
