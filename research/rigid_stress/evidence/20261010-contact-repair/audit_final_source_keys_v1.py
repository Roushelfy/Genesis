import json
from pathlib import Path

run = Path(__file__).resolve().parent
reference = None
for batch in (1024, 2048, 32768, 49152, 61440):
    for seed in (510000, 623001):
        for scope in ("rigid", "recovery", "live", "policy"):
            directory = f"final-native-recovery-v10-b{batch}-seed{seed}" if scope == "recovery" else f"final-native-v9-b{batch}-seed{seed}"
            source = json.loads((run / directory / "summary.json").read_bytes())
            row = next(item for item in source["measurements"] if item["scope"] == scope)
            hashes = row["source_sha256"]
            if reference is None:
                reference = hashes
            differences = {key: [reference.get(key), hashes.get(key)] for key in reference.keys() | hashes.keys() if reference.get(key) != hashes.get(key)}
            print(batch, seed, scope, json.dumps(differences), flush=True)
