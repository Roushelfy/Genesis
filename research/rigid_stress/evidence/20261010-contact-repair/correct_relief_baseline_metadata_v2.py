"""Preserve and correct the inactive research override's baseline configuration label."""

import argparse
import hashlib
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", type=Path)
args = parser.parse_args()
content = args.path.read_bytes()
metadata = json.loads(content)
assert metadata["cached_relief"] is False and metadata["micro"] is False
backup = args.path.with_name(args.path.stem + ".original.json")
assert not backup.exists()
backup.write_bytes(content)
metadata["reduced_wrench_atomics"] = False
metadata["metadata_correction"] = {
    "reason": "The reduction override is applied only when cached_relief or micro is enabled. Both flags are false in this baseline.",
    "original_path": str(backup),
    "original_sha256": hashlib.sha256(content).hexdigest(),
}
args.path.write_text(json.dumps(metadata, indent=2) + "\n")
print("Corrected inactive baseline label and preserved", backup)
