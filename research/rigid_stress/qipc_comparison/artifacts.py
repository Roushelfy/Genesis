"""Small metadata helpers independent of either simulator's import tree."""

import hashlib
import json
from pathlib import Path


def file_id(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
