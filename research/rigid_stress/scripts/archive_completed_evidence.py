"""Append explicitly completed artifacts without disturbing earlier provenance records."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from research.rigid_stress.scripts.archive_contact_evidence import compact_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"artifacts": []}
    entries = {item["runtime_path"]: item for item in manifest["artifacts"]}
    for name in args.names:
        runtime = args.run / name
        content = runtime.read_bytes()
        original_bytes = len(content)
        original_hash = hashlib.sha256(content).hexdigest()
        artifact = args.output / name.replace("/", "__")
        format_name = "complete"
        if name.endswith(".json") and original_bytes > 500_000:
            result = compact_arrays(json.loads(content))
            result["publication_note"] = (
                "Long environment arrays summarized. Exact original remains at runtime_path with original_sha256."
            )
            content = (json.dumps(result, indent=2) + "\n").encode()
            artifact = artifact.with_suffix(".summary.json")
            format_name = "summary"
        elif original_bytes > 500_000:
            content = gzip.compress(content, mtime=0)
            artifact = artifact.with_suffix(artifact.suffix + ".gz")
            format_name = "gzip"
        artifact.write_bytes(content)
        entries[str(runtime)] = {
            "source_revision": args.revision,
            "runtime_path": str(runtime),
            "artifact": artifact.name,
            "format": format_name,
            "original_sha256": original_hash,
            "original_bytes": original_bytes,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    manifest["artifacts"] = list(entries.values())
    manifest["status"] = "Completed checkpoints; final acceptance status is recorded in the results document."
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Archived", len(args.names), "explicit completed artifacts; manifest contains", len(entries), flush=True)


if __name__ == "__main__":
    main()
