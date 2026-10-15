"""Verify archived bytes against actual producer paths as well as staging representations."""

import argparse
import json
from pathlib import Path

from research.rigid_stress.scripts.verify_nonconvex_publication import digest, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = {}
    archived = verify_manifest(args.manifest, True, cache)
    manifest = json.loads(args.manifest.read_text())
    derived = set()
    actual = {}
    for index in manifest["summary"].get("sharded_original_files", []):
        original = Path(index["original"])
        relative = original.relative_to(args.runtime_root)
        staged = args.staging_root / relative
        derived.add(staged.with_name(staged.name + ".chunks.json"))
        derived.update(staged.parent / part["file"] for part in index["parts"])
        expected = (index["sha256"], index["bytes"])
        assert digest(original) == expected, original
        actual[str(original)] = expected
    derived_checked = 0
    for entry in manifest["files"]:
        source = Path(entry["original"])
        if source in derived:
            derived_checked += 1
            continue
        if source.is_relative_to(args.staging_root):
            source = args.runtime_root / source.relative_to(args.staging_root)
        expected = (entry["sha256"], entry["bytes"])
        if str(source) not in actual:
            assert digest(source) == expected, source
            actual[str(source)] = expected
        else:
            assert actual[str(source)] == expected, source
    assert derived_checked == len(derived)
    result = {
        "scope": __doc__,
        "manifest_sha256": digest(args.manifest)[0],
        "archive_entries_verified": archived["verified_entries"],
        "actual_runtime_originals_verified": len(actual),
        "reconstructed_large_originals": archived["reconstructed_complete_files"],
        "derived_chunk_and_index_entries_verified": derived_checked,
        "actual_original_sha256_and_bytes": actual,
        "passed": True,
        "goal_completion": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "actual_original_sha256_and_bytes"}))


if __name__ == "__main__":
    main()
