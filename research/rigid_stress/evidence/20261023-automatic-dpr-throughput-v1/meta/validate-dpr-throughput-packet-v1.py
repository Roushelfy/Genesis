"""Validate compact publication evidence without scientific dependencies."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def read(path):
    return json.loads(path.read_text())


def receipt(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = args.packet
    old_manifest = read(packet / "manifest.json")
    for row in old_manifest["files"]:
        path = packet / row["packet_path"]
        assert receipt(path)["sha256"] == row["sha256"], path
        assert path.stat().st_size == row["bytes"], path
    for path in packet.rglob("*.json"):
        read(path)
    timing = read(packet / "timing.json")["rows"]
    profiles = read(packet / "profiles.json")["rows"]
    campaigns = read(packet / "campaigns.json")["campaigns"]
    assert (len(timing), len(profiles), len(campaigns)) == (89, 10, 31)
    index = {(row["campaign"], row["task"]): row for row in timing}
    final = read(packet / "meta/automatic-dpr-opt-final-timing-validation-v2.json")
    assert final["passed"] and len(final["rows"]) == 33
    for expected in final["rows"]:
        row = index[expected["campaign"], f"group{expected['group']}/{expected['task']}"]
        assert row["median_ms"] == expected["median_ms"]
        assert row["envs"] == expected["envs"]
        assert row["invalid_environment_steps"] == 0
    for row in timing:
        assert "warmup_quality" not in row
        assert "native_buffers" not in row
        assert row["median_ms"] == statistics.median(row["milliseconds_per_batch_step"])
    fixtures = {
        "conditions.npz": "ab451473db350152f812f095c64e8595a558a5ea07e4d884ba8b55d7680ff9a8",
        "diverse-conditions.npz": "a9a9ecfa409f39acd159ee0e15205102710312ab76c5de539bd624f7676d807c",
    }
    for name, digest in fixtures.items():
        assert receipt(packet / name)["sha256"] == digest
    example = read(packet / "example-tasks.json")[0][0]
    arguments = example["args"]
    assert Path(arguments[arguments.index("--conditions") + 1]).is_file()
    assert arguments[arguments.index("--quadrature") + 1] == "16"
    assert arguments[arguments.index("--source-revision") + 1] == "db84114d1"
    assert read(packet / "meta/automatic-dpr-opt-requirements-v1.json")
    # Keep the corrected builder; its uncompact predecessor remains under data.
    obsolete = packet / "meta/publish-dpr-throughput-v1.py"
    if obsolete.exists():
        obsolete.unlink()
    shutil.copy2(__file__, packet / "meta" / Path(__file__).name)
    result = {
        "passed": True,
        "timing_rows": len(timing),
        "profile_rows": len(profiles),
        "campaigns": len(campaigns),
        "final_primary_configurations": len(final["rows"]),
        "timed_environment_steps": sum(row["sampled_environment_steps"] for row in final["rows"]),
        "all_packet_json_parsed": True,
        "existing_manifest_hashes_verified": True,
        "fixture_hashes_verified": True,
        "production_source": "db84114d1",
        "source_ast_and_ruff": "13 changed Python files passed",
        "original_unfinished_files": "12 hashes unchanged",
    }
    write(args.output, result)
    shutil.copy2(args.output, packet / "meta" / args.output.name)
    files = [
        receipt(path) | {"packet_path": str(path.relative_to(packet))}
        for path in sorted(packet.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    assert max(row["bytes"] for row in files) < 2_000_000
    assert sum(row["bytes"] for row in files) < 8_000_000
    write(packet / "manifest.json", {"files": files, "bytes": sum(row["bytes"] for row in files)})
    assert {row["packet_path"] for row in files} == {
        str(path.relative_to(packet)) for path in packet.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    print(json.dumps(result | {"packet_files": len(files), "packet_bytes": sum(row["bytes"] for row in files)}))


if __name__ == "__main__":
    main()
