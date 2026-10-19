"""Verify archived manifest payloads directly from a published Git tree."""

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

PREFIX = "research/rigid_stress/evidence/20261019-qualified-ranking/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tree = subprocess.check_output(["git", "rev-parse", f"{args.tree}^{{tree}}"], text=True).strip()
    listing = subprocess.check_output(["git", "ls-tree", "-r", "-z", tree])
    blobs = {}
    for entry in listing.split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            _, kind, oid = metadata.split()
            if kind == b"blob":
                blobs[name.decode()] = oid.decode()
    process = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def read_blob(name):
        if name not in blobs:
            raise ValueError(f"Manifest payload missing from Git tree: {name}")
        process.stdin.write((blobs[name] + "\n").encode())
        process.stdin.flush()
        _, kind, size = process.stdout.readline().split()
        if kind != b"blob":
            raise ValueError(f"Expected blob: {name}")
        data = process.stdout.read(int(size))
        if process.stdout.read(1) != b"\n":
            raise ValueError("Invalid cat-file framing")
        return data

    packets = []
    try:
        for name in sorted(blobs):
            if not name.startswith(PREFIX) or not name.endswith("/manifest.json"):
                continue
            if len(PurePosixPath(name[len(PREFIX) :]).parts) != 2:
                continue
            manifest = json.loads(read_blob(name))
            count = decoded_count = total = 0
            for row in manifest.get("files", []):
                if "copy" not in row:
                    continue
                relative = PurePosixPath(row["copy"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Invalid payload path: {relative}")
                payload_name = str(PurePosixPath(name).parent / relative)
                data = read_blob(payload_name)
                if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(f"Encoded size/hash mismatch: {payload_name}")
                if "original_sha256" in row:
                    encoding = row.get("encoding", "identity")
                    if encoding not in ("gzip", "identity"):
                        raise ValueError(f"Unsupported encoding: {encoding}")
                    original = gzip.decompress(data) if encoding == "gzip" else data
                    if (
                        len(original) != row["original_bytes"]
                        or hashlib.sha256(original).hexdigest() != row["original_sha256"]
                    ):
                        raise ValueError(f"Decoded size/hash mismatch: {payload_name}")
                    decoded_count += 1
                count += 1
                total += len(data)
            packets.append({"manifest": name, "payloads": count, "decoded_payloads": decoded_count, "bytes": total})
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError("git cat-file failed")
    report = {"tree": tree, "passed": True, "packets": packets, "payloads": sum(p["payloads"] for p in packets)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"tree": tree, "passed": True, "packets": len(packets), "payloads": report["payloads"]}))


if __name__ == "__main__":
    main()
