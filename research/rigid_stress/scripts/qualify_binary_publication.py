"""Qualify exact NPZ reconstruction while keeping every published artifact below50MB."""

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path

from research.rigid_stress.scripts.publish_nonconvex_checkpoint import publish_bounded
from research.rigid_stress.scripts.verify_nonconvex_publication import verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.output / "runtime/fixture"
    run.mkdir(parents=True)
    header = b"{'descr': '|u1', 'fortran_order': False, 'shape': (67108864,), }"
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    with (
        zipfile.ZipFile(run / "weights.npz", "w", compression=zipfile.ZIP_STORED) as archive,
        archive.open("weights.npy", "w") as stream,
    ):
        stream.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
        for index in range(64):
            stream.write(hashlib.shake_256(f"binary-publication-{index}".encode()).digest(1024 * 1024))
    (run / "source.sha256").write_text("")
    destination = args.output / "publication"
    publish_bounded(
        args.output,
        destination,
        [run],
        [],
        {"scope": "Synthetic binary reconstruction fixture; no physical simulation evidence"},
        "Preserve exact binary bytes.",
        args.output / "staging",
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert len(manifest["summary"]["sharded_original_files"]) == 1
    assert max((destination / entry["artifact"]).stat().st_size for entry in manifest["files"]) < 50_000_000
    result = verify_manifest(destination / "manifest.json", True, {})
    assert result["passed"] and result["reconstructed_complete_files"] == 1
    (args.output / "summary.json").write_text(
        json.dumps(
            {"scope": __doc__, "verification": result, "synthetic_fixture_only": True, "goal_completion": False},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
