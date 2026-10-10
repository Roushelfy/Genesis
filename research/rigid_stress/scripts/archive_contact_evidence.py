"""Archive completed contact-repair checkpoints with exact hashes and source labels."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def compact_arrays(value):
    """Keep configurations/records, summarize long per-environment numerical arrays."""
    if isinstance(value, dict):
        return {key: compact_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > 32 and all(isinstance(item, (int, float, bool)) for item in value):
            finite = [item for item in value if math.isfinite(item)]
            return {
                "array_length": len(value),
                "finite_count": len(finite),
                "minimum": min(finite) if finite else None,
                "maximum": max(finite) if finite else None,
                "mean": sum(finite) / len(finite) if finite else None,
                "maximum_indices": sorted(range(len(value)), key=lambda index: value[index], reverse=True)[:5],
            }
        if len(value) > 32 and value and all(isinstance(item, list) for item in value):
            widths = {len(item) for item in value}
            if len(widths) == 1 and next(iter(widths)) <= 16:
                return {"array_length": len(value), "columns": [compact_arrays(list(col)) for col in zip(*value)]}
        return [compact_arrays(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "development_surface_traction_trial": (
            "apex-diagnosis.log",
            "surface-traction-v3.patch",
            "traction-tests-v2.log",
            "traction-apex-v3.log",
            "traction-suite-v3.log",
            "baseline-live-b2048-v3.log",
            "surface-failure-v3.json",
            "surface-sliding-lp.json",
        ),
        "6494f7f7": (
            "local-apex-diagnosis.log",
            "local-sliding-diagnosis.log",
            "local-apex-centroid-diagnosis.log",
            "adaptive-tests-v1.log",
            "adaptive-suite-v1.log",
            "adaptive-cpu-v1.log",
            "baseline-live-b2048-adaptive-v1.json",
            "baseline-live-b2048-adaptive-v1.log",
            "baseline-profile-b1024-adaptive-v1.json",
            "baseline-profile-b1024-adaptive-v1.log",
            "baseline-audit-b2048-adaptive-v1.json",
            "baseline-audit-b2048-adaptive-v1.log",
        ),
        "e1fbf937": (
            "surface-inverse-probe-v1.json",
            "surface-inverse-probe-v1.log",
            "surface-suite-gpu-v1.log",
            "surface-suite-cpu-v1.log",
            "surface-comparison-gpu.txt",
            "surface-comparison-memory.csv",
            "full-live-b1024-v2.json",
            "full-live-b1024-v2.log",
            "surface-live-b1024-v1.json",
            "surface-live-b1024-v1.log",
            "surface-live-b2048-v1.json",
            "surface-live-b2048-v1.log",
        ),
        "e0b6714c": (
            "face-scatter-probe-v1.json",
            "face-scatter-probe-v1.log",
            "face-suite-gpu-v1.log",
            "face-audit-policy-b2048-v1.json",
            "face-audit-policy-b2048-v1.log",
            "face-comparison-gpu.txt",
            "serial-scatter-live-b1024-v1.json",
            "serial-scatter-live-b1024-v1.log",
            "face-live-b1024-v1.json",
            "face-live-b1024-v1.log",
            "face-live-b2048-v1.json",
            "face-live-b2048-v1.log",
            "face-seed623001-live-b4096-v1.json",
            "face-seed623001-live-b4096-v1.log",
            "face-live-b8192-v1.json",
            "face-live-b8192-v1.log",
            "face-profile-b1024-v1.json",
            "face-profile-b1024-v1.log",
        ),
    }
    artifacts = []
    for source, names in checkpoints.items():
        for name in names:
            runtime = args.run / name
            if not runtime.is_file():
                continue
            content = runtime.read_bytes()
            original_bytes = len(content)
            original_hash = hashlib.sha256(content).hexdigest()
            artifact = args.output / name
            format_name = "complete"
            if name.endswith(".json") and original_bytes > 500_000:
                result = compact_arrays(json.loads(content))
                result["publication_note"] = (
                    "Long environment arrays summarized. Exact original remains at runtime_path with original_sha256."
                )
                content = (json.dumps(result, indent=2) + "\n").encode()
                artifact = artifact.with_suffix(".summary.json")
                format_name = "summary"
                previous = (args.output / name).with_suffix(".json.gz")
                if previous.is_file():
                    previous.unlink()
            elif len(content) > 500_000:
                content = gzip.compress(content, mtime=0)
                artifact = artifact.with_suffix(artifact.suffix + ".gz")
            artifact.write_bytes(content)
            artifacts.append(
                {
                    "source_revision": source,
                    "runtime_path": str(runtime),
                    "artifact": artifact.name,
                    "format": format_name,
                    "original_sha256": original_hash,
                    "original_bytes": original_bytes,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
            )
    manifest = {
        "status": "Development checkpoints. Final repeated selection remains in progress.",
        "artifacts": artifacts,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
