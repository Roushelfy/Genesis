"""Collect compact publication evidence using only the standard library."""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
import shutil
import statistics


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def receipt(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--registry", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    root = args.registry.parent.parent
    packet = args.output
    timing, profiles, campaigns = [], [], []
    for campaign, job in read(args.registry).items():
        directory = root / campaign
        item = {"campaign": campaign, "job": job, "artifacts": [], "return_codes": {}}
        source_path = directory / "source.json"
        if source_path.exists():
            source = read(source_path)
            item["source"] = {key: source[key] for key in ("revision", "patch_sha256", "source")}
            item["source_receipt"] = receipt(source_path)
        for name in ("tasks.json", "submission.json", "cancelled.json"):
            path = directory / name
            if path.exists():
                write(packet / "campaigns" / campaign / name, read(path))
        for path in sorted(directory.glob("group*/*")):
            if path.is_file():
                item["artifacts"].append(receipt(path))
                if path.name.endswith("returncode.txt"):
                    item["return_codes"][str(path.relative_to(directory))] = path.read_text().strip()
                    if path.read_text().strip() != "0":
                        log = path.with_name(path.name.removesuffix(".returncode.txt") + ".log")
                        if log.exists():
                            destination = packet / "failures" / campaign / path.parent.name / (log.stem + ".txt")
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            lines = log.read_text().splitlines()[-120:]
                            destination.write_text("\n".join(re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).rstrip()
                                                             for line in lines).rstrip() + "\n")
                if path.suffix:
                    continue
                raw = read(path)
                assert (path.parent / (path.name + ".returncode.txt")).read_text().strip() == "0"
                skip = {"native_buffers", "maximum_stress_Pa", "egg_position_m", "contacts_per_environment",
                        "candidate_count", "pressure_evaluations", "grid_samples_visited", "grid_samples_inside",
                        "complete_residual_N", "rhs_norm_N", "warmup_quality"}
                row = {key: value for key, value in raw.items() if key not in skip}
                row.update(campaign=campaign, task=str(path.relative_to(directory)), raw=receipt(path))
                row["final_positions_sha256"] = hashlib.sha256(json.dumps(raw["egg_position_m"]).encode()).hexdigest()
                if "repeats" in raw:
                    row["repeats"] = [{key: value for key, value in repeat.items() if key != "failed_steps_per_environment"}
                                      for repeat in raw["repeats"]]
                    samples = [1000 * repeat["seconds"] / repeat["steps"] for repeat in raw["repeats"]]
                    row["milliseconds_per_batch_step"] = samples
                    row["median_ms"] = statistics.median(samples)
                    row["range_ms"] = [min(samples), max(samples)]
                    row["median_environment_steps_per_second"] = raw["envs"] * 1000 / row["median_ms"]
                    row["invalid_environment_steps"] = sum(r["invalid_environment_steps"] for r in raw["repeats"])
                    timing.append(row)
                elif "stage_wall_ms" in raw:
                    row["fit_evaluations_histogram"] = dict(collections.Counter(raw["pressure_evaluations"]))
                    row["loaded_contacts"] = sum(raw["contacts_per_environment"])
                    # The original keys omitted captured CUDA graphs. Keep their values and interpretation together.
                    row["recorded_kernel_time_is_complete"] = "export QD_GRAPH=0" in (
                        directory / f"workload-{path.parent.name.removeprefix('group')}.sh"
                    ).read_text()
                    profiles.append(row)
            elif path.is_dir():
                for result in sorted(path.iterdir()):
                    if result.is_file() and result.suffix in (".json", ".npz", ".xml"):
                        item["artifacts"].append(receipt(result))
                        if result.suffix in (".json", ".xml") and result.name != "progress.json":
                            destination = packet / "results" / result.relative_to(root)
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(result, destination)
        campaigns.append(item)
    write(packet / "timing.json", {"rows": timing})
    write(packet / "profiles.json", {"rows": profiles,
          "note": "Intrusive stages are not additive scene costs. Graph-disabled diagnostic scheduling differs from production."})
    write(packet / "campaigns.json", {"data_root": str(root), "campaigns": campaigns})
    manifest = [receipt(path) | {"packet_path": str(path.relative_to(packet))}
                for path in sorted(packet.rglob("*")) if path.is_file() and path.name != "manifest.json"]
    write(packet / "manifest.json", {"files": manifest, "bytes": sum(row["bytes"] for row in manifest)})
    print(json.dumps({"timing_rows": len(timing), "profile_rows": len(profiles), "files": len(manifest),
                      "bytes": sum(row["bytes"] for row in manifest)}))


if __name__ == "__main__":
    main()
