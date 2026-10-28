"""Retain automatic DPR receipts and original primary measurements in a compact packet."""

import argparse
import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import file_id


def stable_bytes(path):
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Artifact changed during packaging: {path}")
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run.resolve()
    ledger = json.loads(stable_bytes(args.ledger))
    primary = json.loads(stable_bytes(run / "meta/primary-throughput-v1-terminal-audit.json"))
    if not primary["complete"] or not primary["passed"] or primary["finished_rows"] != 102:
        raise ValueError("Require all 102 original primary processes and their completed audit.")
    if len(ledger["requirements"]) != 22 or not all(row["satisfied"] for row in ledger["requirements"]):
        raise ValueError("Require the reviewed 22-clause ledger.")

    campaigns = sorted(path.parent for path in run.glob("*/tasks.json"))
    jobs = {
        campaign.name: str(json.loads(stable_bytes(campaign / "submission.json"))["job"]).split(";")[0]
        for campaign in campaigns
        if (campaign / "submission.json").is_file()
    }
    accounting = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-j",
            ",".join(sorted(set(jobs.values()))),
            "-n",
            "-P",
            "--format=JobID,State,ExitCode,Elapsed",
        ],
        text=True,
    )
    states = {fields[0]: fields[1:] for line in accounting.splitlines() if (fields := line.split("|"))}
    terminal = {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
    for campaign, job in jobs.items():
        selected = {key: values for key, values in states.items() if key == job or key.startswith(job + "_")}
        if not selected or any(
            values[0] not in terminal and not values[0].startswith("CANCELLED") for values in selected.values()
        ):
            raise ValueError(f"Campaign has missing or nonterminal accounting: {campaign}: {selected}")

    required = {args.ledger.resolve()}
    for requirement in ledger["requirements"]:
        required.update((run / name).resolve() for name in requirement["evidence"])
    paths = set(required)
    paths.update(path for path in (run / "meta").rglob("*") if path.is_file() and path.suffix in (".json", ".py"))
    paths.add(Path(__file__).resolve())
    tasks = []
    for campaign in campaigns:
        paths.update(
            path for path in campaign.iterdir() if path.is_file() and path.suffix in (".json", ".patch", ".sh")
        )
        source = json.loads(stable_bytes(campaign / "source.json")) if (campaign / "source.json").is_file() else {}
        groups = json.loads(stable_bytes(campaign / "tasks.json"))
        for group, definitions in enumerate(groups):
            for definition in definitions:
                folder = campaign / f"group{group}"
                name = definition["name"]
                markers = [folder / f"{name}.returncode.txt", folder / f"{name}.returncode"]
                marker = next((path for path in markers if path.is_file()), None)
                code = int(stable_bytes(marker).strip()) if marker else None
                record = {
                    "campaign": campaign.name,
                    "group": group,
                    "task": definition,
                    "source": source.get("source"),
                    "revision": source.get("revision"),
                    "job": jobs.get(campaign.name),
                    "returncode": code,
                    "scope": "Task execution receipt. Numerical qualification is in the requirement ledger.",
                }
                if marker:
                    paths.add(marker)
                worker = campaign / "analysis-source" / (definition["module"].replace(".", "/") + ".py")
                if worker.is_file():
                    paths.add(worker)
                    record["captured_worker"] = file_id(worker)
                output = folder / name
                candidates = (
                    [output / "result.json", output / "start.json", output / "junit.xml"]
                    if output.is_dir()
                    else [output]
                )
                for path in candidates:
                    if path.is_file():
                        paths.add(path)
                        if code not in (0, None):
                            required.add(path)
                if campaign.name == "primary-throughput-v1":
                    for path in (output / "result.json", output / "benchmark.json"):
                        if path.is_file():
                            paths.add(path)
                            required.add(path)
                log = folder / f"{name}.log"
                if log.is_file():
                    record["external_log"] = file_id(log)
                tasks.append(record)

    payloads, artifacts, omitted = {}, [], []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        content = stable_bytes(path)
        identity = {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        compressed = gzip.compress(content, compresslevel=9, mtime=0)
        keep = path in required or path.is_relative_to(run / "meta") or len(compressed) <= 128 * 1024
        if keep:
            target = "payloads/" + identity["sha256"] + path.suffix + ".gz"
            payloads[target] = compressed
            identity["payload"] = target
        else:
            omitted.append({"path": str(path), "reason": "Large optional diagnostic remains at its hashed data path."})
        artifacts.append(identity)

    index = {
        "schema": "genesis.automatic-dpr-evidence-packet.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run),
        "measured_source": primary["source_revision"],
        "primary_processes": 102,
        "campaigns": len(campaigns),
        "tasks": tasks,
        "artifacts": artifacts,
        "omitted_payloads": omitted,
        "slurm_accounting": accounting,
        "scope": "Exact compressed metadata, requirement evidence and all original primary benchmark/result JSON. Full fields, histories, arrays and logs remain on data; their identities in retained audits keep their original validation scope. This is not an exhaustive archive of every runtime file.",
    }
    payloads["index.json.gz"] = gzip.compress((json.dumps(index, indent=2) + "\n").encode(), compresslevel=9, mtime=0)
    payloads["requirements-ledger.json"] = stable_bytes(args.ledger)
    payloads["README.md"] = (
        "# Automatic DPR fast path and defaults: evidence packet\n\n"
        f"Measured source: `{primary['source_revision']}`. Run root: `{run}`.\n\n"
        "`requirements-ledger.json` records all 22 decisions and their limits. "
        "`index.json.gz` maps original paths, source pins, captured workers, tasks, terminal accounting and payloads. "
        "Each payload is the exact original file compressed with gzip. Its filename is the uncompressed SHA256. "
        "All 102 primary receipts and every available original benchmark are retained, including failures. "
        "Complete field arrays and large optional histories remain at their hashed data paths. "
        "Intermediate failures and audits retain their original incomplete/negative status.\n\n"
        "Verify this packet with `sha256sum -c SHA256SUMS`. "
        "Decode the index with `gzip -dc index.json.gz`; then decode a selected payload with `gzip -dc payloads/<hash>.<extension>.gz`. "
        "For scientific reproduction, use the task's frozen engine revision and captured analysis worker, "
        "the original argument/input identities, Slurm rtx-mid / image 1_26 and a fresh data output. "
        "Set `QD_GRAPH=0` before imports and reuse the recorded data-side environment and private caches. "
        "Primary paired blocks run serially on one allocated GPU each. Profiles and memory sampling have separate jobs. "
        "The results report gives commands and the measurement scope.\n"
    ).encode()
    payloads["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in sorted(payloads.items())
    ).encode()
    total = sum(map(len, payloads.values()))
    if total > 10 * 1024 * 1024:
        raise ValueError(f"Packet exceeds 10 MiB: {total} bytes. No output was written.")
    args.output.mkdir(parents=True)
    for name, content in payloads.items():
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": total,
                "files": len(payloads),
                "campaigns": len(campaigns),
                "tasks": len(tasks),
            }
        )
    )


if __name__ == "__main__":
    main()
