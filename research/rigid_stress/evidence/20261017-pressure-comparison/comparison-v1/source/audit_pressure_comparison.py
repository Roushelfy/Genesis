"""Close the declared comparison cohorts, source identities and scheduler receipts."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PAIRED = {
    ("shared", "rigid"),
    ("pressure-default", "live"),
    ("intrinsic", "live"),
    ("shared", "policy-rigid"),
    ("pressure-default", "policy"),
    ("intrinsic", "policy"),
}
TUNED = {(profile.replace("intrinsic", "pressure-tuned"), scope) for profile, scope in PAIRED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    data = json.loads(args.metadata.read_text())
    assert data["passed_metadata_checks"] and not data["cohorts_still_incomplete"]
    expected = {
        name: PAIRED
        for name in ("preflight-b32-v1", "core-b2048-v1", "wide-b8192-v1", "core-b16384-v1", "wide-b32768-v1")
    }
    for width in (2048, 16384):
        expected[f"tuned-confirm-b{width}-v1"] = TUNED
    expected["tuning-b2048-v1"] = {("shared", "rigid")} | {
        (profile, "live")
        for profile in (
            "pressure-default",
            "pressure-scatter128",
            "pressure-scatter256",
            "pressure-fast",
            "pressure-tuned",
            "pressure-history4",
        )
    }
    for width in (32, 16384):
        for mode in ("max", "full"):
            expected[f"output-mode-b{width}-{mode}-v1"] = {("pressure-default", "live"), ("intrinsic", "live")}
    for width in (32, 2048, 16384, 32768):
        expected[f"memory-b{width}-v1"] = PAIRED
    expected["tuned-memory-b16384-v1"] = {
        (profile, scope) for profile in ("pressure-default", "pressure-tuned") for scope in ("live", "policy")
    }
    for width in (49152, 61440):
        expected[f"capacity-b{width}-v1"] = {("pressure-default", "live")}
    expected["trace-b2048-v1"] = {
        (profile, "live") for profile in ("pressure-default", "pressure-scatter128", "intrinsic")
    }
    expected["trace-pressure-unfused-v1"] = {("pressure-unfused", "live")}
    assert set(expected) == {row["cohort"] for row in data["records"]}
    for cohort, cases in expected.items():
        rows = [row for row in data["records"] if row["cohort"] == cohort]
        assert {(row["profile"], row["scope"]) for row in rows} == cases and len(rows) == len(cases), cohort
        assert all(row["complete_valid_rollout"] for row in rows), cohort
        summary = json.loads((args.root / cohort / "summary.json").read_text())
        assert summary["all_requested_cases_finished"] and summary["all_program_exits_zero"]
    assert len(data["records"]) == 91 and sum(r["headline_eligible"] for r in data["records"]) == 57
    failures = {(r["cohort"], r["classification"]) for r in data["failures"]}
    assert failures == {
        ("capacity-b65536-v1", "rigid_array_capacity_guard"),
        ("capacity-b131072-v1", "rigid_array_capacity_guard"),
        ("capacity-intrinsic-b40960-v1", "out_of_memory"),
    }
    assert all(row["successful_rollout_rate"] is None for row in data["failures"])
    contract = json.loads((args.root / "meta/contract-v1.json").read_text())
    snapshot = json.loads((args.root / "source-v1/snapshot.json").read_text())
    for row in snapshot["files"]:
        path = args.root / "source-v1" / row["path"]
        assert path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    assert (
        hashlib.sha256((args.root / "source-v1/snapshot-files.sha256").read_bytes()).hexdigest()
        == contract["benchmark_manifest_sha256"]
    )
    for path in args.root.glob("pressure-oracle-*/oracle.json"):
        assert (path.parent / "returncode.txt").read_text().strip() == "0"
    for name in ("egg-rlm-v1", "model-audit-v2", "reward-audit-v1", "negative-input-v1"):
        assert (args.root / name / "returncode.txt").read_text().strip() == "0"
    negative = json.loads((args.root / "negative-input-v1/results/summary.json").read_text())
    assert negative["classification"] == "continuous_load_incompatible" and negative["successful_rollout_rate"] is None
    # The experiments intentionally include three failed capacity jobs. Derived
    # rendering/aggregation retries are separate and cannot replace scientific jobs.
    successful = {
        "725602_0",
        "725608_0",
        "725608_1",
        "725609_0",
        "725611_0",
        "725611_1",
        "725611_2",
        "725611_3",
        "725615_0",
        "725615_1",
        "725618_0",
        "725619_0",
        "725619_1",
        "725619_2",
        "725620_0",
        "725620_1",
        "725624_0",
        "725624_1",
        "725624_2",
        "725624_3",
        "725625_0",
        "725634_0",
        "725634_1",
        "725634_2",
        "725634_3",
        "725638_0",
        "725646_0",
        "725646_1",
        "725667_0",
        "725668_0",
        "725671_0",
        "725671_1",
        "725672_0",
        "725674_0",
        "725674_1",
        "725674_2",
        "725679_0",
        "725682_0",
        "725687_0",
        "725693_0",
        "725708_0",
    }
    successful |= {"725715_0", "725716_0", "725726_0", "725727_0", "725728_0", "725729_0"}
    rejected = {"725626_0", "725626_1", "725670_0"}
    command = [
        "sacct",
        "-j",
        ",".join(sorted(successful | rejected)),
        "-X",
        "-n",
        "-P",
        "-o",
        "JobID,State,ExitCode,Start,End,ElapsedRaw,ReqTRES,NodeList",
    ]
    receipt = subprocess.check_output(command, text=True)
    scheduler = {}
    for line in receipt.splitlines():
        if not line.strip():
            continue
        job, state, code, start, end, elapsed, resources, node = line.split("|")[:8]
        scheduler[job] = {
            "state": state,
            "exit_code": code,
            "start": start,
            "end": end,
            "elapsed_seconds": int(elapsed),
            "requested_resources": resources,
            "node": node,
        }
    assert set(scheduler) == successful | rejected
    assert all(scheduler[job]["state"] == "COMPLETED" and scheduler[job]["exit_code"] == "0:0" for job in successful)
    assert all(scheduler[job]["state"] == "FAILED" and scheduler[job]["exit_code"] == "1:0" for job in rejected)
    result = {
        "scope": __doc__,
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "native_source_files_verified": len(snapshot["files"]),
        "closed_successful_benchmark_records": 91,
        "eligible_normal_records": 57,
        "preserved_capacity_failures": 3,
        "scheduler": scheduler,
        "requirements": {
            "literal_public_default": "default marker and complete option equality checked by metadata collector",
            "paired_four_scope_curve": "five widths, both models, all six processes per allocation",
            "tuning": "B2048 screen and B2048/B16384 default/tuned confirmation",
            "max_full": "matched live pressure/intrinsic, B32 and B16384, three repeats",
            "memory_capacity": "separate four-scope sampled memory and explicit successful/failed brackets",
            "profiles": "separate 600-step fused/default/tuned/intrinsic plus unfused pressure diagnosis",
            "accuracy": "16 egg cases, 28 qualified controls, four unsupported pressure inputs, original independent gates",
            "lifecycle": "four pressure oracle variants, 480 complete sampled fields and 128 partial resets; unchanged-engine prior qualification reused",
            "reward": "230400 complete recorded env-steps, separate whole/lift analysis, model/radius and threshold sensitivity",
            "failure_semantics": "current incompatible input and capacity failures retained; historical near-cancellation identity in archive index",
            "delivery": "report, guide, compact source/config/hardware/measurement packet; figure/export/source QA must close separately",
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".scheduler.tsv").write_text(receipt)


if __name__ == "__main__":
    main()
