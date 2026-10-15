"""Publish closed thickness diagnostics and partition-verified local meshes without accepting unfinished references."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from research.rigid_stress.scripts.publish_nonconvex_checkpoint import publish_bounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    p = args.runtime_root / "20261014-physical-validation"
    meta = p / f"thickness-local-diagnosis-publication-{args.version}"
    meta.mkdir(parents=True, exist_ok=False)
    jobs = {
        "egg-fixed-surface-four-level-audit-v1": "718556_0",
        "egg-thickness45-tensor-diagnosis-v1": "718569_0",
        "egg-thickness-local-bulk-meshes-v1": "718572_0",
    }
    query = ["sacct", "-n", "-P", "-X", "-j", ",".join(job.split("_")[0] for job in jobs.values())]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%30000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    identities, records = [], []
    extras = {meta / "jobs.txt", meta / "job-commands.txt"}
    for name, job in jobs.items():
        run = p / name
        assert states[job] == "COMPLETED" and int((run / "returncode.txt").read_text()) == 0
        for kind in ["source", "input-source"]:
            before = run / f"{kind}.sha256"
            if not before.exists():
                continue
            assert before.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in before.read_text().splitlines():
                digest, relative = line.split("  ", 1)
                actual = Path(relative) if Path(relative).is_absolute() else run / kind / relative
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == digest
                identities.append({"run": name, "kind": kind, "file": relative, "sha256": digest})
        extras.update(p.glob(f"*{job}.log"))
        records.append({"run": name, "job": job, "state": "COMPLETED", "returncode": 0})
    audit = json.loads((p / "egg-fixed-surface-four-level-audit-v1/results/summary.json").read_text())
    diagnosis = json.loads((p / "egg-thickness45-tensor-diagnosis-v1/results/summary.json").read_text())
    meshes = json.loads((p / "egg-thickness-local-bulk-meshes-v1/results/summary.json").read_text())
    assert audit["fields_verified"] == len(audit["rows"]) == 20 and not audit["physical_accuracy_accepted"]
    assert len(diagnosis["rows"]) == 5 and all(row["unchanged_probe_count"] == 36 for row in diagnosis["rows"])
    for path, digest in diagnosis["input_sha256"].items():
        path = Path(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        extras.add(path)
    assert [row["tetrahedra"] for row in meshes["rows"]] == [502230, 516287, 554574]
    for row in meshes["rows"][1:]:
        assert row["parent_partition"]["all_child_tetrahedra_contained_in_declared_parent"]
        assert row["parent_partition"]["each_parent_volume_preserved"]
    seed = p / "egg-fixed-surface-layers5-v1/mesh.npz"
    assert all(row["seed_sha256"] == hashlib.sha256(seed.read_bytes()).hexdigest() for row in meshes["rows"])
    extras.update([seed, seed.with_suffix(".json")])
    for name in [
        "thickness-four-level-audit-commands-v1.json",
        "thickness-four-level-closed-readonly-v1.json",
        "thickness45-tensor-diagnosis-commands-v1.json",
        "thickness-local-bulk-mesh-commands-v1.json",
        "frozen-traction-local-bulk-commands-v1.json",
        "u-recorded-history-qualification-commands-v1.json",
        "u-recorded-history-audit-commands-v1.json",
        "u-recorded-history-audit-commands-v2.json",
        "u-recorded-history-bit-identity-observation-v1.json",
    ]:
        extras.add(p / "meta" / name)
    summary = {
        "scope": __doc__,
        "closed_runs": records,
        "source_identity_records": identities,
        "twenty_independent_fixed_boundary_fields": audit,
        "original_probe_tensor_diagnosis": diagnosis,
        "partition_verified_local_bulk_meshes": meshes,
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    destination = args.destination_root / f"thickness-local-diagnosis-{args.version}"
    publish_bounded(
        args.runtime_root,
        destination,
        [p / name for name in jobs],
        sorted(extras),
        summary,
        "All20 original2/3/4/5-layer fields are independently checked, retaining3.18-4.52% last tensor changes. All original36 probes and raw maxima remain; five probe locations account for97-99% of squared tensor change, predominantly deviatoric. Three locally refined meshes preserve each parent tetrahedral domain/volume plus global area/volume/COM, but no fine solved field or pressure convergence is inferred. This packet preserves meshes, all180 diagnostic probe/component records, traction inputs and audit/source records; complete large original FEM fields remain at immutable hashed runtime paths for the separate full thickness publication, not silently claimed present here. Frozen-traction qualification/fine solves, recorded U history and its replacement audit are unfinished and excluded as successful evidence; live-prefix residual bit differences and an unstarted cancelled strict-identity audit remain explicit. Full physical/model/feedback/warmed-cost acceptance remains open.",
        p / "publication-staging",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.rigid_stress.scripts.verify_runtime_publication",
            "--manifest",
            str(destination / "manifest.json"),
            "--runtime-root",
            str(args.runtime_root),
            "--staging-root",
            str(p / "publication-staging" / destination.name),
            "--output",
            str(meta / "runtime-verification.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
