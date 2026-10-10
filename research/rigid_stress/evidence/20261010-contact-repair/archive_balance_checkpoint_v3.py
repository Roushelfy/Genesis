"""Allocated analysis and single-writer archival of terminal balance jobs."""
import json
import os
import sys
from pathlib import Path

from research.rigid_stress.scripts import archive_completed_evidence as archive
from research.rigid_stress.scripts import summarize_benchmarks as summarize
from research.rigid_stress.scripts import verify_contact_evidence as verify

run = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "runs/20261010-contact-repair"
publication = Path("research/rigid_stress/evidence/20261010-contact-repair")

groups = {
    "native-balance-large-pair-v2-summary": sorted(run.glob("native-balance-*-b32768-seed623001-v2/live-*.json")),
    "balance-small-all-v2-summary": sorted(run.glob("balance-small-all-seed*-v2/*-b*-seed*.json")),
    "wrench-reuse-small-pair-v1-summary": [run / f"wrench-reuse-{kind}-live-b1024-v1.json" for kind in ("base", "reused")],
}
for label, paths in groups.items():
    assert len(paths) == (16 if "small-all" in label else 2), (label, paths)
    sys.argv = ["summarize", "--output", str(run / (label + ".json")), *map(str, paths)]
    summarize.main()

prefixes = (
    "native-balance-serial-b", "native-balance-t8-b", "native-balance-small-pairs-v2-summary",
    "native-balance-large-pair-v2-summary", "native-balance-large-gpu-v2",
    "balance-small-all-", "balance-small-audits-v2-summary", "balance-audit-", "balance-profile-",
    "balanced-launch-", "archive_balance_checkpoint_v3", "native-balance-tests-styled-v3",
)
existing = {item["runtime_path"] for item in json.loads((publication / "manifest.json").read_text())["artifacts"]}
paths = sorted(path for path in run.rglob("*") if path.is_file() and path.relative_to(run).parts[0].startswith(prefixes))
names = [str(path.relative_to(run)) for path in paths if str(path) not in existing and path.suffix in (".json", ".log", ".csv", ".py", ".patch")]
assert len(names) > 50, len(names)
sys.argv = ["archive", "--run", str(run), "--output", str(publication), "--revision", "e6094ff9-native-balance-t8-v2-and-styled-v3", *names]
archive.main()
sys.argv = ["verify", "--manifest", str(publication / "manifest.json"), "--originals", "--output", str(run / "balance-checkpoint-v3-evidence-verification.json")]
verify.main()
