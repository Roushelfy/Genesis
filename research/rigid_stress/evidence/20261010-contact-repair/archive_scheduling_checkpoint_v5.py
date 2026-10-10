"""Allocated single-writer publication of explicitly terminal scheduling jobs."""
import json
import os
import sys
from pathlib import Path

from research.rigid_stress.scripts import archive_completed_evidence as archive
from research.rigid_stress.scripts import summarize_benchmarks as summarize
from research.rigid_stress.scripts import verify_contact_evidence as verify

run = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "runs/20261010-contact-repair"
publication = Path("research/rigid_stress/evidence/20261010-contact-repair")
paths = sorted((run / "balance-policy-b49152-v2").glob("policy-*.json"))
assert len(paths) == 2
sys.argv = ["summarize", "--output", str(run / "balance-policy-b49152-v2-summary.json"), *map(str, paths)]
summarize.main()
prefixes = (
    "residual-layout-base-live-", "residual-layout-tiled-live-", "residual-layout-large-pair-",
    "wrench-reuse-crossover-", "packed-layout-crossover-", "packed-traversal-",
    "contact-moments-micro-", "contact-moments-full-oracle-", "contact-moments-probe-frozen-",
    "episode-live-b32-", "episode-policy-b32-", "episode-probe-frozen-",
    "native-wrench-block-full-oracle-", "native-wrench-block-overflow-substeps4-",
    "native-wrench-block-gpu-v1", "native-wrench-block-cpu-v2", "native-wrench-block-test-fix-",
    "native-wrench-block-tests-", "native-wrench-block-guard-v1", "native-wrench-block-guard-v2",
    "balance-policy-b49152-v2", "archive_scheduling_checkpoint_v5",
)
existing = {item["runtime_path"] for item in json.loads((publication / "manifest.json").read_text())["artifacts"]}
paths = sorted(path for path in run.rglob("*") if path.is_file() and path.relative_to(run).parts[0].startswith(prefixes))
names = [str(path.relative_to(run)) for path in paths if str(path) not in existing and path.suffix in (".json", ".log", ".csv", ".py", ".patch")]
assert len(names) > 50, len(names)
sys.argv = ["archive", "--run", str(run), "--output", str(publication), "--revision", "468f1d5f-scheduling-research-and-59b16811-native-candidate-checkpoints", *names]
archive.main()
sys.argv = ["verify", "--manifest", str(publication / "manifest.json"), "--originals", "--output", str(run / "scheduling-checkpoint-v5-evidence-verification.json")]
verify.main()
