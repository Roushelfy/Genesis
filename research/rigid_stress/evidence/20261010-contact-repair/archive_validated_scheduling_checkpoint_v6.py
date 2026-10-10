"""Archive only named terminal v3 validation and v2 traversal controls."""
import json
import os
import sys
from pathlib import Path

from research.rigid_stress.scripts import archive_completed_evidence as archive
from research.rigid_stress.scripts import verify_contact_evidence as verify

run = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "runs/20261010-contact-repair"
publication = Path("research/rigid_stress/evidence/20261010-contact-repair")
existing = {item["runtime_path"] for item in json.loads((publication / "manifest.json").read_text())["artifacts"]}
names = [
    "native-wrench-block-residual-v3.patch",
    "native-wrench-block-residual-balance-v3.py",
    "native-wrench-block-residual-gpu-v3.log",
    "native-wrench-block-residual-cpu-v3.log",
    "packed-traversal-probe-frozen-v2.py",
    "scheduling-checkpoint-v5-evidence-verification.json",
    "scheduling-checkpoint-v5-analysis-archive.log",
    "archive_validated_scheduling_checkpoint_v6.py",
]
for basename in (
    "native-wrench-block-residual-oracle-seed510000-v3",
    "native-wrench-block-residual-oracle-seed623001-v3",
    "native-wrench-block-residual-overflow-substeps4-v3",
    "packed-traversal-control-b1024-v2",
    "packed-traversal-control-b32768-v2",
    "contact-moments-overflow-substeps4-v1",
    "contact-moments-base-live-b1024-v1",
    "contact-moments-reused-live-b1024-v1",
    "contact-moments-live-pair-b1024-v1-summary",
):
    names.extend(f"{basename}{suffix}" for suffix in (".json", ".log", ".variant.json") if (run / f"{basename}{suffix}").is_file())
names = [name for name in names if str(run / name) not in existing]
assert len(names) > 20, len(names)
sys.argv = ["archive", "--run", str(run), "--output", str(publication), "--revision", "468f1d5f-terminal-research-controls-and-59b16811-validated-native-v3", *names]
archive.main()
sys.argv = ["verify", "--manifest", str(publication / "manifest.json"), "--originals", "--output", str(run / "validated-scheduling-checkpoint-v6-evidence-verification.json")]
verify.main()
